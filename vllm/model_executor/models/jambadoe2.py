# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Jamba model."""
from collections.abc import Iterable
from typing import Any, Optional, Union
import re
import torch
from torch import nn

from transformers import PretrainedConfig

from vllm.attention.layer import Attention, AttentionType

from vllm.config import (CacheConfig, ModelConfig, VllmConfig,
                         get_current_vllm_config)
from vllm.distributed import get_tensor_model_parallel_world_size, divide
from vllm.distributed.parallel_state import get_pp_group, get_ep_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_mixer import MambaMixer
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2, extra_groups_for_head_shards
from vllm.model_executor.layers.mamba.mamba2_metadata import (
    Mamba2Metadata, prepare_mamba2_metadata)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.mamba_cache import (MambaCacheManager,
                                                    MambaCacheParams)
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput
from vllm.utils import LayerBlockType
from vllm.transformers_utils.configs import JambaDoEConfig

from .interfaces import (HasInnerState, IsHybrid, SupportsLoRA, SupportsPP,
                         SupportsV0Only)
from .utils import (PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

# Copyright 2024 AI21 Labs Ltd. and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Jamba model configuration"""

import math


class JambaDoEMoE(nn.Module):

    def __init__(
        self,
        config: JambaDoEConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        # enable_eplb: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor

        self.ep_group = get_ep_group().device_group
        self.ep_rank = self.ep_group.rank()
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        if config.hidden_act != "silu" and config.hidden_act != "swiglu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. "
                             "Only silu is supported for now.")

        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.n_routed_experts,
                                     bias=False,
                                     quant_config=None,
                                     params_dtype=torch.bfloat16,
                                     prefix=f"{prefix}.gate")

        self.shared_experts_gate = ReplicatedLinear(config.hidden_size,
                                     config.n_shared_experts,
                                     bias=False,
                                     quant_config=None,
                                     params_dtype=torch.bfloat16,
                                     prefix=f"{prefix}.shared_experts_gate")
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts))
        else:
            self.gate.e_score_correction_bias = None

        # Load balancing settings.
        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        # self.enable_eplb = enable_eplb

        self.n_redundant_experts = parallel_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = (self.n_logical_experts +
                                   self.n_redundant_experts)
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = (self.ep_rank *
                                      self.n_local_physical_experts)
        self.physical_expert_end = (self.physical_expert_start +
                                    self.n_local_physical_experts)

        self.experts = FusedMoE(            
                            num_experts=config.n_routed_experts,
                            top_k=config.num_experts_per_tok,
                            hidden_size=config.hidden_size,
                            intermediate_size=config.moe_intermediate_size,
                            reduce_results=True,
                            renormalize=True,
                            use_grouped_topk=False,
                            quant_config=quant_config,
                            params_dtype=torch.bfloat16,
                            scoring_func=config.scoring_func,
                            prefix=f"{prefix}.experts")
            

        if config.n_shared_experts is not None:
            intermediate_size = (config.moe_intermediate_size *
                                 config.n_shared_experts)
            self.shared_experts = JambaDoEMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=self.experts.must_reduce_shared_expert_outputs(
                ),
                prefix=f"{prefix}.shared_experts",
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.n_shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
            if self.shared_experts_gate is not None:
                shared_output = torch.nn.functional.sigmoid(
                    self.shared_experts_gate(hidden_states)[0]) * shared_output
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)

        if hidden_states.dtype != torch.float16:
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits) * self.routed_scaling_factor
        else:
            # Fix FP16 overflow
            # See DeepseekV2DecoderLayer for more details.
            final_hidden_states = self.experts(hidden_states=hidden_states,
                                               router_logits=router_logits)
        if shared_output is not None:
            if hidden_states.dtype != torch.float16:
                final_hidden_states = final_hidden_states + shared_output
            else:
                # Fix FP16 overflow
                # See DeepseekV2DecoderLayer for more details.
                final_hidden_states = final_hidden_states + shared_output \
                    * (1. / self.routed_scaling_factor)


        return final_hidden_states.view(num_tokens, hidden_dim)


class JambaDoEMambaDecoderLayer(nn.Module):

    def __init__(self,
                 config: JambaDoEConfig,
                 layer_idx: int,
                 prefix: str = "",
                 # model_config: ModelConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 is_lora_enabled: Optional[bool] = False,
                 **kwargs) -> None:
        super().__init__()
        self.config = config
        self.is_lora_enabled = is_lora_enabled

        self.mamba = MambaMixer2(hidden_size= config.hidden_size,
                                ssm_state_size = config.mamba_d_state,
                                conv_kernel_size = config.mamba_d_conv,
                                intermediate_size = config.mamba_expand *\
                                                    config.hidden_size,
                                use_conv_bias = config.mamba_conv_bias,
                                use_bias = config.mamba_proj_bias,
                                n_groups=config.mamba_n_groups,
                                num_heads=config.mamba_n_heads,
                                head_dim=config.mamba_d_head,
                                rms_norm_eps=config.rms_norm_eps,
                                activation=config.hidden_act,
                                quant_config=quant_config,
                                prefix=f"{prefix}.mixer",
                                params_dtype=torch.bfloat16,
                                chunk_size=config.mamba_chunk_size,
                                )

        num_experts = config.layers_num_experts[layer_idx]
        ffn_layer_class = JambaDoEMoE if num_experts > 1 else JambaDoEMLP
        self.mlp = ffn_layer_class(config,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=torch.bfloat16)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size,
                                        eps=config.rms_norm_eps,
                                        dtype=torch.bfloat16)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        mamba_cache_params: MambaCacheParams,
        mamba_metadata: Mamba2Metadata,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.mamba(hidden_states, mamba_cache_params, mamba_metadata)
        # Fully Connected
        hidden_states, residual = self.pre_ff_layernorm(
            hidden_states, residual)

        hidden_states = self.mlp(hidden_states)

        # hidden_states = hidden_states + residual

        return hidden_states, residual


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class JambaDoEMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config,
                                           reduce_results=reduce_results,
                                           prefix=f"{prefix}.down_proj")
        if hidden_act != "silu" and hidden_act != "swiglu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x

class JambaDoEMLAAttention(nn.Module):
    """
    For more info see MLACommonImpl in: vllm/attention/backends/mla/utils.py
    """

    def __init__(
        self,
        config: JambaDoEConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(self.hidden_size,
                                             self.q_lora_rank,
                                             bias=False,
                                             quant_config=quant_config,
                                             params_dtype=torch.bfloat16,
                                             prefix=f"{prefix}.q_a_proj")

            self.q_b_proj = ColumnParallelLinear(q_lora_rank,
                                                 self.num_heads *
                                                 self.qk_head_dim,
                                                 bias=False,
                                                 quant_config=quant_config,
                                                 params_dtype=torch.bfloat16,
                                                 prefix=f"{prefix}.q_b_proj")
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.num_heads *
                                               self.qk_head_dim,
                                               bias=False,
                                               quant_config=quant_config,
                                               params_dtype=torch.bfloat16,
                                               prefix=f"{prefix}.q_proj")

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            params_dtype=torch.bfloat16,
            prefix=f"{prefix}.kv_a_proj_with_mqa")

        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            params_dtype=torch.bfloat16,
            prefix=f"{prefix}.kv_b_proj")
        self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim,
                                        self.hidden_size,
                                        bias=False,
                                        quant_config=quant_config,
                                        params_dtype=torch.bfloat16,
                                        prefix=f"{prefix}.o_proj")

        if rope_scaling:
            rope_scaling["rope_type"] = 'deepseek_yarn'
        self.rotary_emb = get_rope(qk_rope_head_dim,
                                   rotary_dim=qk_rope_head_dim,
                                   max_position=max_position_embeddings,
                                   base=rope_theta,
                                   rope_scaling=rope_scaling,
                                   is_neox_style=False)
        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        self.attn = Attention(self.num_local_heads,
                              self.qk_head_dim,
                              self.scaling,
                              num_kv_heads=self.num_local_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")


    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            # q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads,
                                         self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(-1, self.num_local_heads,
                                                   self.qk_head_dim)
        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim],
                               dim=-1)
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_a, _ = latent_cache.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = kv_a.contiguous()
        # kv_a = self.kv_a_layernorm(kv_a.contiguous())
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads,
                     self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = latent_cache[:, :, self.kv_lora_rank:]

        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        q[..., self.qk_nope_head_dim:] = q_pe
        k = torch.empty_like(q)
        k[..., :self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim:] = k_pe
        # padding value to qk_head_dim for alignment
        v = torch.nn.functional.pad(
            v, [0, self.qk_head_dim - self.v_head_dim],
            value=0).view(-1, self.num_local_heads * self.qk_head_dim)
        attn_output = self.attn(q, k, v)
        attn_output = attn_output.view(
            -1, self.num_local_heads,
            self.qk_head_dim)[..., :self.v_head_dim].reshape(
                -1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output, kv_a


class JambaDoEKnowledgeBlockLayer(nn.Module):
    """
    """
    def __init__(
        self,
        config: JambaDoEConfig,
        hidden_size: int,
        num_heads: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0


        self.knowledge_block_heads_num = config.knowledge_block_heads_num
        self.knowledge_block_heads_dim = config.knowledge_block_heads_dim
        self.knowledge_block_fields_num = config.knowledge_block_fields_num
 
        self.hidden_size = hidden_size
        self.scaling = self.knowledge_block_heads_dim**-0.5
        self.num_local_heads = self.knowledge_block_heads_num // tp_size
        
        self.query_projection_size = self.knowledge_block_heads_dim * self.knowledge_block_heads_num

        self.kv_lora_rank = config.kv_lora_rank
        
        # Layer norm
        self.kn_layernorm = RMSNorm(
            hidden_size=self.kv_lora_rank,
            eps=config.rms_norm_eps,
            dtype=torch.bfloat16,
        )
        # Input.
        self.kn_up_proj = ColumnParallelLinear(self.kv_lora_rank,
                                               self.query_projection_size,
                                               bias=False,
                                               quant_config=quant_config,
                                               params_dtype=torch.bfloat16,
                                               prefix=f"{prefix}.kn_up_proj")
        

        #Core att
        self.core_attention = Attention(
                            self.num_local_heads,
                            self.knowledge_block_heads_dim,
                            self.scaling,
                            num_kv_heads=self.num_local_heads,
                            cache_config=cache_config,
                            quant_config=quant_config,
                            attn_type=AttentionType.ENCODER_DECODER,
                            prefix=f"{prefix}.kn_block_attn")


        # Output.
        self.kn_att_out_proj = RowParallelLinear(self.query_projection_size,
                                        self.hidden_size,
                                        bias=False,
                                        quant_config=quant_config,
                                        params_dtype=torch.bfloat16,
                                        prefix=f"{prefix}.kn_out_proj")
        

        # Key matrix
        self.key_matrix = torch.nn.Parameter(
            torch.empty((self.knowledge_block_fields_num, self.knowledge_block_heads_num, self.knowledge_block_heads_dim), dtype=torch.bfloat16)
        )

        # Value matrix
        self.value_matrix = torch.nn.Parameter(
            torch.empty((self.knowledge_block_fields_num, self.knowledge_block_heads_num, self.knowledge_block_heads_dim), dtype=torch.bfloat16)
        )

    def get_query_key_value_tensors(self, query_input):
        """
        Derives `query` tensor from `hidden_states`, and `key`/`value` tensors
        from `key_value_states`.
        """

        query, _ = self.kn_up_proj(self.kn_layernorm(query_input))

        new_tensor_shape_query = query.size()[:-1] + (
            self.num_local_heads,
            self.knowledge_block_heads_dim,
        )
    
        query = query.view(*new_tensor_shape_query).contiguous()

        new_tensor_shape_kv = (
            self.knowledge_block_fields_num,
            self.knowledge_block_heads_num,
            self.knowledge_block_heads_dim,
        )
        
        key = self.key_matrix.view(*new_tensor_shape_kv).contiguous()

        value = self.value_matrix.view(*new_tensor_shape_kv).contiguous()

        return query, key, value

    def forward(
        self,
        # positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        
        query, key, value = self.get_query_key_value_tensors(hidden_states)

        attn_out = self.core_attention(
            query,
            key,
            value,
            output_shape=(hidden_states.shape[0],
                          self.hidden_size))
        return self.kn_att_out_proj(attn_out)[0]

class JambaDoEAttentionDecoderLayer(nn.Module):
    def __init__(
        self,
        config: JambaDoEConfig,
        layer_idx: int,
        prefix: str = "",
        # model_config: ModelConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        is_lora_enabled: Optional[bool] = False,
        **kwargs
        # enable_eplb: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep='.')[-1])
        self.layer_idx = layer_idx
        self.conv1d = torch.nn.Conv1d(
                    in_channels=config.hidden_size,
                    out_channels=config.hidden_size,
                    #   * self.config.n_heads,

                    bias=False,
                    kernel_size=config.conv_attention_kernel_size,
                    # groups=self.config.d_model,
                    groups = config.num_attention_heads,
                    padding=config.conv_attention_kernel_size - 1,
                    device=torch.cuda.current_device(),
                    dtype=torch.bfloat16,
                )
        attn_cls = JambaDoEMLAAttention
        self.self_attn = attn_cls(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank
            if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        self.knowledge_attn = JambaDoEKnowledgeBlockLayer(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_kn_attn",
        )

        if config.n_routed_experts is not None:

            self.mlp = JambaDoEMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",

            )
        else:
            self.mlp = JambaDoEMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=torch.bfloat16,)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=torch.bfloat16,)
        self.routed_scaling_factor = config.routed_scaling_factor

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
           
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states

        if len(hidden_states.shape) == 3:
            hidden_states = self.conv1d(hidden_states.permute(1, 2, 0))[:, :,:-(self.config.conv_attention_kernel_size - 1)].permute(2, 0, 1).contiguous()

        else:
            forward_context = get_forward_context()
            num_prefills = forward_context.attn_metadata.num_prefills
            hidden_states = hidden_states.view(-1, num_prefills, self.hidden_size)
            hidden_states = self.conv1d(hidden_states.permute(1, 2, 0))[:, :, :-(self.config.conv_attention_kernel_size - 1)].permute(2, 0, 1).contiguous().view(-1, self.hidden_size)

        hidden_states = self.input_layernorm(hidden_states)


        hidden_states, kv_cache = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )


        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        knowledge_output = self.knowledge_attn(kv_cache)

        hidden_states = hidden_states + knowledge_output

        hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states + residual

        return hidden_states, None
    


ALL_DECODER_LAYER_TYPES = {
    "attention": JambaDoEAttentionDecoderLayer,
    "mamba": JambaDoEMambaDecoderLayer
}


class JambaDoEModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: JambaDoEConfig = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        # lora_config = vllm_config.lora_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.org_vocab_size = config.vocab_size
        print("### config.vocab_size: {}\n".format(config.vocab_size))
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                # padding_size=0,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        extra_kwargs = {"is_lora_enabled": bool(vllm_config.lora_config)}

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            layer_class = ALL_DECODER_LAYER_TYPES["attention"] if layer_idx % 4 == 3 else ALL_DECODER_LAYER_TYPES["mamba"]
            return layer_class(config,
                               layer_idx,
                               cache_config=cache_config,
                               quant_config=quant_config,
                               prefix=prefix,
                               **extra_kwargs)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers")
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        if get_pp_group().is_last_rank:
            self.final_layernorm = RMSNorm(config.hidden_size,
                                        eps=config.rms_norm_eps,
                                        dtype=torch.bfloat16,)
        else:
            self.final_layernorm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mamba_cache_params: MambaCacheParams,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:


        attn_metadata = get_forward_context().attn_metadata
        mamba2_metadata = prepare_mamba2_metadata(
                    chunk_size=self.config.mamba_chunk_size,
                    attn_metadata=attn_metadata,
                )

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
                
        kv_cache_index = 0
        mamba_cache_index = 0
        for layer in self.layers[self.start_layer:self.end_layer]:
            layer_mamba_cache_params = None
            if isinstance(layer, JambaDoEAttentionDecoderLayer):
                kv_cache_index += 1

                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    )
                
            if isinstance(layer, JambaDoEMambaDecoderLayer):
                current_state_layer = mamba_cache_index
                layer_mamba_cache_params = mamba_cache_params.at_layer_idx(
                    current_state_layer)
                mamba_cache_index += 1
                
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    mamba_cache_params=layer_mamba_cache_params,
                    mamba_metadata=mamba2_metadata,
                    )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.final_layernorm(hidden_states, residual)
        return hidden_states


class JambaDoEForCausalLM(nn.Module, HasInnerState, SupportsLoRA, SupportsPP,
                       IsHybrid, SupportsV0Only):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        lora_config = vllm_config.lora_config
        scheduler_config = vllm_config.scheduler_config
        assert not cache_config.enable_prefix_caching, \
            "Jamba currently does not support prefix caching"

        super().__init__()
        self.config = config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = scheduler_config
        self.model = JambaDoEModel(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "decoder"))
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE
                # We need bigger padding if using lora for kernel
                # compatibility
                if not lora_config else lora_config.lora_vocab_padding_size,
            )
        else:
            self.lm_head = PPMissingLayer()
        # Used to track and store by the Mamba cache between steps.
        self.mamba_cache: Optional[MambaCacheManager] = None

        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs):
        if self.mamba_cache is None:
            num_mamba_layers = self.model_config.get_num_layers_by_block_type(
                self.vllm_config.parallel_config, LayerBlockType.mamba)
            self.mamba_cache = MambaCacheManager(
                self.vllm_config, torch.bfloat16, num_mamba_layers,
                *self._get_mamba_cache_shape())

        mamba_cache_params = self.mamba_cache.current_run_tensors(**kwargs)

        hidden_states = self.model(input_ids, positions, mamba_cache_params,
                                   intermediate_tensors, inputs_embeds)
        return hidden_states

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def _get_mamba_cache_shape(
            self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Calculate shapes for Mamba's convolutional and state caches.
        
        Returns:
            Tuple containing:
            - conv_state_shape: Shape for convolutional state cache
            - temporal_state_shape: Shape for state space model cache
        """
        world_size = get_tensor_model_parallel_world_size()

        intermediate_size = self.config.mamba_expand * self.config.hidden_size

        # Extend groups if needed to ensure all groups needed by a head
        # are sharded together

        # if n_groups is not divisible by world_size, need to extend the shards
        # to ensure all groups needed by a head is sharded along with it
        n_groups = (self.config.mamba_n_groups + extra_groups_for_head_shards(
            self.config.mamba_n_groups, world_size))

        # Calculate conv state shape (includes groups)
        # - heads and n_groups are TP-ed
        conv_dim = (intermediate_size +
                    2 * n_groups * self.config.mamba_d_state)
        conv_state_shape = (
            divide(conv_dim, world_size),
            self.config.mamba_d_conv - 1,
        )

        # Calculate temporal state shape (per-head states)
        # These are not TP-ed as they depend on A, dt_bias, D
        # - they are typically small
        #   e.g., (h_heads, d_head, d_state) = (128, 64, 128)
        temporal_state_shape = (
            divide(divide(intermediate_size, self.config.mamba_d_head),
                   world_size),
            self.config.mamba_d_head,
            self.config.mamba_d_state,
        )

        return conv_state_shape, temporal_state_shape

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        layer_num_pattern = r'\.layers\.(\d+)\.'
        exp1_weight_pattern = r'linear_fc1.weight(\d+)$'
        exp2_weight_pattern = r'linear_fc2.weight(\d+)$'
        for name, loaded_weight in weights:
            layer_match = re.search(layer_num_pattern, name)
            if layer_match:
                layer_id = int(layer_match.group(1))
                if layer_id < self.model.start_layer or layer_id >= self.model.end_layer:
                    continue


            if "embedding.word_embeddings.weight" in name:
                if not get_pp_group().is_first_rank:
                    continue
                name = name.replace("embedding.word_embeddings.weight", "model.embed_tokens.weight")

            if "decoder.final_norm.weight" in name:
                if not get_pp_group().is_last_rank:
                    continue
                name = name.replace("decoder.final_norm.weight", "model.final_layernorm.weight")
                
            if "output_layer.weight" in name:
                if not get_pp_group().is_last_rank:
                    continue
                name = name.replace("output_layer.weight", "lm_head.weight")
                

            if name.startswith("decoder"):
                name = name.replace("decoder", "model")

            if "mixer" in name:
                name = name.replace("mixer", "mamba")
            
            if "A_log" in name:
                name = name.replace("A_log", "A")

            if "mlp" in name:
                if "router" in name:
                    name = name.replace("router", "gate")
                if "shared_experts" in name:
                    if "shared_experts.gate_weight" in name:
                        name = name.replace("shared_experts.gate_weight", "shared_experts_gate.weight")
                    elif "linear_fc1" in name:
                        name = name.replace("linear_fc1", "gate_up_proj")
                    elif "linear_fc2" in name:
                        name = name.replace("linear_fc2", "down_proj")

                elif "linear_fc1" in name or "linear_fc2" in name:
                    if "linear_fc1" in name:
                        match = re.search(exp1_weight_pattern, name)
                        expert_id = int(match.group(1))
                        shard_id = "w3"
                        name = name[:-len(match.group(1))].replace("linear_fc1.weight", "w13_weight")
                    elif "linear_fc2" in name:
                        match = re.search(exp2_weight_pattern, name)
                        expert_id = int(match.group(1))
                        shard_id = "w2"
                        name = name[:-len(match.group(1))].replace("linear_fc2.weight", "w2_weight")

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    loaded_params.add(name)
                    continue

            

            if "self_attention" in name:
                name = name.replace("self_attention", "self_attn")

                if "linear_q_down_proj" in name:
                    name = name.replace("linear_q_down_proj", "q_a_proj")

                elif "linear_q_up_proj" in name:
                    name = name.replace("linear_q_up_proj", "q_b_proj")

                elif "linear_kv_down_proj" in name:
                    name = name.replace("linear_kv_down_proj", "kv_a_proj_with_mqa")

                elif "linear_kv_up_proj" in name:
                    name = name.replace("linear_kv_up_proj", "kv_b_proj")

                elif "linear_proj" in name:
                    name = name.replace("linear_proj", "o_proj")

            if "knowledge_attention" in name:
                name = name.replace("knowledge_attention", "knowledge_attn")

            if "pre_mlp_layernorm" in name:
                if layer_id % 4 != 3:
                    name = name.replace("pre_mlp_layernorm", "pre_ff_layernorm")
                else:
                    name = name.replace("pre_mlp_layernorm", "post_attention_layernorm")


            if ".norm.weight" in name and not _is_mamba_layer(name):
                name = name.replace(".norm.weight", ".input_layernorm.weight")
            
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


def _is_moe_layer(name: str):
    return any(
        [experts_name in name for experts_name in [
            "experts",
            "router",
        ]])


def _is_mamba_layer(name: str):
    return any(
        [mamba_name in name for mamba_name in [
            "mamba",
            "mixer",
        ]])

