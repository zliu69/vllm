# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Jamba model."""
from collections.abc import Iterable
from typing import Any, Optional, Union
import re
import torch
from torch import nn
# from transformers import JambaConfig
from transformers import PretrainedConfig

from vllm.attention.layer import Attention, AttentionType
# from vllm.config import CacheConfig, VllmConfig
from vllm.config import (CacheConfig, ModelConfig, VllmConfig,
                         get_current_vllm_config)
from vllm.distributed import get_tensor_model_parallel_world_size, divide
from vllm.distributed.parallel_state import get_pp_group, get_ep_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
# from vllm.model_executor.layers.linear import (QKVParallelLinear,
#                                                ReplicatedLinear,
#                                                RowParallelLinear)
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


# coding=utf-8
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


# class JambaDoEConfig(PretrainedConfig):
#     r"""
#     This is the configuration class to store the configuration of a [`JambaModel`]. It is used to instantiate a
#     Jamba model according to the specified arguments, defining the model architecture. Instantiating a configuration
#     with the defaults will yield a similar configuration to that of the Jamba-v0.1 model.

#     [ai21labs/Jamba-v0.1](https://huggingface.co/ai21labs/Jamba-v0.1)

#     Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
#     documentation from [`PretrainedConfig`] for more information.


#     Args:
#         vocab_size (`int`, *optional*, defaults to 65536):
#             Vocabulary size of the Jamba model. Defines the number of different tokens that can be represented by the
#             `inputs_ids` passed when calling [`JambaModel`]
#         tie_word_embeddings (`bool`, *optional*, defaults to `False`):
#             Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
#             model has a output word embedding layer.
#         hidden_size (`int`, *optional*, defaults to 4096):
#             Dimension of the hidden representations.
#         intermediate_size (`int`, *optional*, defaults to 14336):
#             Dimension of the MLP representations.
#         num_hidden_layers (`int`, *optional*, defaults to 32):
#             Number of hidden layers in the Transformer encoder.
#         num_attention_heads (`int`, *optional*, defaults to 32):
#             Number of attention heads for each attention layer in the Transformer encoder.
#         num_key_value_heads (`int`, *optional*, defaults to 8):
#             This is the number of key_value heads that should be used to implement Grouped Query Attention. If
#             `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
#             `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
#             converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
#             by meanpooling all the original heads within that group. For more details checkout [this
#             paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `8`.
#         hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
#             The non-linear activation function (function or string) in the decoder.
#         initializer_range (`float`, *optional*, defaults to 0.02):
#             The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
#         rms_norm_eps (`float`, *optional*, defaults to 1e-06):
#             The epsilon used by the rms normalization layers.
#         use_cache (`bool`, *optional*, defaults to `True`):
#             Whether or not the model should return the last key/values attentions (not used by all models). Only
#             relevant if `config.is_decoder=True`.
#         num_logits_to_keep (`int` or `None`, *optional*, defaults to 1):
#             Number of prompt logits to calculate during generation. If `None`, all logits will be calculated. If an
#             integer value, only last `num_logits_to_keep` logits will be calculated. Default is 1 because only the
#             logits of the last prompt token are needed for generation. For long sequences, the logits for the entire
#             sequence may use a lot of memory so, setting `num_logits_to_keep=1` will reduce memory footprint
#             significantly.
#         output_router_logits (`bool`, *optional*, defaults to `False`):
#             Whether or not the router logits should be returned by the model. Enabling this will also
#             allow the model to output the auxiliary loss. See [here]() for more details
#         router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
#             The aux loss factor for the total loss.
#         pad_token_id (`int`, *optional*, defaults to 0):
#             The id of the padding token.
#         bos_token_id (`int`, *optional*, defaults to 1):
#             The id of the "beginning-of-sequence" token.
#         eos_token_id (`int`, *optional*, defaults to 2):
#             The id of the "end-of-sequence" token.
#         sliding_window (`int`, *optional*):
#             Sliding window attention window size. If not specified, will default to `None`.
#         max_position_embeddings (`int`, *optional*, defaults to 262144):
#             This value doesn't have any real effect. The maximum sequence length that this model is intended to be
#             used with. It can be used with longer sequences, but performance may degrade.
#         attention_dropout (`float`, *optional*, defaults to 0.0):
#             The dropout ratio for the attention probabilities.
#         num_experts_per_tok (`int`, *optional*, defaults to 2):
#             The number of experts to root per-token, can be also interpreted as the `top-p` routing
#             parameter
#         num_experts (`int`, *optional*, defaults to 16):
#             Number of experts per Sparse MLP layer.
#         expert_layer_period (`int`, *optional*, defaults to 2):
#             Once in this many layers, we will have an expert layer
#         expert_layer_offset (`int`, *optional*, defaults to 1):
#             The first layer index that contains an expert mlp layer
#         attn_layer_period (`int`, *optional*, defaults to 8):
#             Once in this many layers, we will have a vanilla attention layer
#         attn_layer_offset (`int`, *optional*, defaults to 4):
#             The first layer index that contains a vanilla attention mlp layer
#         use_mamba_kernels (`bool`, *optional*, defaults to `True`):
#             Flag indicating whether or not to use the fast mamba kernels. These are available only if `mamba-ssm` and
#             `causal-conv1d` are installed, and the mamba modules are running on a CUDA device. Raises ValueError if
#             `True` and kernels are not available
#         mamba_d_state (`int`, *optional*, defaults to 16):
#             The dimension the mamba state space latents
#         mamba_d_conv (`int`, *optional*, defaults to 4):
#             The size of the mamba convolution kernel
#         mamba_expand (`int`, *optional*, defaults to 2):
#             Expanding factor (relative to hidden_size) used to determine the mamba intermediate size
#         mamba_dt_rank (`Union[int,str]`, *optional*, defaults to `"auto"`):
#             Rank of the mamba discretization projection matrix. `"auto"` means that it will default to `math.ceil(self.hidden_size / 16)`
#         mamba_conv_bias (`bool`, *optional*, defaults to `True`):
#             Flag indicating whether or not to use bias in the convolution layer of the mamba mixer block.
#         mamba_proj_bias (`bool`, *optional*, defaults to `False`):
#             Flag indicating whether or not to use bias in the input and output projections (["in_proj", "out_proj"]) of the mamba mixer block

#     """

#     model_type = "jamba_doe"
#     keys_to_ignore_at_inference = ["past_key_values"]

#     def __init__(
#         self,
#         vocab_size=151936,
#         tie_word_embeddings=False,
#         hidden_size=4096,
#         intermediate_size=8192,
#         num_hidden_layers=30,
#         num_attention_heads=32,
#         num_key_value_heads=32,
#         hidden_act="silu",
#         initializer_range=0.006,
#         rms_norm_eps=1e-5,
#         use_cache=True,
#         num_logits_to_keep=1,
#         output_router_logits=False,
#         router_aux_loss_coef=0.001,
#         pad_token_id=151643,
#         bos_token_id=151644,
#         eos_token_id=151645,
#         sliding_window=None,
#         max_position_embeddings=131072,
#         rope_theta=10000,
#         rope_scaling=None,
#         attention_dropout=0.0,
#         num_experts_per_tok=2,
#         num_experts=16,
#         n_routed_experts=16,
#         n_shared_experts=1,
#         routed_scaling_factor=1.0,
#         topk_method="greedy",
#         moe_intermediate_size=8192,
#         conv_attention_kernel_size=4,
#         norm_topk_prob=True,
#         n_group=1,
#         topk_group=1,
#         scoring_func="sigmoid",
#         expert_layer_period=1,
#         expert_layer_offset=0,
#         attn_layer_period=4,
#         attn_layer_offset=3,
#         use_mamba_kernels=True,
#         mamba_d_state=128,
#         mamba_d_conv=4,
#         mamba_expand=2,
#         mamba_n_groups=8,
#         mamba_n_heads=128,
#         mamba_d_head=64,
#         mamba_dt_rank="auto",
#         mamba_conv_bias=True,
#         mamba_proj_bias=False,
#         qk_nope_head_dim=128,
#         qk_rope_head_dim=64,
#         v_head_dim=128,
#         q_lora_rank=512,
#         kv_lora_rank=512,
#         knowledge_block_fields_num=64,
#         knowledge_block_heads_num=32,
#         knowledge_block_heads_dim=128,
#         params_dtype='bf16',
#         moe_router_dtype='fp32',
#         **kwargs,
#     ):
#         self.vocab_size = vocab_size
#         self.tie_word_embeddings = tie_word_embeddings
#         self.hidden_size = hidden_size
#         self.intermediate_size = intermediate_size
#         self.num_hidden_layers = num_hidden_layers
#         self.num_attention_heads = num_attention_heads
#         self.sliding_window = sliding_window
#         self.max_position_embeddings = max_position_embeddings
#         self.rope_theta = rope_theta
#         self.rope_scaling = rope_scaling
#         self.attention_dropout = attention_dropout

#         # for backward compatibility
#         if num_key_value_heads is None:
#             num_key_value_heads = num_attention_heads

#         self.num_key_value_heads = num_key_value_heads
#         self.hidden_act = hidden_act
#         self.initializer_range = initializer_range
#         self.rms_norm_eps = rms_norm_eps

#         self.use_cache = use_cache
#         self.num_logits_to_keep = num_logits_to_keep
#         self.output_router_logits = output_router_logits
#         self.router_aux_loss_coef = router_aux_loss_coef

#         self.num_experts_per_tok = num_experts_per_tok
#         self.num_experts = num_experts
#         self.n_routed_experts = n_routed_experts
#         self.n_shared_experts = n_shared_experts
#         self.routed_scaling_factor = routed_scaling_factor
#         self.topk_method = topk_method
#         self.moe_intermediate_size = moe_intermediate_size
#         self.conv_attention_kernel_size = conv_attention_kernel_size
#         self.norm_topk_prob = norm_topk_prob
#         self.n_group = n_group
#         self.topk_group = topk_group
#         self.scoring_func = scoring_func
#         self.expert_layer_period = expert_layer_period
#         self.expert_layer_offset = expert_layer_offset
#         self.attn_layer_period = attn_layer_period
#         self.attn_layer_offset = attn_layer_offset

#         self._check_supported_offset("attention", self.attn_layer_period, self.attn_layer_offset)
#         self._check_supported_offset("expert", self.expert_layer_period, self.expert_layer_offset)

#         self.use_mamba_kernels = use_mamba_kernels
#         self.mamba_d_state = mamba_d_state
#         self.mamba_d_conv = mamba_d_conv
#         self.mamba_expand = mamba_expand
#         self.mamba_n_groups = mamba_n_groups
#         self.mamba_n_heads = mamba_n_heads
#         self.mamba_d_head = mamba_d_head
#         assert self.mamba_n_heads == int(self.mamba_expand * self.hidden_size) // self.mamba_d_head
#         self.mamba_dt_rank = math.ceil(self.hidden_size / 16) if mamba_dt_rank == "auto" else mamba_dt_rank
#         self.mamba_conv_bias = mamba_conv_bias
#         self.mamba_proj_bias = mamba_proj_bias
#         self.qk_nope_head_dim = qk_nope_head_dim
#         self.qk_rope_head_dim = qk_rope_head_dim
#         self.v_head_dim = v_head_dim
#         self.q_lora_rank = q_lora_rank
#         self.kv_lora_rank = kv_lora_rank
#         self.knowledge_block_fields_num=knowledge_block_fields_num
#         self.knowledge_block_heads_num=knowledge_block_heads_num
#         self.knowledge_block_heads_dim=knowledge_block_heads_dim
#         if params_dtype == "fp16" or params_dtype == "bf16" or params_dtype == "fp32":
#             self.params_dtype = params_dtype
#         # elif params_dtype == "bf16":
#         #     self.params_dtype = torch.bfloat16
#         # elif params_dtype == "fp32":
#         #     self.params_dtype = torch.float32
#         else:
#             raise ValueError(f"Unsupported params_dtype: {params_dtype}. "
#                              "Only fp16, fp32, bf16 is supported for now.")

#         if moe_router_dtype == "fp16" or moe_router_dtype == "bf16" or moe_router_dtype == "fp32":
#             self.moe_router_dtype = moe_router_dtype
#         # elif moe_router_dtype == "bf16":
#         #     self.moe_router_dtype = torch.bfloat16
#         # elif moe_router_dtype == "fp32":
#         #     self.moe_router_dtype = torch.float32
#         else:
#             raise ValueError(f"Unsupported moe_router_dtype: {moe_router_dtype}. "
#                              "Only fp16, fp32, bf16 is supported for now.")

#         super().__init__(
#             pad_token_id=pad_token_id,
#             bos_token_id=bos_token_id,
#             eos_token_id=eos_token_id,
#             tie_word_embeddings=tie_word_embeddings,
#             **kwargs,
#         )

#     @property
#     def layers_block_type(self):
#         return [
#             "attention" if i % self.attn_layer_period == self.attn_layer_offset else "mamba"
#             for i in range(self.num_hidden_layers)
#         ]

#     @property
#     def layers_num_experts(self):
#         return [
#             self.num_experts if i % self.expert_layer_period == self.expert_layer_offset else 1
#             for i in range(self.num_hidden_layers)
#         ]

#     def _check_supported_offset(self, property_: str, period: int, offset: int):
#         if offset >= period:
#             raise ValueError(
#                 f"{property_} layer offset ({offset}) must be smaller than {property_} layer period ({period})"
#             )



# class JambaDoEMoE(nn.Module):

#     def __init__(self,
#                  config: JambaDoEConfig,
#                  num_experts: Optional[int] = None,
#                  top_k: Optional[int] = None,
#                  params_dtype: Optional[torch.dtype] = None,
#                  tp_size: Optional[int] = None,
#                  quant_config: Optional[QuantizationConfig] = None,
#                  prefix: str = ""):
#         super().__init__()
#         self.num_total_experts = num_experts or config.num_experts
#         self.top_k = top_k or config.num_experts_per_tok
#         self.hidden_size = config.hidden_size
#         self.intermediate_size = config.intermediate_size

#         if self.num_total_experts > 1:
#             self.router = ReplicatedLinear(self.hidden_size,
#                                            self.num_total_experts,
#                                            bias=False,
#                                            quant_config=None,
#                                            params_dtype=params_dtype)

#         self.experts = FusedMoE(self.num_total_experts,
#                                 self.top_k,
#                                 self.hidden_size,
#                                 self.intermediate_size,
#                                 tp_size=tp_size,
#                                 params_dtype=params_dtype,
#                                 reduce_results=True,
#                                 renormalize=False,
#                                 use_grouped_topk=False,
#                                 quant_config=quant_config,
#                                 prefix=f"{prefix}.experts")

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         orig_shape = hidden_states.shape
#         hidden_states = hidden_states.view(-1, self.hidden_size)
#         # router_logits: (batch * sequence_length, n_experts)
#         if self.num_total_experts > 1:
#             router_logits, _ = self.router(hidden_states)
#         else:
#             router_logits = torch.ones((hidden_states.shape[0], 1),
#                                        device=hidden_states.device,
#                                        dtype=hidden_states.dtype)
#         hidden_states = self.experts(hidden_states, router_logits)
#         return hidden_states.view(orig_shape)


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
        # if config.moe_router_dtype == "bf16":
        #     self.params_dtype = torch.bfloat16
        # elif params_dtype == "fp32":
        #     self.params_dtype = torch.float32
        # self.moe_router_dtype
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
                            # tp_size=tp_size,
                            # params_dtype=params_dtype,
                            reduce_results=True,
                            renormalize=True,
                            use_grouped_topk=False,
                            quant_config=quant_config,
                            params_dtype=torch.bfloat16,
                            scoring_func=config.scoring_func,
                            prefix=f"{prefix}.experts")
        
        
        # FusedMoE(
        #     num_experts=config.n_routed_experts,
        #     top_k=config.num_experts_per_tok,
        #     hidden_size=config.hidden_size,
        #     intermediate_size=config.moe_intermediate_size,
        #     reduce_results=False,
        #     renormalize=config.norm_topk_prob,
        #     quant_config=quant_config,
        #     use_grouped_topk=False,
        #     num_expert_group=config.n_group,
        #     topk_group=config.topk_group,
        #     prefix=f"{prefix}.experts",
        #     scoring_func=config.scoring_func,
        #     e_score_correction_bias=self.gate.e_score_correction_bias,
        #     enable_eplb=self.enable_eplb,
        #     num_redundant_experts=self.n_redundant_experts)

            

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

        # if self.tp_size > 1:
        #     final_hidden_states = (
        #         self.experts.maybe_all_reduce_tensor_model_parallel(
        #             final_hidden_states))

        return final_hidden_states.view(num_tokens, hidden_dim)


# class JambaDoEMLP(JambaDoEMoE):

#     def __init__(self,
#                  config: JambaDoEConfig,
#                  params_dtype: Optional[torch.dtype] = None,
#                  tp_size: Optional[int] = None,
#                  quant_config: Optional[QuantizationConfig] = None,
#                  prefix: str = ""):
#         super().__init__(config,
#                          num_experts=1,
#                          top_k=1,
#                          params_dtype=torch.bfloat16,
#                          tp_size=tp_size,
#                          quant_config=quant_config,
#                          prefix=prefix)


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
        # self.mamba = MambaMixer(hidden_size= config.hidden_size,
        #                         ssm_state_size = config.mamba_d_state,
        #                         conv_kernel_size = config.mamba_d_conv,
        #                         intermediate_size = config.mamba_expand *\
        #                                             config.hidden_size,
        #                         time_step_rank = config.mamba_dt_rank,
        #                         use_conv_bias = config.mamba_conv_bias,
        #                         use_bias = config.mamba_proj_bias,
        #                         use_rms_norm=True,
        #                         rms_norm_eps=config.rms_norm_eps,
        #                         activation=config.hidden_act,
        #                         is_lora_enabled = self.is_lora_enabled
        #                         )

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


# class DeepseekV2Attention(nn.Module):

#     def __init__(
#         self,
#         config: PretrainedConfig,
#         hidden_size: int,
#         num_heads: int,
#         qk_nope_head_dim: int,
#         qk_rope_head_dim: int,
#         v_head_dim: int,
#         q_lora_rank: int,
#         kv_lora_rank: int,
#         rope_theta: float = 10000,
#         rope_scaling: Optional[dict[str, Any]] = None,
#         max_position_embeddings: int = 8192,
#         cache_config: Optional[CacheConfig] = None,
#         quant_config: Optional[QuantizationConfig] = None,
#         prefix: str = "",
#     ) -> None:
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.qk_nope_head_dim = qk_nope_head_dim
#         self.qk_rope_head_dim = qk_rope_head_dim
#         self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
#         self.v_head_dim = v_head_dim
#         self.q_lora_rank = q_lora_rank
#         self.kv_lora_rank = kv_lora_rank
#         self.num_heads = num_heads
#         tp_size = get_tensor_model_parallel_world_size()
#         assert num_heads % tp_size == 0
#         self.num_local_heads = num_heads // tp_size
#         self.scaling = self.qk_head_dim**-0.5
#         self.rope_theta = rope_theta
#         self.max_position_embeddings = max_position_embeddings

#         if self.q_lora_rank is not None:
#             self.q_a_proj = ReplicatedLinear(self.hidden_size,
#                                              self.q_lora_rank,
#                                              bias=False,
#                                              quant_config=quant_config,
#                                              prefix=f"{prefix}.q_a_proj")
#             self.q_a_layernorm = RMSNorm(self.q_lora_rank,
#                                          eps=config.rms_norm_eps)
#             self.q_b_proj = ColumnParallelLinear(q_lora_rank,
#                                                  self.num_heads *
#                                                  self.qk_head_dim,
#                                                  bias=False,
#                                                  quant_config=quant_config,
#                                                  prefix=f"{prefix}.q_b_proj")
#         else:
#             self.q_proj = ColumnParallelLinear(self.hidden_size,
#                                                self.num_heads *
#                                                self.qk_head_dim,
#                                                bias=False,
#                                                quant_config=quant_config,
#                                                prefix=f"{prefix}.q_proj")

#         self.kv_a_proj_with_mqa = ReplicatedLinear(
#             self.hidden_size,
#             self.kv_lora_rank + self.qk_rope_head_dim,
#             bias=False,
#             quant_config=quant_config,
#             prefix=f"{prefix}.kv_a_proj_with_mqa")
#         self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
#                                       eps=config.rms_norm_eps)
#         self.kv_b_proj = ColumnParallelLinear(
#             self.kv_lora_rank,
#             self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
#             bias=False,
#             quant_config=quant_config,
#             prefix=f"{prefix}.kv_b_proj")
#         # O projection.
#         self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim,
#                                         self.hidden_size,
#                                         bias=False,
#                                         quant_config=quant_config,
#                                         prefix=f"{prefix}.o_proj")
#         if rope_scaling:
#             rope_scaling["rope_type"] = 'deepseek_yarn'

#         self.rotary_emb = get_rope(qk_rope_head_dim,
#                                    rotary_dim=qk_rope_head_dim,
#                                    max_position=max_position_embeddings,
#                                    base=rope_theta,
#                                    rope_scaling=rope_scaling,
#                                    is_neox_style=False)

#         if rope_scaling:
#             mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
#             scaling_factor = rope_scaling["factor"]
#             mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
#             self.scaling = self.scaling * mscale * mscale

#         self.attn = Attention(self.num_local_heads,
#                               self.qk_head_dim,
#                               self.scaling,
#                               num_kv_heads=self.num_local_heads,
#                               cache_config=cache_config,
#                               quant_config=quant_config,
#                               prefix=f"{prefix}.attn")

#     def forward(
#         self,
#         positions: torch.Tensor,
#         hidden_states: torch.Tensor,
#     ) -> torch.Tensor:
#         if self.q_lora_rank is not None:
#             q = self.q_a_proj(hidden_states)[0]
#             q = self.q_a_layernorm(q)
#             q = self.q_b_proj(q)[0].view(-1, self.num_local_heads,
#                                          self.qk_head_dim)
#         else:
#             q = self.q_proj(hidden_states)[0].view(-1, self.num_local_heads,
#                                                    self.qk_head_dim)
#         q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim],
#                                dim=-1)
#         latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
#         kv_a, _ = latent_cache.split(
#             [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
#         latent_cache = latent_cache.unsqueeze(1)
#         kv_a = self.kv_a_layernorm(kv_a.contiguous())
#         kv = self.kv_b_proj(kv_a)[0]
#         kv = kv.view(-1, self.num_local_heads,
#                      self.qk_nope_head_dim + self.v_head_dim)
#         k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
#         k_pe = latent_cache[:, :, self.kv_lora_rank:]

#         q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

#         q[..., self.qk_nope_head_dim:] = q_pe
#         k = torch.empty_like(q)
#         k[..., :self.qk_nope_head_dim] = k_nope
#         k[..., self.qk_nope_head_dim:] = k_pe
#         # padding value to qk_head_dim for alignment
#         v = torch.nn.functional.pad(
#             v, [0, self.qk_head_dim - self.v_head_dim],
#             value=0).view(-1, self.num_local_heads * self.qk_head_dim)
#         attn_output = self.attn(q, k, v)
#         attn_output = attn_output.view(
#             -1, self.num_local_heads,
#             self.qk_head_dim)[..., :self.v_head_dim].reshape(
#                 -1, self.num_local_heads * self.v_head_dim)
#         output, _ = self.o_proj(attn_output)
#         return output
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
            # self.q_a_layernorm = RMSNorm(self.q_lora_rank,
            #                              eps=config.rms_norm_eps,
            #                              dtype=torch.bfloat16,)
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
        # self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
        #                               eps=config.rms_norm_eps,
        #                               dtype=torch.bfloat16,)
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

        # In the MLA backend, kv_cache includes both k_c and
        # pe (i.e. decoupled position embeddings). In particular,
        # the concat_and_cache_mla op requires
        #     k_c.size(1) + k_pe.size(1) == kv_cache.size(2)
        # i.e.
        #     kv_lora_rank + qk_rope_head_dim == head_size
        # self.mla_attn = Attention(
        #     num_heads=self.num_local_heads,
        #     head_size=self.kv_lora_rank + self.qk_rope_head_dim,
        #     scale=self.scaling,
        #     num_kv_heads=self.num_local_heads,
        #     cache_config=cache_config,
        #     quant_config=quant_config,
        #     # prefix=f"{prefix}.attn",
        #     # use_mla=True,
        #     # # MLA Args
        #     # q_lora_rank=self.q_lora_rank,
        #     # kv_lora_rank=self.kv_lora_rank,
        #     # qk_nope_head_dim=self.qk_nope_head_dim,
        #     # qk_rope_head_dim=self.qk_rope_head_dim,
        #     # qk_head_dim=self.qk_head_dim,
        #     # v_head_dim=self.v_head_dim,
        #     # kv_b_proj=self.kv_b_proj,
        # )
        self.attn = Attention(self.num_local_heads,
                              self.qk_head_dim,
                              self.scaling,
                              num_kv_heads=self.num_local_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

        # self.prefix = prefix
        # self.debug_layer_idx = int(self.prefix.split(".")[-2])

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
        # self.num_query_groups = config.knowledge_block_heads_num
        self.hidden_size = hidden_size
        self.scaling = self.knowledge_block_heads_dim**-0.5
        self.num_local_heads = self.knowledge_block_heads_num // tp_size
        # self.config.num_query_groups = None
        # print("### kn att after self.config.num_attention_heads:{}\nself.config.kv_channels :{}\n".format(self.config.num_attention_heads, self.config.kv_channels ))
        # self.layer_number = layer_number
        # self.attn_mask_type = attn_mask_type
        # self.attention_type = attention_type
        # assert self.config.knowledge_block_freq is not None and self.config.knowledge_block_freq > 0
        # assert self.config.knowledge_block_heads_dim is not None and self.config.knowledge_block_heads_dim > 0
        # assert self.config.knowledge_block_heads_num is not None and self.config.knowledge_block_heads_num > 0

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so these two will be the same
        self.query_projection_size = self.knowledge_block_heads_dim * self.knowledge_block_heads_num
        # self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups
        self.kv_lora_rank = config.kv_lora_rank
        # Per attention head and per partition values.
        # world_size = parallel_state.get_tensor_model_parallel_world_size()
        # self.tensor_model_parallel_size = world_size
        # self.rank = parallel_state.get_tensor_model_parallel_rank()
        # self.hidden_size_per_attention_head = divide(
        #     self.query_projection_size, self.config.knowledge_block_heads_num
        # )
        # self.num_attention_heads_per_partition = divide(self.config.knowledge_block_heads_num, world_size)
        # self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        
        # self.checkpoint_core_attention = self.config.recompute_granularity == 'selective'
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
        # print("### kn att Core att self.config:{}\n self.config.num_attention_heads:{}\nself.config.kv_channels :{}\n".format(self.config, self.config.num_attention_heads, self.config.kv_channels ))
        self.core_attention = Attention(
                            self.num_local_heads,
                            self.knowledge_block_heads_dim,
                            self.scaling,
                            num_kv_heads=self.num_local_heads,
                            cache_config=cache_config,
                            quant_config=quant_config,
                            attn_type=AttentionType.ENCODER_DECODER,
                            is_kn_att=True,
                            prefix=f"{prefix}.kn_block_attn")
        
        
        
        
        # build_module(
        #     submodules.core_attention,
        #     config=self.config,
        #     layer_number=self.layer_number,
        #     attn_mask_type=self.attn_mask_type,
        #     attention_type=self.attention_type,
        #     cp_comm_type=cp_comm_type,
        # )


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
        # if self.config.perform_initialization:
        #     self.config.init_method(self.key_matrix)
        # self.key_matrix.data = self.key_matrix.data.to(dtype=self.config.params_dtype)

        # Value matrix
        self.value_matrix = torch.nn.Parameter(
            torch.empty((self.knowledge_block_fields_num, self.knowledge_block_heads_num, self.knowledge_block_heads_dim), dtype=torch.bfloat16)
        )
        # if self.config.perform_initialization:
        #     self.config.init_method(self.value_matrix)
        # self.value_matrix.data = self.value_matrix.data.to(dtype=self.config.params_dtype)

    def get_query_key_value_tensors(self, query_input):
        """
        Derives `query` tensor from `hidden_states`, and `key`/`value` tensors
        from `key_value_states`.
        """
        # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
        # Attention heads [s, D] --> [s, D]
        # bs = query_input.size(1)
        
        # print("### KnowledgeAttention get_query_key_value_tensors before kn_up_proj query_input: {}\n query_input shape: {},\n".format(query_input, query_input.shape))

        query, _ = self.kn_up_proj(self.kn_layernorm(query_input))

        # print("### KnowledgeAttention get_query_key_value_tensors after kn_up_proj query: {}\n query shape: {},\n".format(query, query.shape))

        new_tensor_shape_query = query.size()[:-1] + (
            self.num_local_heads,
            self.knowledge_block_heads_dim,
        )
    
        query = query.view(*new_tensor_shape_query).contiguous()
        # print("### KnowledgeAttention get_query_key_value_tensors after reshape query: {}\n query shape: {},\n".format(query, query.shape))

        new_tensor_shape_kv = (
            self.knowledge_block_fields_num,
            # bs,
            self.knowledge_block_heads_num,
            self.knowledge_block_heads_dim,
        )
        
        # print("### KnowledgeAttention self.key_matrix: {}\n self.key_matrix: {},\n".format(self.key_matrix, self.key_matrix.shape))
        # if self.tensor_model_parallel_size > 1:
        #     key = self.key_matrix[:,self.rank * self.num_attention_heads_per_partition : self.rank * self.num_attention_heads_per_partition + self.num_attention_heads_per_partition,:].unsqueeze(1).expand(-1, bs, -1, -1).view(*new_tensor_shape_kv).contiguous()

        #     # print("### KnowledgeAttention get_query_key_value_tensors key: {}\n key shape: {},\n".format(key, key.shape))

        #     value = self.value_matrix[:,self.rank * self.num_attention_heads_per_partition : self.rank * self.num_attention_heads_per_partition + self.num_attention_heads_per_partition,:].unsqueeze(1).expand(-1, bs, -1, -1).view(*new_tensor_shape_kv).contiguous()

        # else:
        # if bs > 1:
        key = self.key_matrix.view(*new_tensor_shape_kv).contiguous()

        # print("### KnowledgeAttention get_query_key_value_tensors key: {}\n key shape: {},\n".format(key, key.shape))

        value = self.value_matrix.view(*new_tensor_shape_kv).contiguous()
        # else:
        #     key = self.key_matrix.unsqueeze(1).expand(-1, bs, -1, -1).view(*new_tensor_shape_kv).contiguous().clone()

        #     # print("### KnowledgeAttention get_query_key_value_tensors key: {}\n key shape: {},\n".format(key, key.shape))

        #     value = self.value_matrix.unsqueeze(1).expand(-1, bs, -1, -1).view(*new_tensor_shape_kv).contiguous().clone()


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
        self.conv1d_cache = torch.zeros([config.conv_attention_kernel_size - 1, config.hidden_size], device=torch.cuda.current_device(), dtype=torch.bfloat16)
        # if model_config.use_mla:
        attn_cls = JambaDoEMLAAttention
        # else:
        #     attn_cls = DeepseekV2Attention
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
                # and layer_idx >= config.first_k_dense_replace
                # and layer_idx % config.moe_layer_freq == 0):
            self.mlp = JambaDoEMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                # enable_eplb=enable_eplb,
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

        forward_context = get_forward_context()
        # num_prefills = forward_context.attn_metadata.num_prefills
        num_decode_tokens = forward_context.attn_metadata.num_decode_tokens
        print("### rank:{} JambaDoEAttentionDecoderLayer forward input shape: {}, num_decode_tokens: {}\n".format(torch.distributed.get_rank(), hidden_states.shape, num_decode_tokens))
        
        if num_decode_tokens > 0:
            # decoding
            conv1d_input = torch.cat([self.conv1d_cache, hidden_states], dim=0)
            conv1d_input = conv1d_input.view(-1, 1, self.hidden_size)
            print("### rank:{} JambaDoEAttentionDecoderLayer forward conv1d_input shape reshaped: {}\n".format(torch.distributed.get_rank(), conv1d_input.shape))
            conv1d_output = self.conv1d(conv1d_input.permute(1, 2, 0))[:, :, :-(self.config.conv_attention_kernel_size - 1)].permute(2, 0, 1).contiguous().view(-1, self.hidden_size)
            self.conv1d_cache = conv1d_output[-3:]
            hidden_states = conv1d_output[-num_decode_tokens:]
        
        else:
            # prefill
            # if len(hidden_states.shape) == 3:
            #     hidden_states = self.conv1d(hidden_states.permute(1, 2, 0))[:, :,:-(self.config.conv_attention_kernel_size - 1)].permute(2, 0, 1).contiguous()

            # else:
                # forward_context = get_forward_context()
                # num_prefills = forward_context.attn_metadata.num_prefills
            hidden_states = hidden_states.view(-1, 1, self.hidden_size)
            print("### rank:{} JambaDoEAttentionDecoderLayer forward input shape reshaped: {}\n".format(torch.distributed.get_rank(), hidden_states.shape))
            hidden_states = self.conv1d(hidden_states.permute(1, 2, 0))[:, :, :-(self.config.conv_attention_kernel_size - 1)].permute(2, 0, 1).contiguous().view(-1, self.hidden_size)
            L = hidden_states.size(0)
            if L >= self.config.conv_attention_kernel_size - 1:
                self.conv1d_cache.copy_(hidden_states[-(self.config.conv_attention_kernel_size - 1)])
            else:
                self.conv1d_cache[-L:] = hidden_states

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
        # if hidden_states.dtype == torch.float16:
        #     # Fix FP16 overflow
        #     # We scale both hidden_states and residual before
        #     # rmsnorm, and rmsnorm result would not affect by scale.
        #     hidden_states *= 1. / self.routed_scaling_factor
        #     if self.layer_idx == 0:
        #         # The residual is shared by all layers, we only scale it on
        #         # first layer.
        #         residual *= 1. / self.routed_scaling_factor

        # Fully Connected
        # hidden_states, residual = self.post_attention_layernorm(
        #     hidden_states, residual)
        # hidden_states = self.mlp(hidden_states)

        # if isinstance(self.mlp,
        #               DeepseekV2MLP) and hidden_states.dtype == torch.float16:
        #     # Fix FP16 overflow
        #     # Scaling the DeepseekV2MLP output, it is the input of
        #     # input_layernorm of next decoder layer.
        #     # The scaling of DeepseekV2MOE output would be done in the forward
        #     # of DeepseekV2MOE
        #     hidden_states *= 1. / self.routed_scaling_factor

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
        lora_config = vllm_config.lora_config

        self.config = config
        # lora_vocab = ((lora_config.lora_extra_vocab_size *
        #                (lora_config.max_loras or 1)) if lora_config else 0)
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
                
                
        print("### rank:{} JambaDoEModel forward hidden_states shape: {}\n".format(torch.distributed.get_rank(), hidden_states.shape))

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
    # packed_modules_mapping = {
    #     "qkv_proj": [
    #         "q_proj",
    #         "k_proj",
    #         "v_proj",
    #     ],
    #     "in_proj": ["in_proj"],
    # }

    # # LoRA specific attributes
    # embedding_modules = {
    #     "embed_tokens": "input_embeddings",
    #     "lm_head": "output_embeddings",
    # }
    # embedding_padding_modules = ["lm_head"]

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
            
            
        print("### rank:{} JambaDoEForCausalLM forward input_ids input shape: {}\n".format(torch.distributed.get_rank(), input_ids.shape))
        print("### rank:{} JambaDoEForCausalLM forward input_ids positions shape: {}\n".format(torch.distributed.get_rank(), positions.shape))

        mamba_cache_params = self.mamba_cache.current_run_tensors(**kwargs)

        hidden_states = self.model(input_ids, positions, mamba_cache_params,
                                   intermediate_tensors, inputs_embeds)
        return hidden_states

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    # def _get_mamba_cache_shape(
    #         self) -> tuple[tuple[int, int], tuple[int, int]]:
    #     world_size = get_tensor_model_parallel_world_size()
    #     hidden_size = self.config.hidden_size
    #     conv_state_shape = (
    #         self.config.mamba_expand * hidden_size // world_size,
    #         self.config.mamba_d_conv - 1,
    #     )
    #     temporal_state_shape = (
    #         self.config.mamba_expand * hidden_size // world_size,
    #         self.config.mamba_d_state,
    #     )
    #     return conv_state_shape, temporal_state_shape
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
        # stacked_params_mapping = [
        #     # (param_name, shard_name, shard_id)
        #     ("qkv_proj", "q_proj", "q"),
        #     ("qkv_proj", "k_proj", "k"),
        #     ("qkv_proj", "v_proj", "v"),
        # ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        # expert_params_mapping = FusedMoE.make_expert_params_mapping(
        #     ckpt_gate_proj_name="gate_proj",
        #     ckpt_down_proj_name="down_proj",
        #     ckpt_up_proj_name="up_proj",
        #     num_experts=self.config.num_experts)

        params_dict = dict(self.named_parameters())
        # print("### rank: {}, jamba doe weight loader expert_params_mapping: {}\n".format(get_pp_group().rank, expert_params_mapping))
        # if get_pp_group().rank == 0:
        #     for k, v in params_dict.items():
        #         print("### rank: {}, jamba doe weight loader params_dict k: {}, v shape: {}\n".format(get_pp_group().rank, k, v.shape))
        #     for name, loaded_weight in weights:
        #         print("### rank: {}, jamba doe weight loader weights all name: {}, shape: {}\n".format(get_pp_group().rank, name, loaded_weight.shape))

        loaded_params: set[str] = set()
        layer_num_pattern = r'\.layers\.(\d+)\.'
        exp1_weight_pattern = r'linear_fc1.weight(\d+)$'
        exp2_weight_pattern = r'linear_fc2.weight(\d+)$'
        # pattern = r'\.layers\.(\d+)\.'
        # re.search(r'weight(\d+)$', name)
        for name, loaded_weight in weights:
            # print("### jamba doe weight loader weights name: {}, shape: {}\n".format(name, loaded_weight.shape))
            # if "rotary_emb.inv_freq" in name:
            #     continue
            layer_match = re.search(layer_num_pattern, name)
            if layer_match:
                layer_id = int(layer_match.group(1))
                if layer_id < self.model.start_layer or layer_id >= self.model.end_layer:
                    # print("### rank: {}, layer_id: {} skip ppmising layer name: {}, loaded_weight shape: {}\n".format(get_pp_group().rank, layer_id,name, loaded_weight.shape))
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
                # name = name.replace("mlp", "feed_forward")
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
                ## map MLP layers to expert with ID=0
                name = name.replace(".norm.weight", ".input_layernorm.weight")
            
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
            # for param_name, weight_name, shard_id in stacked_params_mapping:
            #     if weight_name not in name:
            #         continue
            #     if 'experts' in name:
            #         continue
            #     name = name.replace(weight_name, param_name)
            #     # Skip loading extra bias for GPTQ models.

            #     if name.endswith(".bias") and name not in params_dict:
            #         continue
            #     # Skip layers on other devices.
            #     if is_pp_missing_parameter(name, self):
            #         continue
            #     param = params_dict[name]
            #     weight_loader = param.weight_loader
            #     weight_loader(param, loaded_weight, shard_id)
            #     break
            # else:
            #     for (
            #             param_name,
            #             weight_name,
            #             expert_id,
            #             shard_id,
            #     ) in expert_params_mapping:
            #         if weight_name not in name:
            #             continue

            #         if is_pp_missing_parameter(name, self):
            #             continue
            #         name = name.replace(weight_name, param_name)
            #         param = params_dict[name]
            #         weight_loader = param.weight_loader
            #         weight_loader(param,
            #                       loaded_weight,
            #                       name,
            #                       shard_id=shard_id,
            #                       expert_id=expert_id)
            #         break
            #     else:
            #         # Skip loading extra bias for GPTQ models.
            #         if name.endswith(".bias") and name not in params_dict:
            #             continue
            #         if is_pp_missing_parameter(name, self):
            #             continue

            #         param = params_dict[name]
            #         weight_loader = getattr(param, "weight_loader",
            #                                 default_weight_loader)
            #         weight_loader(param, loaded_weight)
            # loaded_params.add(name)
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



# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
# """Inference-only DeepseekV2/DeepseekV3 model."""
# import typing
# from collections.abc import Callable, Iterable
# from typing import Any, Optional, Union

# import torch
# from torch import nn
# from transformers import PretrainedConfig

# from vllm.attention import Attention
# from vllm.compilation.decorators import support_torch_compile
# from vllm.config import (CacheConfig, ModelConfig, VllmConfig,
#                          get_current_vllm_config)
# from vllm.distributed import (get_ep_group, get_pp_group,
#                               get_tensor_model_parallel_world_size)
# from vllm.model_executor.layers.activation import SiluAndMul
# from vllm.model_executor.layers.fused_moe import FusedMoE
# from vllm.model_executor.layers.layernorm import RMSNorm
# from vllm.model_executor.layers.linear import (ColumnParallelLinear,
#                                                MergedColumnParallelLinear,
#                                                ReplicatedLinear,
#                                                RowParallelLinear)
# from vllm.model_executor.layers.logits_processor import LogitsProcessor
# from vllm.model_executor.layers.quantization import QuantizationConfig
# from vllm.model_executor.layers.rotary_embedding import get_rope
# from vllm.model_executor.layers.vocab_parallel_embedding import (
#     ParallelLMHead, VocabParallelEmbedding)
# from vllm.model_executor.model_loader.weight_utils import (
#     default_weight_loader, maybe_remap_kv_scale_name)
# from vllm.model_executor.sampling_metadata import SamplingMetadata
# from vllm.sequence import IntermediateTensors

# from .interfaces import MixtureOfExperts, SupportsPP
# from .utils import (PPMissingLayer, is_pp_missing_parameter,
#                     make_empty_intermediate_tensors_factory, make_layers,
#                     maybe_prefix)


# class DeepseekV2MLP(nn.Module):

#     def __init__(
#         self,
#         hidden_size: int,
#         intermediate_size: int,
#         hidden_act: str,
#         quant_config: Optional[QuantizationConfig] = None,
#         reduce_results: bool = True,
#         prefix: str = "",
#     ) -> None:
#         super().__init__()
#         self.gate_up_proj = MergedColumnParallelLinear(
#             hidden_size, [intermediate_size] * 2,
#             bias=False,
#             quant_config=quant_config,
#             prefix=f"{prefix}.gate_up_proj")
#         self.down_proj = RowParallelLinear(intermediate_size,
#                                            hidden_size,
#                                            bias=False,
#                                            quant_config=quant_config,
#                                            reduce_results=reduce_results,
#                                            prefix=f"{prefix}.down_proj")
#         if hidden_act != "silu":
#             raise ValueError(f"Unsupported activation: {hidden_act}. "
#                              "Only silu is supported for now.")
#         self.act_fn = SiluAndMul()

#     def forward(self, x):
#         gate_up, _ = self.gate_up_proj(x)
#         x = self.act_fn(gate_up)
#         x, _ = self.down_proj(x)
#         return x


# class DeepseekV2MoE(nn.Module):

#     def __init__(
#         self,
#         config: PretrainedConfig,
#         quant_config: Optional[QuantizationConfig] = None,
#         prefix: str = "",
#         enable_eplb: bool = False,
#     ):
#         super().__init__()
#         self.tp_size = get_tensor_model_parallel_world_size()
#         self.routed_scaling_factor = config.routed_scaling_factor

#         self.ep_group = get_ep_group().device_group
#         self.ep_rank = self.ep_group.rank()
#         self.ep_size = self.ep_group.size()
#         self.n_routed_experts: int = config.n_routed_experts
#         self.n_shared_experts: int = config.n_shared_experts

#         if config.hidden_act != "silu":
#             raise ValueError(f"Unsupported activation: {config.hidden_act}. "
#                              "Only silu is supported for now.")

#         self.gate = ReplicatedLinear(config.hidden_size,
#                                      config.n_routed_experts,
#                                      bias=False,
#                                      quant_config=None,
#                                      prefix=f"{prefix}.gate")
#         if config.topk_method == "noaux_tc":
#             self.gate.e_score_correction_bias = nn.Parameter(
#                 torch.empty(config.n_routed_experts))
#         else:
#             self.gate.e_score_correction_bias = None

#         # Load balancing settings.
#         vllm_config = get_current_vllm_config()
#         parallel_config = vllm_config.parallel_config
#         self.enable_eplb = enable_eplb

#         self.n_redundant_experts = parallel_config.num_redundant_experts
#         self.n_logical_experts = self.n_routed_experts
#         self.n_physical_experts = (self.n_logical_experts +
#                                    self.n_redundant_experts)
#         self.n_local_physical_experts = self.n_physical_experts // self.ep_size

#         self.physical_expert_start = (self.ep_rank *
#                                       self.n_local_physical_experts)
#         self.physical_expert_end = (self.physical_expert_start +
#                                     self.n_local_physical_experts)

#         self.experts = FusedMoE(
#             num_experts=config.n_routed_experts,
#             top_k=config.num_experts_per_tok,
#             hidden_size=config.hidden_size,
#             intermediate_size=config.moe_intermediate_size,
#             reduce_results=False,
#             renormalize=config.norm_topk_prob,
#             quant_config=quant_config,
#             use_grouped_topk=True,
#             num_expert_group=config.n_group,
#             topk_group=config.topk_group,
#             prefix=f"{prefix}.experts",
#             scoring_func=config.scoring_func,
#             e_score_correction_bias=self.gate.e_score_correction_bias,
#             enable_eplb=self.enable_eplb,
#             num_redundant_experts=self.n_redundant_experts)

#         if config.n_shared_experts is not None:
#             intermediate_size = (config.moe_intermediate_size *
#                                  config.n_shared_experts)
#             self.shared_experts = DeepseekV2MLP(
#                 hidden_size=config.hidden_size,
#                 intermediate_size=intermediate_size,
#                 hidden_act=config.hidden_act,
#                 quant_config=quant_config,
#                 reduce_results=self.experts.must_reduce_shared_expert_outputs(
#                 ),
#                 prefix=f"{prefix}.shared_experts",
#             )

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         num_tokens, hidden_dim = hidden_states.shape
#         hidden_states = hidden_states.view(-1, hidden_dim)
#         if self.n_shared_experts is not None:
#             shared_output = self.shared_experts(hidden_states)
#         # router_logits: (num_tokens, n_experts)
#         router_logits, _ = self.gate(hidden_states)

#         if hidden_states.dtype != torch.float16:
#             final_hidden_states = self.experts(
#                 hidden_states=hidden_states,
#                 router_logits=router_logits) * self.routed_scaling_factor
#         else:
#             # Fix FP16 overflow
#             # See DeepseekV2DecoderLayer for more details.
#             final_hidden_states = self.experts(hidden_states=hidden_states,
#                                                router_logits=router_logits)
#         if shared_output is not None:
#             if hidden_states.dtype != torch.float16:
#                 final_hidden_states = final_hidden_states + shared_output
#             else:
#                 # Fix FP16 overflow
#                 # See DeepseekV2DecoderLayer for more details.
#                 final_hidden_states = final_hidden_states + shared_output \
#                     * (1. / self.routed_scaling_factor)

#         if self.tp_size > 1:
#             final_hidden_states = (
#                 self.experts.maybe_all_reduce_tensor_model_parallel(
#                     final_hidden_states))

#         return final_hidden_states.view(num_tokens, hidden_dim)


# def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
#     import math
#     if scale <= 1:
#         return 1.0
#     return 0.1 * mscale * math.log(scale) + 1.0


# class DeepseekV2Attention(nn.Module):

#     def __init__(
#         self,
#         config: PretrainedConfig,
#         hidden_size: int,
#         num_heads: int,
#         qk_nope_head_dim: int,
#         qk_rope_head_dim: int,
#         v_head_dim: int,
#         q_lora_rank: int,
#         kv_lora_rank: int,
#         rope_theta: float = 10000,
#         rope_scaling: Optional[dict[str, Any]] = None,
#         max_position_embeddings: int = 8192,
#         cache_config: Optional[CacheConfig] = None,
#         quant_config: Optional[QuantizationConfig] = None,
#         prefix: str = "",
#     ) -> None:
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.qk_nope_head_dim = qk_nope_head_dim
#         self.qk_rope_head_dim = qk_rope_head_dim
#         self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
#         self.v_head_dim = v_head_dim
#         self.q_lora_rank = q_lora_rank
#         self.kv_lora_rank = kv_lora_rank
#         self.num_heads = num_heads
#         tp_size = get_tensor_model_parallel_world_size()
#         assert num_heads % tp_size == 0
#         self.num_local_heads = num_heads // tp_size
#         self.scaling = self.qk_head_dim**-0.5
#         self.rope_theta = rope_theta
#         self.max_position_embeddings = max_position_embeddings

#         if self.q_lora_rank is not None:
#             self.q_a_proj = ReplicatedLinear(self.hidden_size,
#                                              self.q_lora_rank,
#                                              bias=False,
#                                              quant_config=quant_config,
#                                              prefix=f"{prefix}.q_a_proj")
#             self.q_a_layernorm = RMSNorm(self.q_lora_rank,
#                                          eps=config.rms_norm_eps)
#             self.q_b_proj = ColumnParallelLinear(q_lora_rank,
#                                                  self.num_heads *
#                                                  self.qk_head_dim,
#                                                  bias=False,
#                                                  quant_config=quant_config,
#                                                  prefix=f"{prefix}.q_b_proj")
#         else:
#             self.q_proj = ColumnParallelLinear(self.hidden_size,
#                                                self.num_heads *
#                                                self.qk_head_dim,
#                                                bias=False,
#                                                quant_config=quant_config,
#                                                prefix=f"{prefix}.q_proj")

#         self.kv_a_proj_with_mqa = ReplicatedLinear(
#             self.hidden_size,
#             self.kv_lora_rank + self.qk_rope_head_dim,
#             bias=False,
#             quant_config=quant_config,
#             prefix=f"{prefix}.kv_a_proj_with_mqa")
#         self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
#                                       eps=config.rms_norm_eps)
#         self.kv_b_proj = ColumnParallelLinear(
#             self.kv_lora_rank,
#             self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
#             bias=False,
#             quant_config=quant_config,
#             prefix=f"{prefix}.kv_b_proj")
#         # O projection.
#         self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim,
#                                         self.hidden_size,
#                                         bias=False,
#                                         quant_config=quant_config,
#                                         prefix=f"{prefix}.o_proj")
#         if rope_scaling:
#             rope_scaling["rope_type"] = 'deepseek_yarn'

#         self.rotary_emb = get_rope(qk_rope_head_dim,
#                                    rotary_dim=qk_rope_head_dim,
#                                    max_position=max_position_embeddings,
#                                    base=rope_theta,
#                                    rope_scaling=rope_scaling,
#                                    is_neox_style=False)

#         if rope_scaling:
#             mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
#             scaling_factor = rope_scaling["factor"]
#             mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
#             self.scaling = self.scaling * mscale * mscale

#         self.attn = Attention(self.num_local_heads,
#                               self.qk_head_dim,
#                               self.scaling,
#                               num_kv_heads=self.num_local_heads,
#                               cache_config=cache_config,
#                               quant_config=quant_config,
#                               prefix=f"{prefix}.attn")

#     def forward(
#         self,
#         positions: torch.Tensor,
#         hidden_states: torch.Tensor,
#     ) -> torch.Tensor:
#         if self.q_lora_rank is not None:
#             q = self.q_a_proj(hidden_states)[0]
#             q = self.q_a_layernorm(q)
#             q = self.q_b_proj(q)[0].view(-1, self.num_local_heads,
#                                          self.qk_head_dim)
#         else:
#             q = self.q_proj(hidden_states)[0].view(-1, self.num_local_heads,
#                                                    self.qk_head_dim)
#         q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim],
#                                dim=-1)
#         latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
#         kv_a, _ = latent_cache.split(
#             [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
#         latent_cache = latent_cache.unsqueeze(1)
#         kv_a = self.kv_a_layernorm(kv_a.contiguous())
#         kv = self.kv_b_proj(kv_a)[0]
#         kv = kv.view(-1, self.num_local_heads,
#                      self.qk_nope_head_dim + self.v_head_dim)
#         k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
#         k_pe = latent_cache[:, :, self.kv_lora_rank:]

#         q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

#         q[..., self.qk_nope_head_dim:] = q_pe
#         k = torch.empty_like(q)
#         k[..., :self.qk_nope_head_dim] = k_nope
#         k[..., self.qk_nope_head_dim:] = k_pe
#         # padding value to qk_head_dim for alignment
#         v = torch.nn.functional.pad(
#             v, [0, self.qk_head_dim - self.v_head_dim],
#             value=0).view(-1, self.num_local_heads * self.qk_head_dim)
#         attn_output = self.attn(q, k, v)
#         attn_output = attn_output.view(
#             -1, self.num_local_heads,
#             self.qk_head_dim)[..., :self.v_head_dim].reshape(
#                 -1, self.num_local_heads * self.v_head_dim)
#         output, _ = self.o_proj(attn_output)
#         return output


# class DeepseekV2MLAAttention(nn.Module):
#     """
#     Main reference: DeepseekV2 paper, and FlashInfer Implementation
#     (https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).
    
#     For more info see MLACommonImpl in: vllm/attention/backends/mla/utils.py
#     """

#     def __init__(
#         self,
#         config: PretrainedConfig,
#         hidden_size: int,
#         num_heads: int,
#         qk_nope_head_dim: int,
#         qk_rope_head_dim: int,
#         v_head_dim: int,
#         q_lora_rank: Optional[int],
#         kv_lora_rank: int,
#         rope_theta: float = 10000,
#         rope_scaling: Optional[dict[str, Any]] = None,
#         max_position_embeddings: int = 8192,
#         cache_config: Optional[CacheConfig] = None,
#         quant_config: Optional[QuantizationConfig] = None,
#         prefix: str = "",
#     ) -> None:
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.qk_nope_head_dim = qk_nope_head_dim
#         self.qk_rope_head_dim = qk_rope_head_dim
#         self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
#         self.v_head_dim = v_head_dim

#         self.q_lora_rank = q_lora_rank
#         self.kv_lora_rank = kv_lora_rank

#         self.num_heads = num_heads
#         tp_size = get_tensor_model_parallel_world_size()
#         assert num_heads % tp_size == 0
#         self.num_local_heads = num_heads // tp_size

#         self.scaling = self.qk_head_dim**-0.5
#         self.rope_theta = rope_theta
#         self.max_position_embeddings = max_position_embeddings

#         if self.q_lora_rank is not None:
#             self.q_a_proj = ReplicatedLinear(self.hidden_size,
#                                              self.q_lora_rank,
#                                              bias=False,
#                                              quant_config=quant_config,
#                                              prefix=f"{prefix}.q_a_proj")
#             self.q_a_layernorm = RMSNorm(self.q_lora_rank,
#                                          eps=config.rms_norm_eps)
#             self.q_b_proj = ColumnParallelLinear(q_lora_rank,
#                                                  self.num_heads *
#                                                  self.qk_head_dim,
#                                                  bias=False,
#                                                  quant_config=quant_config,
#                                                  prefix=f"{prefix}.q_b_proj")
#         else:
#             self.q_proj = ColumnParallelLinear(self.hidden_size,
#                                                self.num_heads *
#                                                self.qk_head_dim,
#                                                bias=False,
#                                                quant_config=quant_config,
#                                                prefix=f"{prefix}.q_proj")

#         self.kv_a_proj_with_mqa = ReplicatedLinear(
#             self.hidden_size,
#             self.kv_lora_rank + self.qk_rope_head_dim,
#             bias=False,
#             quant_config=quant_config,
#             prefix=f"{prefix}.kv_a_proj_with_mqa")
#         self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
#                                       eps=config.rms_norm_eps)
#         self.kv_b_proj = ColumnParallelLinear(
#             self.kv_lora_rank,
#             self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
#             bias=False,
#             quant_config=quant_config,
#             prefix=f"{prefix}.kv_b_proj")
#         self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim,
#                                         self.hidden_size,
#                                         bias=False,
#                                         quant_config=quant_config,
#                                         prefix=f"{prefix}.o_proj")

#         if rope_scaling:
#             rope_scaling["rope_type"] = 'deepseek_yarn'
#         self.rotary_emb = get_rope(qk_rope_head_dim,
#                                    rotary_dim=qk_rope_head_dim,
#                                    max_position=max_position_embeddings,
#                                    base=rope_theta,
#                                    rope_scaling=rope_scaling,
#                                    is_neox_style=False)
#         if rope_scaling:
#             mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
#             scaling_factor = rope_scaling["factor"]
#             mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
#             self.scaling = self.scaling * mscale * mscale

#         # In the MLA backend, kv_cache includes both k_c and
#         # pe (i.e. decoupled position embeddings). In particular,
#         # the concat_and_cache_mla op requires
#         #     k_c.size(1) + k_pe.size(1) == kv_cache.size(2)
#         # i.e.
#         #     kv_lora_rank + qk_rope_head_dim == head_size
#         self.mla_attn = Attention(
#             num_heads=self.num_local_heads,
#             head_size=self.kv_lora_rank + self.qk_rope_head_dim,
#             scale=self.scaling,
#             num_kv_heads=1,
#             cache_config=cache_config,
#             quant_config=quant_config,
#             prefix=f"{prefix}.attn",
#             use_mla=True,
#             # MLA Args
#             q_lora_rank=self.q_lora_rank,
#             kv_lora_rank=self.kv_lora_rank,
#             qk_nope_head_dim=self.qk_nope_head_dim,
#             qk_rope_head_dim=self.qk_rope_head_dim,
#             qk_head_dim=self.qk_head_dim,
#             v_head_dim=self.v_head_dim,
#             kv_b_proj=self.kv_b_proj,
#         )

#         self.prefix = prefix
#         self.debug_layer_idx = int(self.prefix.split(".")[-2])

#     def forward(
#         self,
#         positions: torch.Tensor,
#         hidden_states: torch.Tensor,
#     ) -> torch.Tensor:
#         if self.q_lora_rank is not None:
#             q_c = self.q_a_proj(hidden_states)[0]
#             q_c = self.q_a_layernorm(q_c)
#             q = self.q_b_proj(q_c)[0]
#         else:
#             q = self.q_proj(hidden_states)[0]
#         kv_c, k_pe = self.kv_a_proj_with_mqa(hidden_states)[0].split(
#             [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
#         kv_c_normed = self.kv_a_layernorm(kv_c.contiguous())

#         q = q.view(-1, self.num_local_heads, self.qk_head_dim)
#         # Add head dim of 1 to k_pe
#         k_pe = k_pe.unsqueeze(1)

#         q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(
#             positions, q[..., self.qk_nope_head_dim:], k_pe)

#         attn_out = self.mla_attn(
#             q,
#             kv_c_normed,
#             k_pe,
#             output_shape=(hidden_states.shape[0],
#                           self.num_local_heads * self.v_head_dim))
#         return self.o_proj(attn_out)[0]


# class DeepseekV2DecoderLayer(nn.Module):

#     def __init__(
#         self,
#         config: PretrainedConfig,
#         prefix: str,
#         model_config: ModelConfig,
#         cache_config: Optional[CacheConfig] = None,
#         quant_config: Optional[QuantizationConfig] = None,
#         enable_eplb: bool = False,
#     ) -> None:
#         super().__init__()
#         self.hidden_size = config.hidden_size
#         rope_theta = getattr(config, "rope_theta", 10000)
#         rope_scaling = getattr(config, "rope_scaling", None)
#         max_position_embeddings = getattr(config, "max_position_embeddings",
#                                           8192)
#         # DecoderLayers are created with `make_layers` which passes the prefix
#         # with the layer's index.
#         layer_idx = int(prefix.split(sep='.')[-1])
#         self.layer_idx = layer_idx
#         if model_config.use_mla:
#             attn_cls = DeepseekV2MLAAttention
#         else:
#             attn_cls = DeepseekV2Attention
#         self.self_attn = attn_cls(
#             config=config,
#             hidden_size=self.hidden_size,
#             num_heads=config.num_attention_heads,
#             qk_nope_head_dim=config.qk_nope_head_dim,
#             qk_rope_head_dim=config.qk_rope_head_dim,
#             v_head_dim=config.v_head_dim,
#             q_lora_rank=config.q_lora_rank
#             if hasattr(config, "q_lora_rank") else None,
#             kv_lora_rank=config.kv_lora_rank,
#             rope_theta=rope_theta,
#             rope_scaling=rope_scaling,
#             max_position_embeddings=max_position_embeddings,
#             cache_config=cache_config,
#             quant_config=quant_config,
#             prefix=f"{prefix}.self_attn",
#         )

#         if (config.n_routed_experts is not None
#                 and layer_idx >= config.first_k_dense_replace
#                 and layer_idx % config.moe_layer_freq == 0):
#             self.mlp = DeepseekV2MoE(
#                 config=config,
#                 quant_config=quant_config,
#                 prefix=f"{prefix}.mlp",
#                 enable_eplb=enable_eplb,
#             )
#         else:
#             self.mlp = DeepseekV2MLP(
#                 hidden_size=config.hidden_size,
#                 intermediate_size=config.intermediate_size,
#                 hidden_act=config.hidden_act,
#                 quant_config=quant_config,
#                 prefix=f"{prefix}.mlp",
#             )
#         self.input_layernorm = RMSNorm(config.hidden_size,
#                                        eps=config.rms_norm_eps)
#         self.post_attention_layernorm = RMSNorm(config.hidden_size,
#                                                 eps=config.rms_norm_eps)
#         self.routed_scaling_factor = config.routed_scaling_factor

#     def forward(
#         self,
#         positions: torch.Tensor,
#         hidden_states: torch.Tensor,
#         residual: Optional[torch.Tensor],
#     ) -> torch.Tensor:
#         # Self Attention
#         if residual is None:
#             residual = hidden_states
#             hidden_states = self.input_layernorm(hidden_states)
#         else:
#             hidden_states, residual = self.input_layernorm(
#                 hidden_states, residual)
#         hidden_states = self.self_attn(
#             positions=positions,
#             hidden_states=hidden_states,
#         )

#         if hidden_states.dtype == torch.float16:
#             # Fix FP16 overflow
#             # We scale both hidden_states and residual before
#             # rmsnorm, and rmsnorm result would not affect by scale.
#             hidden_states *= 1. / self.routed_scaling_factor
#             if self.layer_idx == 0:
#                 # The residual is shared by all layers, we only scale it on
#                 # first layer.
#                 residual *= 1. / self.routed_scaling_factor

#         # Fully Connected
#         hidden_states, residual = self.post_attention_layernorm(
#             hidden_states, residual)
#         hidden_states = self.mlp(hidden_states)

#         if isinstance(self.mlp,
#                       DeepseekV2MLP) and hidden_states.dtype == torch.float16:
#             # Fix FP16 overflow
#             # Scaling the DeepseekV2MLP output, it is the input of
#             # input_layernorm of next decoder layer.
#             # The scaling of DeepseekV2MOE output would be done in the forward
#             # of DeepseekV2MOE
#             hidden_states *= 1. / self.routed_scaling_factor

#         return hidden_states, residual


# @support_torch_compile
# class DeepseekV2Model(nn.Module):

#     fall_back_to_pt_during_load = False

#     def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
#         super().__init__()

#         config = vllm_config.model_config.hf_config
#         model_config = vllm_config.model_config
#         cache_config = vllm_config.cache_config
#         quant_config = vllm_config.quant_config
#         enable_eplb = vllm_config.parallel_config.enable_eplb
#         self.config = config

#         self.vocab_size = config.vocab_size

#         if get_pp_group().is_first_rank:
#             self.embed_tokens = VocabParallelEmbedding(
#                 config.vocab_size,
#                 config.hidden_size,
#                 quant_config=quant_config,
#                 prefix=f"{prefix}.embed_tokens")
#         else:
#             self.embed_tokens = PPMissingLayer()

#         self.start_layer, self.end_layer, self.layers = make_layers(
#             config.num_hidden_layers,
#             lambda prefix: DeepseekV2DecoderLayer(
#                 config,
#                 prefix,
#                 model_config=model_config,
#                 cache_config=cache_config,
#                 quant_config=quant_config,
#                 enable_eplb=enable_eplb,
#             ),
#             prefix=f"{prefix}.layers")

#         if get_pp_group().is_last_rank:
#             self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         else:
#             self.norm = PPMissingLayer()
#         self.make_empty_intermediate_tensors = (
#             make_empty_intermediate_tensors_factory(
#                 ["hidden_states", "residual"], config.hidden_size))

#     def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
#         return self.embed_tokens(input_ids)

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         positions: torch.Tensor,
#         intermediate_tensors: Optional[IntermediateTensors],
#         inputs_embeds: Optional[torch.Tensor] = None,
#     ) -> Union[torch.Tensor, IntermediateTensors]:
#         if get_pp_group().is_first_rank:
#             if inputs_embeds is not None:
#                 hidden_states = inputs_embeds
#             else:
#                 hidden_states = self.get_input_embeddings(input_ids)
#             residual = None
#         else:
#             assert intermediate_tensors is not None
#             hidden_states = intermediate_tensors["hidden_states"]
#             residual = intermediate_tensors["residual"]

#         for layer in self.layers[self.start_layer:self.end_layer]:
#             hidden_states, residual = layer(positions, hidden_states, residual)

#         if not get_pp_group().is_last_rank:
#             return IntermediateTensors({
#                 "hidden_states": hidden_states,
#                 "residual": residual
#             })

#         hidden_states, _ = self.norm(hidden_states, residual)
#         return hidden_states


# class DeepseekV2ForCausalLM(nn.Module, SupportsPP, MixtureOfExperts):

#     def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
#         super().__init__()
#         config = vllm_config.model_config.hf_config
#         quant_config = vllm_config.quant_config
#         self.config = config
#         self.quant_config = quant_config
#         self.model = DeepseekV2Model(vllm_config=vllm_config,
#                                      prefix=maybe_prefix(prefix, "model"))
#         if get_pp_group().is_last_rank:
#             self.lm_head = ParallelLMHead(config.vocab_size,
#                                           config.hidden_size,
#                                           quant_config=quant_config)
#         else:
#             self.lm_head = PPMissingLayer()
#         self.logits_processor = LogitsProcessor(config.vocab_size)
#         self.make_empty_intermediate_tensors = (
#             self.model.make_empty_intermediate_tensors)
#         self.expert_weights = []

#         # Set MoE hyperparameters
#         self.num_moe_layers = (config.num_hidden_layers -
#                                config.first_k_dense_replace)
#         self.num_expert_groups = config.n_group

#         self.moe_layers: list[FusedMoE] = []
#         for layer in self.model.layers:
#             assert isinstance(layer, DeepseekV2DecoderLayer)
#             if isinstance(layer.mlp, DeepseekV2MoE):
#                 self.moe_layers.append(layer.mlp.experts)

#         # Pick last one layer since the first ones may be dense layers.
#         example_moe = typing.cast(
#             DeepseekV2MoE, self.model.layers[config.num_hidden_layers - 1].mlp)
#         self.num_logical_experts = example_moe.n_logical_experts
#         self.num_physical_experts = example_moe.n_physical_experts
#         self.num_local_physical_experts = example_moe.n_local_physical_experts
#         self.num_routed_experts = example_moe.n_routed_experts
#         self.num_shared_experts = example_moe.n_shared_experts
#         self.num_redundant_experts = example_moe.n_redundant_experts

#     def set_eplb_state(
#         self,
#         expert_load_view: torch.Tensor,
#         logical_to_physical_map: torch.Tensor,
#         logical_replica_count: torch.Tensor,
#     ) -> None:
#         for layer_idx, layer in enumerate(self.moe_layers):
#             # Register the expert weights.
#             self.expert_weights.append(layer.get_expert_weights())
#             layer.set_eplb_state(
#                 moe_layer_idx=layer_idx,
#                 expert_load_view=expert_load_view,
#                 logical_to_physical_map=logical_to_physical_map,
#                 logical_replica_count=logical_replica_count,
#             )

#     def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
#         return self.model.get_input_embeddings(input_ids)

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         positions: torch.Tensor,
#         intermediate_tensors: Optional[IntermediateTensors] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#     ) -> Union[torch.Tensor, IntermediateTensors]:
#         hidden_states = self.model(input_ids, positions, intermediate_tensors,
#                                    inputs_embeds)
#         return hidden_states

#     def compute_logits(
#         self,
#         hidden_states: torch.Tensor,
#         sampling_metadata: SamplingMetadata,
#     ) -> Optional[torch.Tensor]:
#         logits = self.logits_processor(self.lm_head, hidden_states,
#                                        sampling_metadata)
#         return logits

#     def make_empty_intermediate_tensors(
#             self, batch_size: int, dtype: torch.dtype,
#             device: torch.device) -> IntermediateTensors:
#         return IntermediateTensors({
#             "hidden_states":
#             torch.zeros((batch_size, self.config.hidden_size),
#                         dtype=dtype,
#                         device=device),
#             "residual":
#             torch.zeros((batch_size, self.config.hidden_size),
#                         dtype=dtype,
#                         device=device),
#         })

#     def load_weights(self, weights: Iterable[tuple[str,
#                                                    torch.Tensor]]) -> set[str]:
#         stacked_params_mapping = [
#             # (param_name, shard_name, shard_id)
#             ("gate_up_proj", "gate_proj", 0),
#             ("gate_up_proj", "up_proj", 1),
#         ]

#         # Params for weights, fp8 weight scales, fp8 activation scales
#         # (param_name, weight_name, expert_id, shard_id)
#         expert_params_mapping = FusedMoE.make_expert_params_mapping(
#             ckpt_gate_proj_name="gate_proj",
#             ckpt_down_proj_name="down_proj",
#             ckpt_up_proj_name="up_proj",
#             num_experts=self.config.n_routed_experts,
#             num_redundant_experts=self.num_redundant_experts)

#         params_dict = dict(self.named_parameters())
#         loaded_params: set[str] = set()
#         for name, loaded_weight in weights:
#             if "rotary_emb.inv_freq" in name:
#                 continue

#             spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
#             if spec_layer is not None:
#                 continue  # skip spec decode layers for main model

#             for (param_name, weight_name, shard_id) in stacked_params_mapping:
#                 # Skip non-stacked layers and experts (experts handled below).
#                 if weight_name not in name:
#                     continue
#                 # We have mlp.experts[0].gate_proj in the checkpoint.
#                 # Since we handle the experts below in expert_params_mapping,
#                 # we need to skip here BEFORE we update the name, otherwise
#                 # name will be updated to mlp.experts[0].gate_up_proj, which
#                 # will then be updated below in expert_params_mapping
#                 # for mlp.experts[0].gate_gate_up_proj, which breaks load.
#                 if (("mlp.experts." in name) and name not in params_dict):
#                     continue
#                 name = name.replace(weight_name, param_name)
#                 # Skip loading extra bias for GPTQ models.
#                 if name.endswith(".bias") and name not in params_dict:
#                     continue

#                 if is_pp_missing_parameter(name, self):
#                     continue

#                 param = params_dict[name]
#                 weight_loader = param.weight_loader
#                 weight_loader(param, loaded_weight, shard_id)
#                 break
#             else:
#                 is_expert_weight = False
#                 for mapping in expert_params_mapping:
#                     param_name, weight_name, expert_id, shard_id = mapping
#                     if weight_name not in name:
#                         continue

#                     # Anyway, this is an expert weight and should not be
#                     # attempted to load as other weights later
#                     is_expert_weight = True

#                     # Do not modify `name` since the loop may continue here
#                     # Instead, create a new variable
#                     name_mapped = name.replace(weight_name, param_name)

#                     if is_pp_missing_parameter(name_mapped, self):
#                         continue

#                     param = params_dict[name_mapped]
#                     # We should ask the weight loader to return success or not
#                     # here since otherwise we may skip experts with other
#                     # available replicas.
#                     weight_loader = typing.cast(Callable[..., bool],
#                                                 param.weight_loader)
#                     success = weight_loader(param,
#                                             loaded_weight,
#                                             name_mapped,
#                                             shard_id=shard_id,
#                                             expert_id=expert_id,
#                                             return_success=True)
#                     if success:
#                         name = name_mapped
#                         break
#                 else:
#                     if is_expert_weight:
#                         # We've checked that this is an expert weight
#                         # However it's not mapped locally to this rank
#                         # So we simply skip it
#                         continue

#                     # Skip loading extra bias for GPTQ models.
#                     if name.endswith(".bias") and name not in params_dict:
#                         continue

#                     # Remapping the name of FP8 kv-scale.
#                     name = maybe_remap_kv_scale_name(name, params_dict)
#                     if name is None:
#                         continue

#                     if is_pp_missing_parameter(name, self):
#                         continue

#                     param = params_dict[name]
#                     weight_loader = getattr(param, "weight_loader",
#                                             default_weight_loader)
#                     weight_loader(param, loaded_weight)
#             loaded_params.add(name)

#         return loaded_params


# class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
#     pass


# def get_spec_layer_idx_from_weight_name(config: PretrainedConfig,
#                                         weight_name: str) -> Optional[int]:
#     if hasattr(config,
#                "num_nextn_predict_layers") and (config.num_nextn_predict_layers
#                                                 > 0):
#         layer_idx = config.num_hidden_layers
#         for i in range(config.num_nextn_predict_layers):
#             if weight_name.startswith(f"model.layers.{layer_idx+i}."):
#                 return layer_idx + i
#     return None
