# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm/vllm/entrypoints/openai/serving_chat.py

"""Anthropic Messages API serving handler"""
import asyncio
import copy
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, Final, AsyncIterator
import uuid
import jinja2

from fastapi import Request

from vllm import SamplingParams, RequestOutput
from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicDelta,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicStreamEvent,
    AnthropicUsage,
)
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption, ConversationMessage, ChatCompletionMessageParam
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import ErrorResponse, RequestResponseMetadata

from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.utils import get_max_tokens
from vllm.sampling_params import BeamSearchParams
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = logging.getLogger(__name__)


class AnthropicServingMessages(OpenAIServing):
    """Handler for Anthropic Messages API requests"""

    def __init__(
            self,
            engine_client: EngineClient,
            model_config: ModelConfig,
            models: OpenAIServingModels,
            response_role: str,
            *,
            request_logger: Optional[RequestLogger],
            chat_template: Optional[str],
            chat_template_content_format: ChatTemplateContentFormatOption,
            return_tokens_as_token_ids: bool = False,
            reasoning_parser: str = "",
            enable_auto_tools: bool = False,
            exclude_tools_when_tool_choice_none: bool = False,
            tool_parser: Optional[str] = None,
            enable_prompt_tokens_details: bool = False,
            enable_force_include_usage: bool = False,
    ):
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger,
                         return_tokens_as_token_ids=return_tokens_as_token_ids,
                         enable_force_include_usage=enable_force_include_usage)

        self.response_role = response_role
        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_force_include_usage = enable_force_include_usage
        self.default_sampling_params = (
            self.model_config.get_diff_sampling_param())
        if self.default_sampling_params:
            source = self.model_config.generation_config
            source = "model" if source == "auto" else source
            logger.info("Using default chat sampling params from %s: %s",
                        source, self.default_sampling_params)

    async def create_messages(
            self,
            request: AnthropicMessagesRequest,
            raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], AnthropicMessagesResponse,
    ErrorResponse]:
        """
        Messages API similar to Anthropic's API.

        See https://docs.anthropic.com/en/api/messages
        for the API specification. This API mimics the Anthropic messages API.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        if self.engine_client.errored:
            raise self.engine_client.dead_error

        try:
            model_name = self._get_model_name(request.model)

            tokenizer = await self.engine_client.get_tokenizer()

            if request.system is not None:
                system_message = ChatCompletionMessageParam(
                    role="system",
                    content=request.system,
                )
                request.messages = [system_message] + request.messages

            (
                conversation,
                request_prompts,
                engine_prompts,
            ) = await self._preprocess_chat(
                request,
                tokenizer,
                request.messages,
                chat_template=request.chat_template or self.chat_template,
                chat_template_content_format=self.chat_template_content_format,
                tool_dicts=None,
                chat_template_kwargs=request.chat_template_kwargs,
                tool_parser=None,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
            )

        except (ValueError, TypeError, RuntimeError,
                jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(f"{e} {e.__cause__}")

        request_id = "chatcmpl-" \
                     f"{self._base_request_id(raw_request, request.request_id)}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                sampling_params: Union[SamplingParams, BeamSearchParams]

                if self.default_sampling_params is None:
                    self.default_sampling_params = {}

                max_tokens = get_max_tokens(
                    max_model_len=self.max_model_len,
                    request=request,
                    input_length=len(engine_prompt["prompt_token_ids"]),
                    default_sampling_params=self.default_sampling_params)

                sampling_params = request.to_sampling_params(max_tokens, self.default_sampling_params)

                self._log_inputs(
                    request_id,
                    request_prompts[i],
                    params=sampling_params,
                    lora_request=None,
                )

                trace_headers = (None if raw_request is None else await
                self._get_trace_headers(raw_request.headers))

                generator = self.engine_client.generate(
                    engine_prompt,
                    sampling_params,
                    request_id,
                    trace_headers=trace_headers,
                )
                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert len(generators) == 1
        result_generator, = generators

        # Streaming response
        if request.stream:
            return self.message_stream_generator(
                result_generator,
                request_id,
                model_name,
            )

        try:
            return await self.messages_full_generator(
                result_generator,
                request_id,
                model_name,
            )
        except ValueError as e:
            return self.create_error_response(str(e))

    async def messages_full_generator(
            self,
            result_generator: AsyncIterator[RequestOutput],
            request_id: str,
            model_name: str,
    ) -> Union[ErrorResponse, AnthropicMessagesResponse]:

        final_res: Optional[RequestOutput] = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert final_res is not None

        # Should be only one output
        assert len(final_res.outputs) == 1
        output = final_res.outputs[0]

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs)

        # Tool calls not supported currently
        content = output.text
        stop_reason = "end_turn"
        if output.finish_reason == "length":
            stop_reason = "max_tokens"
        elif output.finish_reason == "stop":
            stop_reason = "stop_sequence"

        message = AnthropicMessagesResponse(
            content=[
                AnthropicContentBlock(
                    type="text",
                    text=content,
                )],
            id=request_id,
            model=model_name,
            stop_reason=stop_reason,
            usage=AnthropicUsage(
                input_tokens=num_prompt_tokens,
                output_tokens=num_generated_tokens,
            ),
        )
        return message

    async def message_stream_generator(
            self,
            result_generator: AsyncIterator[RequestOutput],
            request_id: str,
            model_name: str,
    ) -> AsyncGenerator[str, None]:

        # Send message_start event
        chunk = AnthropicStreamEvent(
            type="message_start",
            messages=AnthropicMessagesResponse(
                id=request_id,
                content=[],
                model=model_name,
                usage=AnthropicUsage(
                    input_tokens=0,
                    output_tokens=0,
                )
            )
        )
        data = chunk.model_dump_json(exclude_unset=True)
        yield f"data: {data}\n\n"

        # Send content_block_start event
        chunk = AnthropicStreamEvent(
            type="content_block_start",
            index=0,
            content_block=AnthropicContentBlock(
                type="text",
                text=""
            ),
        )
        data = chunk.model_dump_json(exclude_unset=True)
        yield f"data: {data}\n\n"

        accumulated_text = ""
        input_tokens = 0
        output_tokens = 0
        try:
            async for res in result_generator:
                if res.prompt_token_ids is not None:
                    input_tokens = len(res.prompt_token_ids)

                for output in res.outputs:
                    delta_text = output.text
                    accumulated_text += delta_text
                    output_tokens += len(output.token_ids)

                    if not delta_text and not output.token_ids:
                        # Chunked prefill case, don't return empty chunks
                        continue

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        content_delta = AnthropicStreamEvent(
                            type="content_block_delta",
                            index=0,
                            delta=AnthropicDelta(
                                type="text_delta",
                                text=delta_text
                            )
                        )

                    # if the model is finished generating
                    else:
                        content_delta = AnthropicStreamEvent(
                            type="content_block_stop",
                            index=0,
                        )

                    data = content_delta.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            final_chunk = AnthropicStreamEvent(
                type="message_stop",
                message=AnthropicMessagesResponse(
                    id=request_id,
                    content=[
                        AnthropicContentBlock(
                            type="text",
                            text=accumulated_text,
                        )
                    ],
                    model=model_name,
                    usage=AnthropicUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    ),
                )
            )
            final_chunk_data = final_chunk.model_dump_json(exclude_unset=True, exclude_none=True)
            yield f"data: {final_chunk_data}\n\n"

        except Exception as e:
            # TODO: Use a vllm-specific Validation Error
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"
