"""
anthropic_runner.py
===================
Runs benchmark prompts against Anthropic models (Claude Sonnet, Haiku, etc.)
using the anthropic SDK with streaming to capture TTFT.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import anthropic

from runners.gemini_runner import ModelCallResult

logger = logging.getLogger(__name__)


class AnthropicRunner:
    """
    Executes a single prompt against an Anthropic Messages endpoint.
    Streams the response to accurately measure Time-To-First-Token (TTFT).
    Anthropic requires system prompt to be passed as a top-level parameter.
    """

    def __init__(self, model_config: dict):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Get your key at: https://console.anthropic.com/settings/keys"
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_config = model_config

    def run(
        self,
        prompt_id: str,
        system_prompt: str,
        user_prompt: str,
        run_index: int,
    ) -> ModelCallResult:
        result = ModelCallResult(
            model_id=self.model_config["id"],
            prompt_id=prompt_id,
            run_index=run_index,
            success=False,
        )

        try:
            t_start = time.perf_counter()
            ttft_recorded = False
            output_chunks: list[str] = []
            input_tokens = 0
            output_tokens = 0

            with self.client.messages.stream(
                model=self.model_config["model_name"],
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=self.model_config.get("max_output_tokens", 4096),
                temperature=0.0,
            ) as stream:
                for text_chunk in stream.text_stream:
                    if text_chunk and not ttft_recorded:
                        result.ttft_ms = (time.perf_counter() - t_start) * 1000
                        ttft_recorded = True
                        logger.debug(
                            "[%s] TTFT=%.1fms (prompt=%s, run=%d)",
                            self.model_config["id"],
                            result.ttft_ms,
                            prompt_id,
                            run_index,
                        )
                    output_chunks.append(text_chunk)

                # Usage metadata is available after stream completes
                final_message = stream.get_final_message()
                input_tokens = final_message.usage.input_tokens
                output_tokens = final_message.usage.output_tokens

            result.total_latency_ms = (time.perf_counter() - t_start) * 1000
            result.output_text = "".join(output_chunks)
            result.input_tokens = input_tokens
            result.output_tokens = output_tokens
            result.success = True

            logger.info(
                "[%s] OK | prompt=%s run=%d | in=%d out=%d | "
                "ttft=%.0fms total=%.0fms",
                self.model_config["id"],
                prompt_id,
                run_index,
                result.input_tokens,
                result.output_tokens,
                result.ttft_ms,
                result.total_latency_ms,
            )

        except Exception as exc:  # noqa: BLE001
            result.error_message = str(exc)
            logger.warning(
                "[%s] FAILED | prompt=%s run=%d | error=%s",
                self.model_config["id"],
                prompt_id,
                run_index,
                exc,
            )

        return result
