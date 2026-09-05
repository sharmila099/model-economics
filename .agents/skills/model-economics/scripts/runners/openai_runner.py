"""
openai_runner.py
================
Runs benchmark prompts against OpenAI models (GPT-4o, GPT-4o mini, o3-mini, etc.)
using the openai SDK with streaming to capture TTFT.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

from openai import OpenAI

# Re-use the same result dataclass shape from gemini_runner for consistency
from runners.gemini_runner import ModelCallResult

logger = logging.getLogger(__name__)


class OpenAIRunner:
    """
    Executes a single prompt against an OpenAI chat completions endpoint.
    Streams the response to accurately measure Time-To-First-Token (TTFT).
    """

    def __init__(self, model_config: dict):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Get your key at: https://platform.openai.com/api-keys"
            )
        self.client = OpenAI(api_key=api_key)
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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            t_start = time.perf_counter()
            ttft_recorded = False
            output_chunks: list[str] = []
            input_tokens = 0
            output_tokens = 0

            with self.client.chat.completions.create(
                model=self.model_config["model_name"],
                messages=messages,
                max_tokens=self.model_config.get("max_output_tokens", 4096),
                temperature=0.0,
                stream=True,
                stream_options={"include_usage": True},  # Returns usage in final chunk
            ) as stream:
                for chunk in stream:
                    # Usage is delivered in the terminal chunk
                    if chunk.usage:
                        input_tokens = chunk.usage.prompt_tokens
                        output_tokens = chunk.usage.completion_tokens

                    if not chunk.choices:
                        continue

                    delta_content = chunk.choices[0].delta.content or ""
                    if delta_content and not ttft_recorded:
                        result.ttft_ms = (time.perf_counter() - t_start) * 1000
                        ttft_recorded = True
                        logger.debug(
                            "[%s] TTFT=%.1fms (prompt=%s, run=%d)",
                            self.model_config["id"],
                            result.ttft_ms,
                            prompt_id,
                            run_index,
                        )
                    output_chunks.append(delta_content)

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
