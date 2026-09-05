"""
gemini_runner.py
================
Runs benchmark prompts against Google Gemini models using the
google-generativeai SDK. Captures TTFT (Time-To-First-Token),
total latency, input/output token counts, and raw text output.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import google.generativeai as genai

logger = logging.getLogger(__name__)


@dataclass
class ModelCallResult:
    model_id: str
    prompt_id: str
    run_index: int
    success: bool
    output_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float = 0.0          # Time to first token (streaming)
    total_latency_ms: float = 0.0
    error_message: str = ""
    raw_response: dict = field(default_factory=dict)


class GeminiRunner:
    """
    Executes a single prompt against a Gemini model endpoint.
    Uses streaming to accurately capture TTFT.
    """

    def __init__(self, model_config: dict):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Get your key at: https://aistudio.google.com/app/apikey"
            )
        genai.configure(api_key=api_key)
        self.model_config = model_config
        self.model = genai.GenerativeModel(
            model_name=model_config["model_name"],
            generation_config=genai.GenerationConfig(
                max_output_tokens=model_config.get("max_output_tokens", 8192),
                temperature=0.0,  # Deterministic for benchmarking
            ),
        )

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

        full_contents = [
            {"role": "user", "parts": [f"{system_prompt}\n\n{user_prompt}"]},
        ]

        try:
            t_start = time.perf_counter()
            ttft_recorded = False
            output_chunks: list[str] = []

            # Stream the response to capture TTFT
            response_stream = self.model.generate_content(
                full_contents,
                stream=True,
            )

            for chunk in response_stream:
                if not ttft_recorded and chunk.text:
                    result.ttft_ms = (time.perf_counter() - t_start) * 1000
                    ttft_recorded = True
                    logger.debug(
                        "[%s] TTFT=%.1fms (prompt=%s, run=%d)",
                        self.model_config["id"],
                        result.ttft_ms,
                        prompt_id,
                        run_index,
                    )
                if chunk.text:
                    output_chunks.append(chunk.text)

            result.total_latency_ms = (time.perf_counter() - t_start) * 1000
            result.output_text = "".join(output_chunks)

            # Resolve usage metadata from the final aggregated response
            response_stream.resolve()
            usage = response_stream.usage_metadata
            result.input_tokens = usage.prompt_token_count or 0
            result.output_tokens = usage.candidates_token_count or 0
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
