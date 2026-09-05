"""
evaluator.py
============
LLM-as-a-Judge quality evaluator.

For each (model_output, prompt_config) pair, it calls a fast judge model
(Gemini Flash by default) to score the output on 4 weighted dimensions:
  - Accuracy       (0.40 weight)
  - Completeness   (0.30 weight)
  - Format         (0.20 weight)
  - Conciseness    (0.10 weight)

Each dimension is scored 1–10. The final quality score is the weighted average.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

import google.generativeai as genai

logger = logging.getLogger(__name__)

JUDGE_PROMPT_TEMPLATE = """\
You are an impartial LLM output quality evaluator. Score the following model
output on four dimensions. Return ONLY a valid JSON object — no markdown, no
explanation outside the JSON.

## Original Prompt
SYSTEM:
{system_prompt}

USER:
{user_prompt}

## Expected Output Format
{expected_format}

## Model Output to Evaluate
{model_output}

## Scoring Instructions
Score each dimension from 1 (very poor) to 10 (excellent):

1. accuracy: Is the answer factually correct and does it follow all instructions precisely?
2. completeness: Does the response fully address all parts of the prompt without missing key elements?
3. format_compliance: Does the output match the expected format?
4. conciseness: Is the response concise without unnecessary filler?

Return exactly this JSON structure:
{{
  "accuracy": <int 1-10>,
  "completeness": <int 1-10>,
  "format_compliance": <int 1-10>,
  "conciseness": <int 1-10>,
  "reasoning": "<one sentence explaining the most significant issue, if any>"
}}
"""

DIMENSION_WEIGHTS = {
    "accuracy": 0.40,
    "completeness": 0.30,
    "format_compliance": 0.20,
    "conciseness": 0.10,
}


class LLMJudgeEvaluator:
    """
    Calls a judge model to score a given model output.

    Args:
        judge_model_config: Dict with 'model_name' and 'provider' keys.
    """

    def __init__(self, judge_model_config: dict):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is required for the judge model.")
        genai.configure(api_key=api_key)
        self.judge = genai.GenerativeModel(
            model_name=judge_model_config["model_name"],
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=512,
            ),
        )

    def score(
        self,
        system_prompt: str,
        user_prompt: str,
        expected_format: str,
        model_output: str,
    ) -> dict:
        """
        Returns a dict with dimension scores (1-10), weighted_score (float),
        and reasoning (str). On failure, returns zeroed scores with error detail.
        """
        if not model_output.strip():
            return self._empty_score(reason="Model produced no output.")

        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            system_prompt=system_prompt,
            user_prompt=user_prompt[:2000],  # Truncate long prompts for judge
            expected_format=expected_format,
            model_output=model_output[:3000],  # Truncate long outputs for judge
        )

        try:
            response = self.judge.generate_content(judge_prompt)
            raw_text = response.text.strip()

            # Strip markdown code fences if present
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

            scores = json.loads(raw_text)
            weighted = sum(
                scores.get(dim, 0) * weight
                for dim, weight in DIMENSION_WEIGHTS.items()
            )
            scores["weighted_score"] = round(weighted, 2)
            return scores

        except json.JSONDecodeError as e:
            logger.warning("Judge returned invalid JSON: %s | raw=%s", e, raw_text[:200])
            return self._empty_score(reason=f"JSON parse error: {e}")
        except Exception as e:  # noqa: BLE001
            logger.warning("Judge call failed: %s", e)
            return self._empty_score(reason=str(e))

    @staticmethod
    def _empty_score(reason: str) -> dict:
        return {
            "accuracy": 0,
            "completeness": 0,
            "format_compliance": 0,
            "conciseness": 0,
            "weighted_score": 0.0,
            "reasoning": reason,
        }
