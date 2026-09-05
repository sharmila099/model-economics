"""
benchmark.py
============
Main orchestrator for the LLM Model Economics Skill.

Usage:
  python benchmark.py [OPTIONS]

Options:
  --models      Comma-separated list of model IDs to benchmark.
                Defaults to all enabled models in config/models.yaml.
                Example: --models gemini-flash,gemini-pro,claude-sonnet
  --prompts     Path to a custom prompts YAML file.
                Defaults to config/prompts.yaml.
  --runs        Number of runs per prompt (default: 3).
  --no-quality  Disable LLM-as-a-Judge quality scoring.
  --priority    Ranking priority: quality_per_dollar (default), quality, cost, latency.
  --output-dir  Directory for output files (default: ./benchmark_results).
  --check-env   Validate that required environment variables and packages exist.
  --verbose     Enable DEBUG-level logging.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("benchmark")

# ── Constants ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DEFAULT_MODELS_CONFIG = SCRIPT_DIR / "config" / "models.yaml"
DEFAULT_PROMPTS_CONFIG = SCRIPT_DIR / "config" / "prompts.yaml"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "benchmark_results"

PROVIDER_RUNNER_MAP = {
    "google": ("runners.gemini_runner", "GeminiRunner"),
    "openai": ("runners.openai_runner", "OpenAIRunner"),
    "anthropic": ("runners.anthropic_runner", "AnthropicRunner"),
}


# ── Environment Check ─────────────────────────────────────────────────────────
def check_environment() -> bool:
    """Validate all required packages and env vars. Returns True if OK."""
    import importlib
    import os

    ok = True
    checks = [
        ("google.generativeai", "GOOGLE_API_KEY", "pip install google-generativeai"),
        ("openai", "OPENAI_API_KEY", "pip install openai"),
        ("anthropic", "ANTHROPIC_API_KEY", "pip install anthropic"),
    ]

    print("\n🔍 Environment Check\n" + "─" * 40)
    for package, env_var, install_cmd in checks:
        pkg_ok = bool(importlib.util.find_spec(package))
        env_ok = bool(os.environ.get(env_var))
        status = "✅" if (pkg_ok and env_ok) else "❌"
        print(
            f"{status} {package:30s} | "
            f"{'package OK' if pkg_ok else f'MISSING ({install_cmd})'} | "
            f"{'env OK' if env_ok else f'{env_var} not set'}"
        )
        if not (pkg_ok and env_ok):
            ok = False

    print("─" * 40)
    print(f"{'✅ All checks passed.' if ok else '❌ Fix the issues above before running.'}\n")
    return ok


# ── Runner Factory ────────────────────────────────────────────────────────────
def get_runner(model_config: dict):
    provider = model_config.get("provider", "").lower()
    if provider not in PROVIDER_RUNNER_MAP:
        raise ValueError(f"Unsupported provider '{provider}' for model '{model_config['id']}'")
    module_path, class_name = PROVIDER_RUNNER_MAP[provider]
    import importlib
    module = importlib.import_module(module_path)
    runner_class = getattr(module, class_name)
    return runner_class(model_config)


# ── Main Orchestrator ─────────────────────────────────────────────────────────
def run_benchmark(
    models_config_path: Path,
    prompts_config_path: Path,
    selected_model_ids: list[str] | None,
    runs: int,
    quality_enabled: bool,
    priority: str,
    output_dir: Path,
) -> Path:
    # Load configs
    with models_config_path.open() as f:
        models_cfg = yaml.safe_load(f)
    with prompts_config_path.open() as f:
        prompts_cfg = yaml.safe_load(f)

    all_models = models_cfg.get("models", [])
    judge_cfg = models_cfg.get("judge_model", {})
    prompts = prompts_cfg.get("prompts", [])

    # Filter models
    if selected_model_ids:
        models_to_run = [m for m in all_models if m["id"] in selected_model_ids]
        missing = set(selected_model_ids) - {m["id"] for m in models_to_run}
        if missing:
            logger.warning("Unknown model IDs (will skip): %s", missing)
    else:
        models_to_run = [m for m in all_models if m.get("enabled", True)]

    if not models_to_run:
        logger.error("No models to benchmark. Check your --models argument or models.yaml.")
        sys.exit(1)

    logger.info(
        "Starting benchmark | models=%d prompts=%d runs_per_prompt=%d quality=%s",
        len(models_to_run),
        len(prompts),
        runs,
        quality_enabled,
    )

    # Initialize judge
    judge = None
    if quality_enabled:
        from evaluator import LLMJudgeEvaluator
        try:
            judge = LLMJudgeEvaluator(judge_cfg)
            logger.info("LLM-as-a-Judge initialized: %s", judge_cfg.get("display_name"))
        except Exception as e:
            logger.warning("Judge initialization failed (%s). Quality scoring disabled.", e)
            quality_enabled = False

    raw_results: list[dict] = []
    total_calls = len(models_to_run) * len(prompts) * runs
    call_count = 0

    for model_cfg in models_to_run:
        logger.info("── Model: %s ─────────────────────────", model_cfg["display_name"])
        try:
            runner = get_runner(model_cfg)
        except EnvironmentError as e:
            logger.warning("Skipping %s: %s", model_cfg["id"], e)
            continue
        except Exception as e:
            logger.warning("Runner init failed for %s: %s", model_cfg["id"], e)
            continue

        for prompt in prompts:
            for run_idx in range(1, runs + 1):
                call_count += 1
                progress = f"[{call_count}/{total_calls}]"
                logger.info(
                    "%s %s | prompt=%s | run=%d/%d",
                    progress,
                    model_cfg["display_name"],
                    prompt["id"],
                    run_idx,
                    runs,
                )

                result = runner.run(
                    prompt_id=prompt["id"],
                    system_prompt=prompt["system"],
                    user_prompt=prompt["user"],
                    run_index=run_idx,
                )

                # Compute cost
                in_cost = (result.input_tokens / 1_000_000) * model_cfg.get("input_cost_per_1m", 0)
                out_cost = (result.output_tokens / 1_000_000) * model_cfg.get("output_cost_per_1m", 0)

                result_dict = {
                    "model_id": result.model_id,
                    "prompt_id": result.prompt_id,
                    "category": prompt.get("category", "unknown"),
                    "run_index": result.run_index,
                    "success": result.success,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "ttft_ms": round(result.ttft_ms, 2),
                    "total_latency_ms": round(result.total_latency_ms, 2),
                    "cost_usd": round(in_cost + out_cost, 7),
                    "output_text": result.output_text,
                    "error_message": result.error_message,
                    "quality_score": None,
                    "quality_details": None,
                }

                # Quality scoring (only on first run to save cost)
                if quality_enabled and judge and result.success and run_idx == 1:
                    logger.debug("Scoring quality for %s / %s...", model_cfg["id"], prompt["id"])
                    scores = judge.score(
                        system_prompt=prompt["system"],
                        user_prompt=prompt["user"],
                        expected_format=prompt.get("expected_format", ""),
                        model_output=result.output_text,
                    )
                    result_dict["quality_score"] = scores.get("weighted_score")
                    result_dict["quality_details"] = scores

                raw_results.append(result_dict)

                # Small courtesy delay to avoid rate limits
                time.sleep(0.5)

    # Generate outputs
    from reporter import generate_report

    with models_config_path.open() as f:
        models_cfg_full = yaml.safe_load(f)

    report_path = generate_report(
        raw_results=raw_results,
        model_configs=models_cfg_full["models"],
        workload_config=prompts_cfg,
        output_dir=output_dir,
        runs_per_prompt=runs,
        quality_enabled=quality_enabled,
        priority=priority,
    )
    return report_path


# ── CLI Entry Point ───────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM Model Economics: Compare models on latency, cost, and quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model IDs (e.g. gemini-flash,claude-sonnet). Default: all enabled.",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS_CONFIG,
        help=f"Path to prompts YAML. Default: {DEFAULT_PROMPTS_CONFIG}",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per prompt for statistical stability. Default: 3.",
    )
    parser.add_argument(
        "--no-quality",
        action="store_true",
        help="Disable LLM-as-a-Judge quality scoring (faster and cheaper).",
    )
    parser.add_argument(
        "--priority",
        choices=["quality_per_dollar", "quality", "cost", "latency"],
        default="quality_per_dollar",
        help="Ranking criterion for the summary table. Default: quality_per_dollar.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Validate environment (packages + API keys) and exit.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    if args.check_env:
        ok = check_environment()
        sys.exit(0 if ok else 1)

    selected = [m.strip() for m in args.models.split(",")] if args.models else None

    report = run_benchmark(
        models_config_path=DEFAULT_MODELS_CONFIG,
        prompts_config_path=args.prompts,
        selected_model_ids=selected,
        runs=args.runs,
        quality_enabled=not args.no_quality,
        priority=args.priority,
        output_dir=args.output_dir,
    )

    print(f"\n📄 Open your report: {report}")
