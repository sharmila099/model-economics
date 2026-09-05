"""
reporter.py
===========
Aggregates raw benchmark results and produces three output files:
  1. report_<timestamp>.md  — Human-readable ranked Markdown table
  2. results_<timestamp>.json — Full structured results
  3. results_<timestamp>.csv  — Spreadsheet-ready flat data

The Markdown report includes:
  - Executive summary rankings table
  - Per-category breakdowns
  - Model recommendation based on configurable priority
  - Cost projection at 10K / 100K / 1M requests/month
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class AggregatedModelStats:
    model_id: str
    display_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0

    # Latency (ms)
    ttft_values: list[float] = field(default_factory=list)
    total_latency_values: list[float] = field(default_factory=list)

    # Tokens & Cost
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0

    # Quality
    quality_scores: list[float] = field(default_factory=list)

    # Per-category aggregates: {category: [scores]}
    category_quality: dict[str, list[float]] = field(default_factory=dict)
    category_costs: dict[str, list[float]] = field(default_factory=dict)

    @property
    def avg_ttft_ms(self) -> float:
        return statistics.mean(self.ttft_values) if self.ttft_values else 0.0

    @property
    def p95_ttft_ms(self) -> float:
        if not self.ttft_values:
            return 0.0
        sorted_v = sorted(self.ttft_values)
        idx = math.ceil(0.95 * len(sorted_v)) - 1
        return sorted_v[max(idx, 0)]

    @property
    def avg_total_latency_ms(self) -> float:
        return statistics.mean(self.total_latency_values) if self.total_latency_values else 0.0

    @property
    def avg_input_tokens(self) -> float:
        return self.total_input_tokens / max(self.successful_calls, 1)

    @property
    def avg_output_tokens(self) -> float:
        return self.total_output_tokens / max(self.successful_calls, 1)

    @property
    def cost_per_call_usd(self) -> float:
        return self.total_cost_usd / max(self.successful_calls, 1)

    @property
    def cost_per_1k_calls_usd(self) -> float:
        return self.cost_per_call_usd * 1000

    @property
    def avg_quality_score(self) -> Optional[float]:
        return round(statistics.mean(self.quality_scores), 2) if self.quality_scores else None

    @property
    def quality_per_dollar(self) -> Optional[float]:
        """Quality score per dollar spent per 1K calls. Higher is better."""
        if self.avg_quality_score and self.cost_per_1k_calls_usd > 0:
            return round(self.avg_quality_score / self.cost_per_1k_calls_usd, 3)
        return None


def _fmt_float(value: Optional[float], precision: int = 2, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}{suffix}"


def _rank_models(
    stats: list[AggregatedModelStats],
    priority: str = "quality_per_dollar",
) -> list[AggregatedModelStats]:
    """Rank models by the given priority. Lower rank = better."""
    key_map = {
        "quality_per_dollar": lambda s: -(s.quality_per_dollar or 0),
        "quality": lambda s: -(s.avg_quality_score or 0),
        "cost": lambda s: s.cost_per_1k_calls_usd,
        "latency": lambda s: s.avg_ttft_ms,
    }
    key_fn = key_map.get(priority, key_map["quality_per_dollar"])
    return sorted(stats, key=key_fn)


def _cost_projection(cost_per_call: float, volume: int) -> str:
    return f"\${cost_per_call * volume:,.2f}"


def generate_report(
    raw_results: list[dict],
    model_configs: list[dict],
    workload_config: dict,
    output_dir: Path,
    runs_per_prompt: int,
    quality_enabled: bool,
    priority: str = "quality_per_dollar",
) -> Path:
    """
    Main entry point. Aggregates raw results and writes all three output files.
    Returns the path to the Markdown report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Build a lookup from model_id → model_config
    config_by_id = {c["id"]: c for c in model_configs}

    # Aggregate raw results by model
    stats_by_model: dict[str, AggregatedModelStats] = {}
    for result in raw_results:
        mid = result["model_id"]
        if mid not in stats_by_model:
            cfg = config_by_id.get(mid, {})
            stats_by_model[mid] = AggregatedModelStats(
                model_id=mid,
                display_name=cfg.get("display_name", mid),
            )
        s = stats_by_model[mid]
        s.total_calls += 1

        if not result.get("success"):
            s.failed_calls += 1
            continue

        s.successful_calls += 1
        s.ttft_values.append(result.get("ttft_ms", 0))
        s.total_latency_values.append(result.get("total_latency_ms", 0))
        s.total_input_tokens += result.get("input_tokens", 0)
        s.total_output_tokens += result.get("output_tokens", 0)

        # Compute call cost
        cfg = config_by_id.get(mid, {})
        in_cost = (result.get("input_tokens", 0) / 1_000_000) * cfg.get("input_cost_per_1m", 0)
        out_cost = (result.get("output_tokens", 0) / 1_000_000) * cfg.get("output_cost_per_1m", 0)
        call_cost = in_cost + out_cost
        s.total_cost_usd += call_cost

        # Quality score
        quality = result.get("quality_score")
        if quality is not None:
            s.quality_scores.append(quality)

        # Per-category
        cat = result.get("category", "unknown")
        s.category_quality.setdefault(cat, [])
        s.category_costs.setdefault(cat, [])
        if quality is not None:
            s.category_quality[cat].append(quality)
        s.category_costs[cat].append(call_cost)

    all_stats = list(stats_by_model.values())
    ranked = _rank_models(all_stats, priority)

    # ── Markdown Report ──────────────────────────────────────────────────────
    prompts = workload_config.get("prompts", [])
    categories = sorted({p["category"] for p in prompts})
    n_prompts = len(prompts)

    lines: list[str] = []
    lines += [
        "# LLM Model Economics Report",
        "",
        f"> **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"> **Workload:** {n_prompts} prompts × {runs_per_prompt} runs each  ",
        f"> **Quality Scoring:** {'Enabled (LLM-as-a-Judge, Gemini Flash)' if quality_enabled else 'Disabled'}  ",
        f"> **Ranking Priority:** `{priority}`",
        "",
        "---",
        "",
        "## 🏆 Summary Rankings",
        "",
    ]

    # Table header
    quality_col = "Quality (/10)" if quality_enabled else "Quality"
    qpd_col = "Quality/\$1K" if quality_enabled else "Quality/\$1K"
    header = (
        "| Rank | Model | Cost/1K Calls | Avg TTFT | p95 TTFT | Total Latency"
        f" | Avg In Tok | Avg Out Tok | {quality_col} | {qpd_col} |"
    )
    sep = "|------|-------|---------------|----------|----------|---------------|-----------|-----------|--------------|-----------:|"
    lines += [header, sep]

    for i, s in enumerate(ranked, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}.")
        q_str = _fmt_float(s.avg_quality_score, 1) if quality_enabled else "—"
        qpd_str = _fmt_float(s.quality_per_dollar, 3) if quality_enabled else "—"
        lines.append(
            f"| {medal} | **{s.display_name}** "
            f"| \${s.cost_per_1k_calls_usd:.3f} "
            f"| {_fmt_float(s.avg_ttft_ms, 0, 'ms')} "
            f"| {_fmt_float(s.p95_ttft_ms, 0, 'ms')} "
            f"| {_fmt_float(s.avg_total_latency_ms, 0, 'ms')} "
            f"| {s.avg_input_tokens:.0f} "
            f"| {s.avg_output_tokens:.0f} "
            f"| {q_str} "
            f"| {qpd_str} |"
        )

    lines += ["", "---", "", "## 💰 Cost Projection at Scale", ""]
    proj_header = "| Model | 10K calls/mo | 100K calls/mo | 1M calls/mo |"
    proj_sep = "|-------|-------------|--------------|-------------|"
    lines += [proj_header, proj_sep]
    for s in ranked:
        lines.append(
            f"| {s.display_name} "
            f"| {_cost_projection(s.cost_per_call_usd, 10_000)} "
            f"| {_cost_projection(s.cost_per_call_usd, 100_000)} "
            f"| {_cost_projection(s.cost_per_call_usd, 1_000_000)} |"
        )

    if quality_enabled:
        lines += ["", "---", "", "## 📊 Quality by Task Category", ""]
        for cat in categories:
            lines += [f"### {cat.replace('_', ' ').title()}", ""]
            cat_header = "| Model | Avg Quality (/10) | Avg Cost/call |"
            cat_sep = "|-------|-------------------|--------------|"
            lines += [cat_header, cat_sep]
            for s in ranked:
                cat_q = s.category_quality.get(cat, [])
                cat_c = s.category_costs.get(cat, [])
                q_val = _fmt_float(statistics.mean(cat_q) if cat_q else None, 1)
                c_val = f"\${statistics.mean(cat_c):.5f}" if cat_c else "N/A"
                lines.append(f"| {s.display_name} | {q_val} | {c_val} |")
            lines.append("")

    # Recommendation
    lines += ["---", "", "## ✅ Recommendation", ""]
    if ranked:
        best_overall = ranked[0]
        cheapest = min(all_stats, key=lambda s: s.cost_per_call_usd)
        fastest = min(all_stats, key=lambda s: s.avg_ttft_ms)
        best_quality = (
            max(all_stats, key=lambda s: s.avg_quality_score or 0) if quality_enabled else None
        )

        lines += [
            f"- **Best overall** (quality per dollar): **{best_overall.display_name}**",
            f"- **Lowest cost**: **{cheapest.display_name}** "
            f"(\${cheapest.cost_per_1k_calls_usd:.3f}/1K calls)",
            f"- **Fastest TTFT**: **{fastest.display_name}** ({fastest.avg_ttft_ms:.0f}ms avg)",
        ]
        if best_quality:
            lines.append(
                f"- **Highest quality**: **{best_quality.display_name}** "
                f"({_fmt_float(best_quality.avg_quality_score, 1)}/10)"
            )

    report_path = output_dir / f"report_{timestamp}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # ── JSON Output ───────────────────────────────────────────────────────────
    json_output = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_prompts": n_prompts,
            "runs_per_prompt": runs_per_prompt,
            "quality_enabled": quality_enabled,
            "ranking_priority": priority,
        },
        "rankings": [
            {
                "rank": i + 1,
                "model_id": s.model_id,
                "display_name": s.display_name,
                "cost_per_1k_calls_usd": round(s.cost_per_1k_calls_usd, 4),
                "avg_ttft_ms": round(s.avg_ttft_ms, 1),
                "p95_ttft_ms": round(s.p95_ttft_ms, 1),
                "avg_total_latency_ms": round(s.avg_total_latency_ms, 1),
                "avg_input_tokens": round(s.avg_input_tokens, 1),
                "avg_output_tokens": round(s.avg_output_tokens, 1),
                "avg_quality_score": s.avg_quality_score,
                "quality_per_dollar": s.quality_per_dollar,
                "successful_calls": s.successful_calls,
                "failed_calls": s.failed_calls,
            }
            for i, s in enumerate(ranked)
        ],
        "raw_results": raw_results,
    }
    json_path = output_dir / f"results_{timestamp}.json"
    json_path.write_text(json.dumps(json_output, indent=2), encoding="utf-8")

    # ── CSV Output ────────────────────────────────────────────────────────────
    csv_path = output_dir / f"results_{timestamp}.csv"
    csv_fields = [
        "model_id", "prompt_id", "category", "run_index", "success",
        "input_tokens", "output_tokens", "ttft_ms", "total_latency_ms",
        "cost_usd", "quality_score", "error_message",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(raw_results)

    print(f"\n✅ Report written to: {report_path}")
    print(f"   JSON:              {json_path}")
    print(f"   CSV:               {csv_path}\n")

    return report_path
