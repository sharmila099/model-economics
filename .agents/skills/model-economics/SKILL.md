---
name: model-economics
description: >-
  Benchmarks a configurable set of LLM model endpoints (Gemini Flash, Gemini Pro,
  GPT-4o, Claude Sonnet, etc.) against a sample prompt workload. Measures TTFT,
  total latency, input/output token counts, dollar cost per call, and LLM-as-a-Judge
  quality scores. Outputs a ranked Markdown comparison table, a JSON results file,
  and a CSV for further analysis.
---

# Model-Economics Skill Runbook

This skill runs an automated, multi-model LLM benchmark and produces a ranked
cost/latency/quality comparison report. Use it whenever you need to make a
data-driven decision about which model to use for a production workload.

---

## When to Activate This Skill

Activate this skill when the user asks any of the following:
- "Which model should I use for this task?"
- "How much will this pipeline cost at scale?"
- "Compare Gemini vs GPT-4o vs Claude for my use case."
- "Run a cost benchmark."
- "Analyze model economics."
- "What is the cheapest model that meets quality bar X?"

---

## Pre-Flight Checklist

Before running the benchmark, confirm with the user:

1. **Which models to include?** (Default: all 6 in `config/models.yaml`)
2. **Which prompt workload to use?**
   - `default` — 5 task types covering summarization, reasoning, code gen, extraction, RAG-QA
   - `custom` — user provides their own prompt file path
3. **How many runs per prompt?** (Default: 3, for statistical stability)
4. **Quality scoring enabled?** (Default: yes — adds ~10s per prompt, costs ~$0.01 extra per run)
5. **API keys set?** Check for: `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`

---

## Execution Steps

### Step 1 — Validate Environment
Run the following and fix any missing dependencies before proceeding:
```bash
cd scripts
pip install -r requirements.txt
python benchmark.py --check-env
```

### Step 2 — Run the Benchmark
```bash
# Default: all models, default workload, 3 runs per prompt, quality scoring on
python scripts/benchmark.py

# Custom: specific models, custom prompts, 5 runs, no quality scoring
python scripts/benchmark.py \
  --models gemini-flash,gemini-pro,claude-sonnet \
  --prompts /path/to/my_prompts.yaml \
  --runs 5 \
  --no-quality
```

### Step 3 — Interpret & Present Results
After the script finishes, it writes three output files to `./benchmark_results/`:
- `report_<timestamp>.md` — Ranked Markdown table (show this to the user)
- `results_<timestamp>.json` — Full structured results for programmatic use
- `results_<timestamp>.csv` — Spreadsheet-ready data

Open `report_<timestamp>.md` and present it inline to the user.

### Step 4 — Recommend a Model
After presenting the table, apply this decision framework:

| User Priority | Recommended Pick |
|---|---|
| Lowest cost, quality ≥ 7/10 | Cheapest model where `avg_quality_score >= 7.0` |
| Lowest latency | Model with lowest `avg_ttft_ms` |
| Best quality, cost secondary | Highest `avg_quality_score` |
| Balanced (cost + quality) | Best `quality_per_dollar` rank |

Always call out the **crossover point**: at what monthly request volume does
the quality gap justify switching to a more expensive model?

---

## Error Handling

- If a model API call fails (rate limit, auth error), log the error, skip that
  model for that prompt, and continue. Do NOT abort the full benchmark.
- If quality scoring fails (judge model unavailable), mark quality as `N/A`
  and note it in the report.
- After the run, report a summary of any skipped calls.

---

## Output Format Reference

The Markdown report follows this structure:

```
# LLM Model-Economics Report
Generated: <timestamp>
Workload: <N prompts × M runs each>

## Summary Rankings

| Rank | Model | Avg Cost/1K calls | Avg TTFT (ms) | Avg Total (ms) | Quality Score | Quality/Dollar |
|------|-------|-------------------|---------------|----------------|---------------|----------------|
| ...  |  ...  |        ...        |      ...      |      ...       |      ...      |      ...       |

## Per-Task Breakdown
[Detailed tables per prompt category]

## Recommendations
[Model selection guidance based on results]
```
