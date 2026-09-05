# Pricing Reference Guide

> ⚠️ LLM pricing changes frequently. Update `config/models.yaml` whenever you see
> pricing changes at the provider dashboards below.

---

## How to Update Pricing

1. Open `.agents/skills/model-economics/scripts/config/models.yaml`
2. For each model, update `input_cost_per_1m` and `output_cost_per_1m` (USD per 1M tokens).
3. Re-run the benchmark — results will use the new pricing automatically.

---

## Pricing Sources (check these before each benchmark)

| Provider | Pricing Page |
|---|---|
| Google Gemini | https://ai.google.dev/pricing |
| OpenAI | https://openai.com/api/pricing |
| Anthropic Claude | https://www.anthropic.com/pricing |

---

## Current Pricing Snapshot (as of Sept 2025)

> These are the values currently set in `models.yaml`. Verify before trusting.

| Model | Input (/1M tokens) | Output (/1M tokens) |
|---|---|---|
| Gemini 2.0 Flash | \$0.10 | \$0.40 |
| Gemini 2.5 Pro (≤200k) | \$1.25 | \$10.00 |
| GPT-4o | \$2.50 | \$10.00 |
| GPT-4o Mini | \$0.15 | \$0.60 |
| Claude Sonnet 4.5 | \$3.00 | \$15.00 |
| Claude Haiku 3.5 | \$0.80 | \$4.00 |

---

## Context Caching Discounts

Some models offer significant discounts for cached input tokens:

| Model | Cached Input Price | Cache Storage (/1M/hr) |
|---|---|---|
| Gemini 2.0 Flash | \$0.025 (75% off) | \$1.00 |
| Gemini 2.5 Pro | \$0.3125 (75% off) | \$4.50 |
| Claude Sonnet | \$0.30 (90% off) | \$3.60 |
| GPT-4o | \$1.25 (50% off) | N/A |

> 💡 If your workload re-uses the same large system prompt across many queries,
> add a `--cache` flag to the benchmark to also model the cached cost scenario.

---

## Cost Estimation Formula

For any single API call:

$$\text{Cost} = \frac{\text{Input Tokens}}{1{,}000{,}000} \times \text{Input Price} + \frac{\text{Output Tokens}}{1{,}000{,}000} \times \text{Output Price}$$

At scale (e.g., 1M calls/month with avg 500 input + 300 output tokens):

$$\text{Monthly Cost} = 1{,}000{,}000 \times \left(\frac{500}{10^6} \times P_{in} + \frac{300}{10^6} \times P_{out}\right)$$
