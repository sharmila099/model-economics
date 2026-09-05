# Model Economics 📊

An automated LLM benchmarking and cost-performance evaluation framework. Compare leading model endpoints (Google Gemini, OpenAI GPT-4o, Anthropic Claude, etc.) against standardized or custom prompt workloads to make data-driven model selection decisions.

---

## 🚀 Key Features

- **Multi-Provider Support**: Benchmark models across Google Gemini, OpenAI, and Anthropic Claude.
- **Full Latency Tracking**: Accurately measures Time-to-First-Token (TTFT), inter-token latency, and end-to-end total latency.
- **Cost Analytics**: Computes real-time per-call and per-1K calls dollar costs based on input and output token consumption and pricing tiers.
- **LLM-as-a-Judge Scoring**: Evaluates response quality (0–10 scale) using automated judge prompts.
- **Multi-Format Reporting**: Outputs ranked Markdown summary tables, structured JSON for pipelines, and CSV for spreadsheet analysis.

---

## 📂 Repository Structure

```
model-economics/
├── .agents/
│   └── skills/
│       └── model-economics/
│           ├── SKILL.md              # Antigravity skill runbook
│           ├── references/
│           │   └── pricing_guide.md  # LLM pricing snapshot & update guide
│           └── scripts/
│               ├── benchmark.py      # Main benchmark orchestrator & CLI
│               ├── reporter.py       # Report and table generator
│               ├── evaluator.py      # LLM-as-a-Judge scoring module
│               ├── requirements.txt  # Python package dependencies
│               ├── config/
│               │   ├── models.yaml   # Configured models and pricing
│               │   └── prompts.yaml  # Benchmark prompt suites
│               └── runners/          # Provider client implementations
│                   ├── gemini_runner.py
│                   ├── openai_runner.py
│                   └── anthropic_runner.py
└── .gitignore
```

---

## 🛠️ Quick Start

### 1. Prerequisites & Installation

```bash
cd .agents/skills/model-economics/scripts
pip install -r requirements.txt
```

### 2. Configure API Keys

Set the environment variables for the providers you want to test:

```bash
export GOOGLE_API_KEY="your-google-api-key"
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

*On Windows PowerShell:*
```powershell
$env:GOOGLE_API_KEY="your-google-api-key"
$env:OPENAI_API_KEY="your-openai-api-key"
$env:ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 3. Run Benchmark

```bash
# Default benchmark across all configured models
python scripts/benchmark.py

# Custom run with specific models and custom prompts
python scripts/benchmark.py \
  --models gemini-flash,gemini-pro,claude-sonnet \
  --prompts config/prompts.yaml \
  --runs 3 \
  --priority quality_per_dollar
```

---

## 📄 License

MIT
