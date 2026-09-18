# ai-eval-harness

[![Tests](https://github.com/kallurayaankit/ai-eval-harness/actions/workflows/test.yml/badge.svg)](https://github.com/kallurayaankit/ai-eval-harness/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

An AI output-evaluation harness with traceability reports. Runs entirely on your machine with [Ollama](https://ollama.ai). No API keys, no cloud bills, no vendor lock-in.

Every score in every report carries its full provenance: judge model, prompt hash, rubric version, evidence, latency. You can audit any verdict down to the exact prompt that produced it.

## What it does

Three things, cleanly separated:

1. **Tracing** — every eval run produces a `Trace` with spans (retrieval, generation, judge calls) and attached scores
2. **Evaluation** — pluggable metric classes: deterministic (instant) and LLM-as-a-judge (semantic)
3. **Reporting** — one self-contained HTML file per run, with a summary, failures section, and per-trace drill-down

## Quickstart

```bash
# 1. Install Ollama and pull models
ollama pull llama3.2:3b
ollama pull mistral:latest

# 2. Set up the project
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -e .

# 3. Get the datasets (separate repo)
git clone https://github.com/kallurayaankit/eval-datasets.git
set DATASETS_DIR=C:\path\to\eval-datasets    # Windows
export DATASETS_DIR=/path/to/eval-datasets   # Linux/macOS

# 4. Run an eval
python run_qa.py

The report lands in reports/. Open it in any browser.
Metrics
Metric	Type	What it measures
exact_match	Deterministic	Output equals reference
contains	Deterministic	Reference appears in output
regex_match	Deterministic	Output matches a pattern
valid_json	Deterministic	Output parses as JSON
correctness	Judge	Semantic match against reference
faithfulness	Judge	Answer grounded in retrieved context
relevance	Judge	Answer addresses the question
hallucination	Judge	Unsupported claims (inverted — high = clean)
toxicity	Judge	Harmful content (inverted — high = safe)
pii	Judge	PII exposure (inverted — high = clean)
tool_name_match	Deterministic	Expected tool name appears
tool_selection	Judge	Agent picked the right tool
Real results

Measured against llama3.2:3b (target model) and mistral:latest (judge).
QA — 10 examples, 3 metrics
Metric	Mean	Pass rate
exact_match	0.60	60%
contains	1.00	100%
correctness	1.00	100%

Finding: exact_match fails on punctuation. "Paris." vs "Paris". That's why contains and correctness exist — exact match is a poor primary metric.
RAG — 5 clinical FAQ examples, 6 metrics
Metric	Mean	Pass rate
faithfulness	1.00	100%
relevance	1.00	100%
hallucination	1.00	100%
toxicity	1.00	100%
pii	1.00	100%
correctness	0.98	100%

Finding: llama3.2:3b handled this dataset cleanly. The examples were too easy — the model had the answer in the retrieved context and simply echoed it. A harder RAG dataset with distractors and contradictions would stress this more.
Agents — 5 tool-selection examples, 2 metrics
Metric	Mean	Pass rate
tool_name_match	0.80	80%
tool_selection	0.80	80%

Finding: The model failed on ag-003 — "Find the current CEO of OpenAI" with tools get_weather, search_web, send_email. Instead of picking search_web, it tried to answer from memory. Classic small-model behavior: prefer parametric recall over tool use.

This is the kind of failure the harness is designed to catch.
Traceability

Every score is auditable. In the HTML report, each score shows:

    The judge model used (mistral:latest)

    A hash of the judge prompt (6dec89c29165)

    The rubric version (v1)

    The judge's reasoning

    The latency of the call

Reproduce a verdict: pin the rubric version, re-run with the same judge, diff the score.
Architecture
text

src/ai_eval/
├── client.py           # Ollama HTTP client
├── trace.py            # Trace, Span, Score data model
├── dataset.py          # JSONL loader with DATASETS_DIR support
├── batch.py            # Parallel batch runner
├── report.py           # HTML report renderer
├── metrics/
│   ├── base.py         # Metric ABC
│   ├── deterministic.py # exact_match, contains, regex, json
│   ├── judge.py        # correctness
│   ├── judge_base.py   # shared judge infra + cache
│   ├── semantic.py     # faithfulness, relevance, hallucination
│   ├── safety.py       # toxicity, PII
│   └── agent.py        # tool_name_match, tool_selection
└── templates/
    ├── report.html
    └── trace.html

Configuration

.env:
text

OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
JUDGE_MODEL=mistral:latest
EVAL_WORKERS=2
DATASETS_DIR=/path/to/eval-datasets

EVAL_WORKERS controls parallelism. Start at 2. Each example runs concurrently, and each metric within an example runs concurrently too. On a 16 GB machine, 2 is safe. On 32 GB+, 4–6 works.

Judge verdicts are cached to ~/.cache/ai_eval/judge_cache.json keyed by (prompt, model). Re-runs are near-instant for cached pairs.
Prior art

Design patterns borrowed from:

    DeepEval — G-Eval, structured judge verdicts

    Phoenix — OpenTelemetry-native tracing

    Opik — full observability lifecycle

    harness-evals — five-dimension evaluation, pytest integration

    evaltrace — review queue for low-scoring traces

Where those tools require cloud APIs, this one runs on a laptop with a local model.