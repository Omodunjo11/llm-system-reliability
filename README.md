# LLM System Reliability

> Python toolkit for measuring and improving reliability in production LLM systems. Covers drift detection across model updates, regression test harnesses, and failure mode classification — calibrated for regulated-industry deployment contexts.

---

## Why This Exists

Consumer AI reliability is measured by user satisfaction. Regulated industry reliability is measured by whether the output would survive regulatory scrutiny — a different standard entirely.

A model that confidently states a non-existent regulation, refuses a standard compliance query, or silently changes its output format isn't just unhelpful. In finance, healthcare, or legal contexts, it's a liability. Most teams catch these failures in post-mortems. This toolkit catches them before deployment.

---

## Components

```
src/
  drift_monitor.py         Detects output distribution shifts between model versions
  eval_harness.py          Regression test harness against golden query sets
  failure_classifier.py    Classifies LLM failures by type and severity
  evaluation.py            Core evaluation metrics (faithfulness, overlap)
  abstention.py            Confidence scoring and abstention logic
  generation.py            Response generation utilities

data/
  golden_queries.json      Curated prompt battery with acceptance criteria

outputs/                   Generated reports (gitignored)
  drift_report.json
  harness_report.json
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Save a baseline snapshot of current model responses
python run.py snapshot

# Run regression harness against golden queries
python run.py harness

# Check for drift against saved baseline
python run.py drift

# Print latest report summaries
python run.py report
```

Set `ANTHROPIC_API_KEY` in your environment. Without it, the toolkit runs in mock mode for testing.

---

## Drift Monitor

`src/drift_monitor.py` compares model responses across two snapshots and produces a structured drift report.

**How it works:**
1. Two sets of prompt→response pairs are compared using token-level Jaccard similarity
2. Failure types are classified per response pair (refusal drift, format degradation, length collapse, confidence miscalibration)
3. Severity is assigned based on similarity score AND failure type — high-severity failure types escalate severity independent of score
4. Report fails the gate if `drift_rate > 10%` or `critical_drift_rate > 2%`

```python
from src.drift_monitor import run_drift_check, print_summary

report = run_drift_check(
    baseline={"q1": "response from v1", "q2": "response from v1"},
    current={"q1": "response from v2", "q2": "response from v2"},
    baseline_version="claude-3-5-sonnet-20241022",
    current_version="claude-3-7-sonnet-20250219",
)
print_summary(report)
```

**Drift thresholds:**

| Metric | Threshold | Action |
|--------|-----------|--------|
| Similarity score | < 0.75 | Minor drift |
| Similarity score | < 0.50 | Moderate drift |
| Similarity score | < 0.25 | Critical drift |
| Drift rate | > 10% | Gate failure |
| Critical drift rate | > 2% | Immediate flag |

---

## Eval Harness

`src/eval_harness.py` runs a curated set of golden queries against the live model and checks acceptance criteria.

**Golden query structure** (`data/golden_queries.json`):
```json
{
  "id": "factual_basic_01",
  "prompt": "What is the primary purpose of the SEC's Regulation Fair Disclosure?",
  "category": "factual",
  "severity": "critical",
  "description": "Tests factual accuracy on a well-known regulation",
  "must_contain": ["material", "disclosure", "public"],
  "must_not_contain": ["i cannot", "i'm unable"],
  "min_length": 30,
  "max_length": 500,
  "expected_format": null
}
```

**Gate logic:** Harness fails if `pass_rate < 90%` OR any `critical` severity query fails.

---

## Failure Taxonomy

`src/failure_classifier.py` classifies failures against a structured taxonomy designed from production incident logs. Each failure type has a `regulated_impact` field explaining why it matters specifically in compliance contexts.

| Code | Severity | Description |
|------|----------|-------------|
| `hallucination_factual` | CRITICAL | Model states false facts with confidence |
| `hallucination_source` | CRITICAL | Model cites non-existent sources/regulations |
| `refusal_drift` | HIGH | Model refuses tasks it previously handled |
| `confidence_miscalibration` | HIGH | Inappropriate certainty on uncertain topics |
| `instruction_following_failure` | HIGH | Ignores explicit prompt instructions |
| `format_degradation` | MEDIUM | Expected format (JSON, list) not produced |
| `scope_creep` | MEDIUM | Answers a different question than asked |
| `length_collapse` | MEDIUM | Response significantly truncated |

---

## Design Decisions

**Why severity is calibrated for regulated contexts, not CSAT:**
A confidence miscalibration on a creative writing task is low stakes. The same failure on a regulatory interpretation query — where an analyst acts on the model's stated certainty — can trigger an audit finding. Severity scores here reflect the former standard.

**Why token-level Jaccard over embedding similarity:**
Embedding similarity is more semantically nuanced but introduces a dependency on an embedding model and makes the drift threshold less interpretable. Token overlap is fast, deterministic, and produces a score that's easy to reason about in a compliance review. For production deployment, a hybrid approach is recommended.

**Why golden queries are versioned as a first-class artifact:**
The golden query set is the definition of "correct behaviour". It should go through review, versioning, and approval the same way a product spec does — not be a developer's private test list.

---

## Related

- [kinage-intelligence](https://github.com/Omodunjo11/kinage-intelligence) — Next.js intelligence dashboard using LLMs for signal analysis
- [Kinage-AL-](https://github.com/Omodunjo11/Kinage-AL-) — Python AI backend with Claude-powered summarisation
