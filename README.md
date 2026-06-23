# LLM System Reliability

> Hand-rolled RAG pipeline from scratch: grounded retrieval, confidence scoring, and graceful abstention when evidence is insufficient.

**Portfolio:** [lapoodunjo.com/projects/llm-reliability](https://lapoodunjo.com/projects/llm-reliability)

## The problem

Production LLM deployments fail when models answer confidently without sufficient evidence. Consumer AI optimizes for helpfulness; regulated contexts must optimize for correctness under uncertainty.

## What this is

A Python toolkit implementing the trust layer of a production LLM pipeline with **zero framework dependency** — no LangChain wrappers.

### Core pipeline

```
User query
  → retrieval.py      keyword search over curated corpus
  → abstention.py     confidence scoring; refuse below threshold
  → generation.py     synthesize from retrieved docs only
  → evaluation.py     faithfulness as answer-to-context overlap
```

### Reliability toolkit

```
src/
  drift_monitor.py         Output distribution shifts between model versions
  eval_harness.py          Regression tests against golden query sets
  failure_classifier.py    Classify LLM failures by type and severity
  evaluation.py            Faithfulness and overlap metrics
  abstention.py            Confidence scoring and abstention logic
  generation.py            Response generation utilities

data/
  golden_queries.json      Curated prompt battery with acceptance criteria
```

## Key design decisions

- **Abstention gate** — below confidence threshold, refuse rather than guess
- **Modular architecture** — retrieval, abstention, generation, and evaluation are independently swappable
- **Faithfulness evaluator** — lightweight hallucination detection without external eval frameworks

## Quick start

```bash
pip install -r requirements.txt
python -m src.eval_harness
```

## Stack

Python · RAG · Abstention · Evaluation · Drift monitoring

## Outcome

Complete four-stage RAG pipeline with abstention gate, faithfulness evaluator, and regression harness — built to prove understanding of the trust layer above the model.
