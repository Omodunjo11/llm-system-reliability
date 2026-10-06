# LLM System Reliability

> Hand-rolled RAG pipeline from scratch: BM25 retrieval, multi-signal confidence calibration, grounded generation, and faithfulness-gated abstention when evidence is insufficient.

**Portfolio:** [lapoodunjo.com/projects/llm-reliability](https://lapoodunjo.com/projects/llm-reliability)

## The problem

Production LLM deployments fail when models answer confidently without sufficient evidence. Consumer AI optimizes for helpfulness; regulated contexts must optimize for correctness under uncertainty.

## What this is

A Python toolkit implementing the trust layer of a production LLM pipeline with **zero ML framework dependency** — no LangChain wrappers. Core modules are stdlib-only; `anthropic` is optional for live-model harness/drift checks.

### Core pipeline

```
User query
  → retrieval.py      BM25 ranked search over curated corpus
  → abstention.py     multi-signal confidence; refuse below threshold
  → generation.py     extractive grounded answer with citations
  → evaluation.py     faithfulness (precision-biased F1 + unsupported tokens)
  → pipeline.py       wires stages + post-generation faithfulness gate
```

### Reliability toolkit

```
src/
  pipeline.py              End-to-end grounded RAG + abstention gate
  retrieval.py             Okapi BM25 retrieval with score floor
  abstention.py            Multi-signal confidence calibration
  generation.py            Extractive grounded generation + citations
  evaluation.py            Faithfulness / hallucination-risk metrics
  drift_monitor.py         Output distribution shifts between model versions
  eval_harness.py          RAG + LLM regression harnesses
  failure_classifier.py    Classify LLM failures by type and severity

data/
  sample_docs.json         Curated compliance / regulatory corpus
  rag_eval_queries.json    Local RAG reliability cases (incl. abstention)
  golden_queries.json      Live-model golden prompt battery
```

## Key design decisions

- **BM25 over substring match** — graded relevance, not accidental phrase hits
- **Multi-signal confidence** — top-hit strength, score margin, term coverage, **IDF-weighted coverage**, evidence density
- **Hard abstain gates** — missing rare/OOV query terms force refusal (learned from adversarial breaks)
- **Extractive grounded answers** — synthesize only from retrieved sentences with source IDs
- **Faithfulness evaluator** — content-word precision/recall/F1 plus unsupported-token hallucination risk
- **Adversarial eval suite** — intentional break cases kept as regression tests
- **Modular architecture** — each stage is independently swappable

## Quick start

```bash
pip install -r requirements.txt

# Local RAG reliability harness (no API key required)
python -m src.eval_harness
python -m src.eval_harness --adversarial
python -m src.eval_harness --all

# equivalent CLI
python run.py eval
python run.py audit

# Interactive / one-shot grounded RAG demo
python run.py rag "What does KYC stand for and what is its purpose in financial services?"
python run.py rag "Tell me about Basel capital requirements for banks"

# Unit + adversarial regression tests
pytest -q
```

See `docs/AUDIT.md` for the break→harden cycle (partial lexical overlap failures and IDF hard gates).

### Optional live-model gates

```bash
export ANTHROPIC_API_KEY=...   # otherwise mock mode
python run.py snapshot         # save baseline responses
python run.py harness          # golden query regression
python run.py drift            # drift vs baseline
python run.py report           # latest report summaries
```

## Stack

Python · BM25 RAG · Abstention · Faithfulness evaluation · Drift monitoring

## Outcome

Complete four-stage RAG pipeline with calibrated abstention, faithfulness gating, and a runnable regression harness — built to demonstrate the trust layer above the model, not a framework wrapper.
