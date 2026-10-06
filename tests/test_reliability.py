"""Unit tests for retrieval, confidence, faithfulness, and RAG harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.abstention import estimate_confidence, should_abstain
from src.eval_harness import load_rag_eval_queries, run_rag_harness
from src.evaluation import evaluate_faithfulness
from src.generation import generate_grounded
from src.pipeline import ReliabilityPipeline
from src.retrieval import BM25Index, load_docs, simple_search


ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = str(ROOT / "data" / "sample_docs.json")


@pytest.fixture(scope="module")
def docs():
    return load_docs(DOCS_PATH)


@pytest.fixture(scope="module")
def pipeline(docs):
    return ReliabilityPipeline.from_docs(docs=docs)


def test_bm25_ranks_relevant_doc_above_noise(docs):
    results = simple_search("Suspicious Activity Report Bank Secrecy Act", docs, min_score=0.1)
    assert results
    top_ids = [r["id"] for r in results[:3]]
    assert any(i.startswith("doc_bsa") for i in top_ids)


def test_bm25_rejects_unrelated_query(docs):
    results = simple_search("Martian crypto-collateral Basel IV-Z weather Lagos", docs, min_score=1.5)
    assert results == [] or all(r["score"] < 3.0 for r in results)


def test_confidence_higher_for_strong_hit_than_empty():
    strong = [{"id": "1", "text": "KYC identity verification", "score": 6.0, "matched_terms": ["kyc", "identity"]}]
    empty = []
    c_strong = estimate_confidence(strong, query="What is KYC identity verification?")
    c_empty = estimate_confidence(empty, query="What is KYC identity verification?")
    assert c_strong.score > c_empty.score
    assert c_empty.score == 0.0
    assert should_abstain(c_empty, threshold=0.5)


def test_confidence_not_just_document_count():
    """Two weak docs should not outrank one strong, high-coverage hit."""
    weak_many = [
        {"id": "a", "text": "banks", "score": 0.6, "matched_terms": ["banks"]},
        {"id": "b", "text": "report", "score": 0.6, "matched_terms": ["report"]},
        {"id": "c", "text": "activity", "score": 0.6, "matched_terms": ["activity"]},
    ]
    strong_one = [
        {
            "id": "d",
            "text": "Banks must file a Suspicious Activity Report for suspicious activity.",
            "score": 7.5,
            "matched_terms": ["banks", "suspicious", "activity", "report"],
        }
    ]
    query = "When must banks file a Suspicious Activity Report?"
    c_weak = estimate_confidence(weak_many, query=query)
    c_strong = estimate_confidence(strong_one, query=query)
    assert c_strong.score > c_weak.score


def test_generation_stays_within_context():
    context = [
        {
            "id": "doc_kyc_01",
            "text": "Know Your Customer (KYC) refers to verifying a customer's identity in financial services.",
            "score": 5.0,
            "matched_terms": ["kyc", "identity"],
        }
    ]
    answer = generate_grounded("What is KYC?", context)
    assert "Know Your Customer" in answer.text
    assert "doc_kyc_01" in answer.citations
    # Should not invent regulators absent from context
    assert "Basel Committee" not in answer.text


def test_faithfulness_flags_unsupported_tokens():
    context = [{"id": "1", "text": "Banks monitor transactions for suspicious activity.", "score": 4.0}]
    grounded = "Banks monitor transactions for suspicious activity."
    hallucinated = "Banks monitor transactions for suspicious activity on Mars using quantum lasers."

    good = evaluate_faithfulness(grounded, context)
    bad = evaluate_faithfulness(hallucinated, context)
    assert good.score > bad.score
    assert bad.hallucination_risk in {"medium", "high"}
    assert "mars" in bad.unsupported_tokens or "quantum" in bad.unsupported_tokens


def test_pipeline_answers_kyc(pipeline):
    result = pipeline.run("What does KYC stand for and what is its purpose in financial services?")
    assert not result.abstained
    assert result.faithfulness is not None
    assert result.faithfulness.score >= 0.55
    assert "know your customer" in result.message.lower()


def test_pipeline_abstains_off_corpus(pipeline):
    result = pipeline.run("What will tomorrow's weather be in Lagos?")
    assert result.abstained
    assert "ABSTAIN" in result.message


def test_rag_harness_gate_passes():
    report = run_rag_harness(
        queries=load_rag_eval_queries(str(ROOT / "data" / "rag_eval_queries.json")),
        docs_path=DOCS_PATH,
    )
    assert report.total >= 8
    assert report.critical_failures == 0
    assert report.gate_passed
