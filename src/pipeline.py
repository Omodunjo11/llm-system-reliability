"""
pipeline.py

End-to-end grounded RAG reliability pipeline:

  query → BM25 retrieval → multi-signal confidence → abstain | generate → faithfulness
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from src.abstention import (
    ConfidenceBreakdown,
    abstention_message,
    estimate_confidence,
    should_abstain,
)
from src.evaluation import FaithfulnessResult, evaluate_faithfulness
from src.generation import GroundedAnswer, generate_grounded
from src.retrieval import BM25Index, ScoredDocument, build_index, load_docs


@dataclass
class PipelineResult:
    query: str
    retrieved: List[ScoredDocument]
    confidence: ConfidenceBreakdown
    abstained: bool
    answer: Optional[GroundedAnswer]
    faithfulness: Optional[FaithfulnessResult]
    message: str

    def as_dict(self) -> Dict:
        return {
            "query": self.query,
            "retrieved": [d.as_dict() for d in self.retrieved],
            "confidence": self.confidence.as_dict(),
            "abstained": self.abstained,
            "answer": self.answer.as_dict() if self.answer else None,
            "faithfulness": self.faithfulness.as_dict() if self.faithfulness else None,
            "message": self.message,
        }


@dataclass
class ReliabilityPipeline:
    """Configurable four-stage reliability pipeline over a document corpus."""

    index: BM25Index
    abstain_threshold: float = 0.5
    top_k: int = 5
    min_retrieval_score: float = 1.0
    faithfulness_floor: float = 0.55

    @classmethod
    def from_docs(
        cls,
        docs: Optional[Sequence[Dict]] = None,
        path: str = "data/sample_docs.json",
        **kwargs,
    ) -> "ReliabilityPipeline":
        documents = list(docs) if docs is not None else load_docs(path)
        return cls(index=BM25Index(documents), **kwargs)

    def run(self, query: str) -> PipelineResult:
        retrieved = self.index.search(
            query,
            top_k=self.top_k,
            min_score=self.min_retrieval_score,
        )
        confidence = estimate_confidence(retrieved, query=query)
        abstain = should_abstain(confidence, threshold=self.abstain_threshold)

        if abstain:
            msg = abstention_message(confidence)
            return PipelineResult(
                query=query,
                retrieved=retrieved,
                confidence=confidence,
                abstained=True,
                answer=None,
                faithfulness=None,
                message=msg,
            )

        answer = generate_grounded(query, retrieved)
        faithfulness = evaluate_faithfulness(answer.text, retrieved)

        # Post-generation safety: abstain if faithfulness collapses
        if faithfulness.score < self.faithfulness_floor or faithfulness.hallucination_risk == "high":
            msg = (
                "ABSTAIN: Generated answer failed faithfulness gate "
                f"(score={faithfulness.score:.2f}, risk={faithfulness.hallucination_risk})."
            )
            return PipelineResult(
                query=query,
                retrieved=retrieved,
                confidence=confidence,
                abstained=True,
                answer=answer,
                faithfulness=faithfulness,
                message=msg,
            )

        return PipelineResult(
            query=query,
            retrieved=retrieved,
            confidence=confidence,
            abstained=False,
            answer=answer,
            faithfulness=faithfulness,
            message=answer.text,
        )


def run_query(query: str, path: str = "data/sample_docs.json", **kwargs) -> PipelineResult:
    pipeline = ReliabilityPipeline.from_docs(path=path, **kwargs)
    return pipeline.run(query)
