"""
evaluation.py

Faithfulness and hallucination-risk metrics for grounded answers.

Measures how much of the answer is supported by retrieved context using
content-word precision/recall/F1, unsupported token detection, and a
composite faithfulness score suitable for regression gates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Union

from src.retrieval import tokenize


ScoredLike = Union[Dict, object]

STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "once", "here", "there", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "don", "should", "now", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "of",
    "as", "it", "its", "this", "that", "these", "those", "i", "you",
    "he", "she", "we", "they", "them", "their", "our", "your", "based",
    "retrieved", "evidence", "sources", "answer", "context", "for",
}


ANSWER_BOILERPLATE = re.compile(
    r"^based on retrieved evidence for ['\"].*?['\"]:\s*",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class FaithfulnessResult:
    score: float
    precision: float
    recall: float
    f1: float
    unsupported_tokens: List[str]
    supported_token_count: int
    answer_content_tokens: int
    context_content_tokens: int
    hallucination_risk: str  # "low" | "medium" | "high"

    def as_dict(self) -> Dict:
        return {
            "score": round(self.score, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "unsupported_tokens": list(self.unsupported_tokens),
            "supported_token_count": self.supported_token_count,
            "answer_content_tokens": self.answer_content_tokens,
            "context_content_tokens": self.context_content_tokens,
            "hallucination_risk": self.hallucination_risk,
        }


def _get(doc: ScoredLike, key: str, default=None):
    if isinstance(doc, dict):
        return doc.get(key, default)
    return getattr(doc, key, default)


def _content_tokens(text: str) -> List[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def _strip_boilerplate(answer: str) -> str:
    cleaned = ANSWER_BOILERPLATE.sub("", answer.strip())
    # Drop trailing "Sources: ..." attribution line from faithfulness numerator
    lines = []
    for line in cleaned.splitlines():
        if line.strip().lower().startswith("sources:"):
            continue
        # Remove trailing [doc_id] citations from extractive bullets
        line = re.sub(r"\s*\[[^\]]+\]\s*$", "", line)
        line = re.sub(r"^\s*[-*]\s*", "", line)
        lines.append(line)
    return "\n".join(lines).strip()


def _context_text(context: Sequence[ScoredLike]) -> str:
    parts = []
    for doc in context:
        text = _get(doc, "text", "")
        if text:
            parts.append(str(text))
    return " ".join(parts)


def _risk(precision: float, unsupported_ratio: float) -> str:
    if precision >= 0.85 and unsupported_ratio <= 0.15:
        return "low"
    if precision >= 0.60 and unsupported_ratio <= 0.40:
        return "medium"
    return "high"


def evaluate_faithfulness(
    answer: str,
    context: Sequence[ScoredLike],
    *,
    max_unsupported_report: int = 12,
) -> FaithfulnessResult:
    """
    Compute faithfulness of an answer relative to retrieved context.

    Precision = fraction of answer content tokens supported by context.
    Recall    = fraction of context content tokens used in the answer.
    Score     = F1 with a precision bias (hallucination gate cares more
                about unsupported answer tokens than unused context).
    """
    if not context:
        return FaithfulnessResult(
            score=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            unsupported_tokens=[],
            supported_token_count=0,
            answer_content_tokens=0,
            context_content_tokens=0,
            hallucination_risk="high",
        )

    answer_body = _strip_boilerplate(answer)
    answer_tokens = _content_tokens(answer_body)
    context_tokens = _content_tokens(_context_text(context))
    context_set = set(context_tokens)

    if not answer_tokens:
        return FaithfulnessResult(
            score=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            unsupported_tokens=[],
            supported_token_count=0,
            answer_content_tokens=0,
            context_content_tokens=len(context_tokens),
            hallucination_risk="high",
        )

    supported = [t for t in answer_tokens if t in context_set]
    unsupported = [t for t in answer_tokens if t not in context_set]
    # Preserve order, unique unsupported for reporting
    seen = set()
    unsupported_unique: List[str] = []
    for t in unsupported:
        if t not in seen:
            seen.add(t)
            unsupported_unique.append(t)

    precision = len(supported) / len(answer_tokens)
    recall = (len(set(supported)) / len(set(context_tokens))) if context_tokens else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    # Precision-biased composite: unsupported answer content is the hallucination signal
    score = 0.7 * precision + 0.3 * f1
    unsupported_ratio = len(unsupported) / len(answer_tokens)

    return FaithfulnessResult(
        score=score,
        precision=precision,
        recall=recall,
        f1=f1,
        unsupported_tokens=unsupported_unique[:max_unsupported_report],
        supported_token_count=len(supported),
        answer_content_tokens=len(answer_tokens),
        context_content_tokens=len(context_tokens),
        hallucination_risk=_risk(precision, unsupported_ratio),
    )


def evaluate(answer: str, context: Sequence[ScoredLike]) -> float:
    """Backward-compatible scalar faithfulness API."""
    return round(evaluate_faithfulness(answer, context).score, 4)
