"""
abstention.py

Multi-signal confidence estimation and abstention gate.

Confidence is not "number of hits." It combines retrieval strength,
score margin, query-term coverage, and evidence density so the system
refuses when evidence is weak, ambiguous, or off-topic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union


ScoredLike = Union[Dict, object]


@dataclass(frozen=True)
class ConfidenceBreakdown:
    score: float
    retrieval_strength: float
    score_margin: float
    term_coverage: float
    evidence_density: float
    n_relevant: int
    reasons: List[str]

    def as_dict(self) -> Dict:
        return {
            "score": round(self.score, 4),
            "retrieval_strength": round(self.retrieval_strength, 4),
            "score_margin": round(self.score_margin, 4),
            "term_coverage": round(self.term_coverage, 4),
            "evidence_density": round(self.evidence_density, 4),
            "n_relevant": self.n_relevant,
            "reasons": list(self.reasons),
        }


DEFAULT_WEIGHTS = {
    "retrieval_strength": 0.40,
    "score_margin": 0.20,
    "term_coverage": 0.25,
    "evidence_density": 0.15,
}


def _get(doc: ScoredLike, key: str, default=None):
    if isinstance(doc, dict):
        return doc.get(key, default)
    return getattr(doc, key, default)


def _doc_score(doc: ScoredLike) -> float:
    value = _get(doc, "score", None)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _matched_terms(doc: ScoredLike) -> List[str]:
    terms = _get(doc, "matched_terms", None) or []
    return [str(t).lower() for t in terms]


def _doc_text(doc: ScoredLike) -> str:
    return str(_get(doc, "text", "") or "")


def estimate_confidence(
    context: Sequence[ScoredLike],
    query: str = "",
    *,
    score_ceiling: float = 8.0,
    relevance_floor: float = 1.0,
    weights: Optional[Dict[str, float]] = None,
) -> ConfidenceBreakdown:
    """
    Estimate answerability confidence from ranked retrieval results.

    Signals
    -------
    retrieval_strength : top BM25 score normalized by score_ceiling
    score_margin       : separation between top-1 and top-2 (ambiguity penalty)
    term_coverage      : fraction of query terms appearing in retrieved docs
    evidence_density   : how many docs clear the relevance floor (capped)
    """
    from src.retrieval import query_terms as extract_query_terms, tokenize

    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    relevant = [d for d in context if _doc_score(d) >= relevance_floor]
    reasons: List[str] = []

    if not relevant:
        return ConfidenceBreakdown(
            score=0.0,
            retrieval_strength=0.0,
            score_margin=0.0,
            term_coverage=0.0,
            evidence_density=0.0,
            n_relevant=0,
            reasons=["no_documents_above_relevance_floor"],
        )

    scores = sorted((_doc_score(d) for d in relevant), reverse=True)
    top = scores[0]
    second = scores[1] if len(scores) > 1 else 0.0

    retrieval_strength = min(1.0, top / max(score_ceiling, 1e-9))
    if retrieval_strength < 0.35:
        reasons.append("weak_top_hit")

    # Large margin => clear best doc; tiny margin => ambiguous / contested evidence
    if top <= 0:
        score_margin = 0.0
    elif second <= 0:
        score_margin = 1.0
    else:
        score_margin = min(1.0, max(0.0, (top - second) / top))
    if score_margin < 0.15 and len(scores) > 1:
        reasons.append("ambiguous_top_candidates")

    q_terms = set(extract_query_terms(query)) if query else set()
    if q_terms:
        corpus_terms = set()
        for doc in relevant:
            matched = _matched_terms(doc)
            if matched:
                corpus_terms.update(matched)
            else:
                corpus_terms.update(tokenize(_doc_text(doc)))
        term_coverage = len(q_terms & corpus_terms) / len(q_terms)
    else:
        term_coverage = 1.0 if relevant else 0.0
    if term_coverage < 0.5:
        reasons.append("poor_query_term_coverage")

    # More independent supporting docs increase confidence, with diminishing returns
    evidence_density = min(1.0, len(relevant) / 3.0)
    if evidence_density < 0.34:
        reasons.append("sparse_supporting_evidence")

    score = (
        w["retrieval_strength"] * retrieval_strength
        + w["score_margin"] * score_margin
        + w["term_coverage"] * term_coverage
        + w["evidence_density"] * evidence_density
    )
    score = max(0.0, min(1.0, score))

    if score >= 0.75 and not reasons:
        reasons.append("strong_grounded_evidence")
    elif not reasons:
        reasons.append("moderate_grounded_evidence")

    return ConfidenceBreakdown(
        score=score,
        retrieval_strength=retrieval_strength,
        score_margin=score_margin,
        term_coverage=term_coverage,
        evidence_density=evidence_density,
        n_relevant=len(relevant),
        reasons=reasons,
    )


def compute_confidence(context, query: str = "", **kwargs) -> float:
    """Backward-compatible scalar confidence API."""
    return estimate_confidence(context, query=query, **kwargs).score


def should_abstain(confidence, threshold: float = 0.5) -> bool:
    if hasattr(confidence, "score"):
        value = float(confidence.score)
    else:
        value = float(confidence)
    return value < threshold


def abstention_message(breakdown: Optional[ConfidenceBreakdown] = None) -> str:
    base = (
        "ABSTAIN: Insufficient reliable evidence to answer. "
        "Refusing rather than guessing."
    )
    if breakdown is None or not breakdown.reasons:
        return base
    detail = "; ".join(breakdown.reasons)
    return f"{base} Signals: {detail}."
