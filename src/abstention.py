"""
abstention.py

Multi-signal confidence estimation and abstention gate.

Confidence is not "number of hits." It combines retrieval strength,
score margin, IDF-weighted query coverage, and evidence density.

Hard gates (learned from adversarial breaks):
  - missing rare / OOV query terms in evidence → abstain
  - IDF-weighted coverage below floor → abstain
Partial lexical overlap on common words (e.g. "banks") must not
clear the gate when distinctive terms (e.g. "Basel") are absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Union


ScoredLike = Union[Dict, object]
IdfFn = Callable[[str], float]


@dataclass(frozen=True)
class ConfidenceBreakdown:
    score: float
    retrieval_strength: float
    score_margin: float
    term_coverage: float
    idf_coverage: float
    evidence_density: float
    n_relevant: int
    unmatched_terms: List[str] = field(default_factory=list)
    missing_rare_terms: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    hard_abstain: bool = False

    def as_dict(self) -> Dict:
        return {
            "score": round(self.score, 4),
            "retrieval_strength": round(self.retrieval_strength, 4),
            "score_margin": round(self.score_margin, 4),
            "term_coverage": round(self.term_coverage, 4),
            "idf_coverage": round(self.idf_coverage, 4),
            "evidence_density": round(self.evidence_density, 4),
            "n_relevant": self.n_relevant,
            "unmatched_terms": list(self.unmatched_terms),
            "missing_rare_terms": list(self.missing_rare_terms),
            "reasons": list(self.reasons),
            "hard_abstain": self.hard_abstain,
        }


DEFAULT_WEIGHTS = {
    "retrieval_strength": 0.30,
    "score_margin": 0.15,
    "term_coverage": 0.20,
    "idf_coverage": 0.25,
    "evidence_density": 0.10,
}

# Floors learned from adversarial audit (see docs/AUDIT.md)
IDF_COVERAGE_FLOOR = 0.55
TERM_COVERAGE_FLOOR = 0.45
RARE_IDF_RATIO = 0.75  # relative to max IDF among query terms
MIN_QUERY_CONTENT_TERMS = 2  # single-token queries are underspecified


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


def _default_idf(term: str) -> float:
    # Unknown terms treated as maximally rare when no index is supplied
    return 3.0


def estimate_confidence(
    context: Sequence[ScoredLike],
    query: str = "",
    *,
    score_ceiling: float = 8.0,
    relevance_floor: float = 1.0,
    weights: Optional[Dict[str, float]] = None,
    idf_fn: Optional[IdfFn] = None,
    idf_coverage_floor: float = IDF_COVERAGE_FLOOR,
    term_coverage_floor: float = TERM_COVERAGE_FLOOR,
) -> ConfidenceBreakdown:
    """
    Estimate answerability confidence from ranked retrieval results.

    Hard abstain triggers when distinctive query mass is unsupported,
    even if common overlapping words produce a moderate BM25 score.
    """
    from src.retrieval import query_terms as extract_query_terms, tokenize

    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    idf = idf_fn or _default_idf

    empty = ConfidenceBreakdown(
        score=0.0,
        retrieval_strength=0.0,
        score_margin=0.0,
        term_coverage=0.0,
        idf_coverage=0.0,
        evidence_density=0.0,
        n_relevant=0,
        unmatched_terms=[],
        missing_rare_terms=[],
        reasons=["no_documents_above_relevance_floor"],
        hard_abstain=True,
    )

    relevant = [d for d in context if _doc_score(d) >= relevance_floor]
    reasons: List[str] = []

    if not relevant:
        return empty

    scores = sorted((_doc_score(d) for d in relevant), reverse=True)
    top = scores[0]
    second = scores[1] if len(scores) > 1 else 0.0

    retrieval_strength = min(1.0, top / max(score_ceiling, 1e-9))
    if retrieval_strength < 0.35:
        reasons.append("weak_top_hit")

    if top <= 0:
        score_margin = 0.0
    elif second <= 0:
        score_margin = 1.0
    else:
        score_margin = min(1.0, max(0.0, (top - second) / top))
    if score_margin < 0.15 and len(scores) > 1:
        reasons.append("ambiguous_top_candidates")

    q_terms = list(dict.fromkeys(extract_query_terms(query))) if query else []
    evidence_terms = set()
    for doc in relevant:
        matched = _matched_terms(doc)
        if matched:
            evidence_terms.update(matched)
        else:
            evidence_terms.update(tokenize(_doc_text(doc)))

    unmatched: List[str] = []
    missing_rare: List[str] = []

    # Underspecified queries ("banks") can fully "cover" themselves while saying nothing.
    if query and len(q_terms) < MIN_QUERY_CONTENT_TERMS:
        reasons.append("underspecified_query")
        return ConfidenceBreakdown(
            score=0.0,
            retrieval_strength=min(1.0, (scores[0] if scores else 0.0) / max(score_ceiling, 1e-9)),
            score_margin=0.0,
            term_coverage=0.0,
            idf_coverage=0.0,
            evidence_density=min(1.0, len(relevant) / 3.0),
            n_relevant=len(relevant),
            unmatched_terms=q_terms,
            missing_rare_terms=q_terms,
            reasons=reasons + ["hard_abstain_gate"],
            hard_abstain=True,
        )

    if q_terms:
        matched_q = [t for t in q_terms if t in evidence_terms]
        unmatched = [t for t in q_terms if t not in evidence_terms]
        term_coverage = len(matched_q) / len(q_terms)

        weights_q = [max(idf(t), 1e-6) for t in q_terms]
        covered_w = sum(max(idf(t), 1e-6) for t in matched_q)
        idf_coverage = covered_w / sum(weights_q)

        max_q_idf = max(weights_q)
        rare_cutoff = RARE_IDF_RATIO * max_q_idf
        missing_rare = [t for t in unmatched if idf(t) >= rare_cutoff]
    else:
        term_coverage = 1.0 if relevant else 0.0
        idf_coverage = term_coverage

    if term_coverage < term_coverage_floor:
        reasons.append("poor_query_term_coverage")
    if idf_coverage < idf_coverage_floor:
        reasons.append("poor_idf_weighted_coverage")
    if missing_rare:
        reasons.append(f"missing_rare_query_terms:{','.join(missing_rare)}")

    evidence_density = min(1.0, len(relevant) / 3.0)
    if evidence_density < 0.34:
        reasons.append("sparse_supporting_evidence")

    score = (
        w["retrieval_strength"] * retrieval_strength
        + w["score_margin"] * score_margin
        + w["term_coverage"] * term_coverage
        + w["idf_coverage"] * idf_coverage
        + w["evidence_density"] * evidence_density
    )
    score = max(0.0, min(1.0, score))

    hard_abstain = bool(missing_rare) or idf_coverage < idf_coverage_floor or term_coverage < term_coverage_floor
    if hard_abstain:
        # Force below typical abstention threshold; keep diagnostic score visible via reasons
        score = min(score, 0.49)
        reasons.append("hard_abstain_gate")

    if score >= 0.75 and not hard_abstain and not any(
        r.startswith("poor_") or r.startswith("missing_") or r.startswith("weak_") for r in reasons
    ):
        reasons.append("strong_grounded_evidence")
    elif not reasons:
        reasons.append("moderate_grounded_evidence")

    return ConfidenceBreakdown(
        score=score,
        retrieval_strength=retrieval_strength,
        score_margin=score_margin,
        term_coverage=term_coverage,
        idf_coverage=idf_coverage,
        evidence_density=evidence_density,
        n_relevant=len(relevant),
        unmatched_terms=unmatched,
        missing_rare_terms=missing_rare,
        reasons=reasons,
        hard_abstain=hard_abstain,
    )


def compute_confidence(context, query: str = "", **kwargs) -> float:
    """Backward-compatible scalar confidence API."""
    return estimate_confidence(context, query=query, **kwargs).score


def should_abstain(confidence, threshold: float = 0.5) -> bool:
    if hasattr(confidence, "hard_abstain") and confidence.hard_abstain:
        return True
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
