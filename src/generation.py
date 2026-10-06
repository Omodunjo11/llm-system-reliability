"""
generation.py

Grounded response generation from retrieved documents only.

No model weights and no free-form invention: answers are extractive
syntheses of retrieved sentences, attributed to source document IDs.
If nothing relevant can be extracted, the caller should have abstained.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

from src.retrieval import query_terms as extract_query_terms, tokenize


ScoredLike = Union[Dict, object]
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class GroundedAnswer:
    text: str
    citations: List[str]
    supporting_sentences: List[Dict[str, str]]
    abstained: bool = False

    def as_dict(self) -> Dict:
        return {
            "text": self.text,
            "citations": list(self.citations),
            "supporting_sentences": list(self.supporting_sentences),
            "abstained": self.abstained,
        }


def _get(doc: ScoredLike, key: str, default=None):
    if isinstance(doc, dict):
        return doc.get(key, default)
    return getattr(doc, key, default)


def _split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in SENTENCE_SPLIT.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _sentence_relevance(sentence: str, q_terms: Sequence[str]) -> float:
    if not q_terms:
        return 0.0
    sent_terms = set(tokenize(sentence))
    if not sent_terms:
        return 0.0
    q_set = set(q_terms)
    overlap_terms = q_set & sent_terms
    if not overlap_terms:
        return 0.0
    overlap = len(overlap_terms)
    density = overlap / max(len(sent_terms), 1)
    coverage = overlap / max(len(q_set), 1)
    # Boost short acronym / distinctive content hits (e.g. kyc, sar, gdpr)
    rare_boost = sum(0.2 for t in overlap_terms if len(t) <= 4)
    return min(1.5, 0.60 * coverage + 0.25 * density + rare_boost)


def generate_grounded(
    query: str,
    context: Sequence[ScoredLike],
    *,
    max_sentences: int = 3,
    min_sentence_score: float = 0.12,
) -> GroundedAnswer:
    """
    Build an answer strictly from retrieved document sentences.

    Sentences are ranked by content-term overlap; only those clearing
    min_sentence_score are kept. Output includes citation IDs.
    """
    if not context:
        return GroundedAnswer(
            text="No relevant information found.",
            citations=[],
            supporting_sentences=[],
            abstained=True,
        )

    q_terms = extract_query_terms(query)
    candidates: List[Dict] = []

    # Only synthesize from strong retrieval neighbors (relative to top hit)
    top_score = max(float(_get(d, "score", 0.0) or 0.0) for d in context)
    score_floor = max(1.0, 0.45 * top_score)

    for doc in context:
        doc_score = float(_get(doc, "score", 0.0) or 0.0)
        if doc_score < score_floor:
            continue
        doc_id = str(_get(doc, "id", "unknown"))
        text = str(_get(doc, "text", "") or "")
        for sentence in _split_sentences(text):
            score = _sentence_relevance(sentence, q_terms)
            score = score + 0.02 * min(doc_score, 10.0) / 10.0
            if score >= min_sentence_score:
                candidates.append(
                    {
                        "doc_id": doc_id,
                        "sentence": sentence,
                        "score": score,
                    }
                )

    candidates.sort(key=lambda c: c["score"], reverse=True)

    selected: List[Dict] = []
    seen_text = set()
    for cand in candidates:
        key = cand["sentence"].lower()
        if key in seen_text:
            continue
        seen_text.add(key)
        selected.append(cand)
        if len(selected) >= max_sentences:
            break

    if not selected:
        # Fall back to top document text rather than inventing content
        top = context[0]
        top_id = str(_get(top, "id", "unknown"))
        top_text = str(_get(top, "text", "") or "").strip()
        selected = [{"doc_id": top_id, "sentence": top_text, "score": 0.0}]

    citations = []
    for item in selected:
        if item["doc_id"] not in citations:
            citations.append(item["doc_id"])

    lines = [f"- {item['sentence']} [{item['doc_id']}]" for item in selected]
    answer = (
        f"Based on retrieved evidence for '{query}':\n"
        + "\n".join(lines)
        + f"\n\nSources: {', '.join(citations)}"
    )

    supporting = [
        {"doc_id": item["doc_id"], "sentence": item["sentence"]}
        for item in selected
    ]

    return GroundedAnswer(
        text=answer,
        citations=citations,
        supporting_sentences=supporting,
        abstained=False,
    )


def generate(query: str, context: Sequence[ScoredLike]) -> str:
    """Backward-compatible string API."""
    return generate_grounded(query, context).text
