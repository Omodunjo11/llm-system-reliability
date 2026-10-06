"""
retrieval.py

Grounded keyword retrieval over a curated document corpus.

Uses Okapi BM25 ranking so relevance is graded by term frequency and
document length, not literal substring match. Documents below a score
floor are dropped so weak matches never inflate confidence.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# Function words that inflate BM25 on short corpora; kept out of query matching.
QUERY_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "where",
    "what", "which", "who", "whom", "whose", "why", "how", "is", "are", "was",
    "were", "be", "been", "being", "am", "do", "does", "did", "doing", "have",
    "has", "had", "having", "will", "would", "could", "should", "may", "might",
    "must", "shall", "can", "to", "of", "in", "on", "for", "with", "as", "by",
    "from", "at", "into", "about", "over", "after", "before", "between", "under",
    "again", "further", "once", "here", "there", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "also", "than", "that",
    "this", "these", "those", "it", "its", "their", "our", "your", "my", "me",
    "we", "you", "they", "them", "he", "she", "his", "her", "i",
}


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def query_terms(text: str) -> List[str]:
    """Tokenize a query and drop stopwords that create false BM25 hits."""
    return [t for t in tokenize(text) if t not in QUERY_STOPWORDS and len(t) > 1]


@dataclass(frozen=True)
class ScoredDocument:
    id: str
    text: str
    score: float
    matched_terms: List[str]
    metadata: Dict[str, str]

    def as_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "score": round(self.score, 4),
            "matched_terms": self.matched_terms,
            "metadata": self.metadata,
        }


class BM25Index:
    """Okapi BM25 over an in-memory document list."""

    def __init__(
        self,
        documents: Sequence[Dict],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.documents: List[Dict] = []
        self._doc_tokens: List[List[str]] = []
        self._doc_len: List[int] = []
        self._df: Dict[str, int] = {}
        self._avgdl = 0.0

        for raw in documents:
            doc = _normalize_doc(raw)
            tokens = tokenize(doc["text"])
            self.documents.append(doc)
            self._doc_tokens.append(tokens)
            self._doc_len.append(len(tokens))
            for term in set(tokens):
                self._df[term] = self._df.get(term, 0) + 1

        n = len(self.documents)
        self._avgdl = (sum(self._doc_len) / n) if n else 0.0
        self._n = n

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        # Standard BM25 IDF with +0.5 smoothing
        return math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 1.0,
    ) -> List[ScoredDocument]:
        terms = query_terms(query)
        if not terms or not self.documents:
            return []

        scored: List[ScoredDocument] = []
        for idx, doc in enumerate(self.documents):
            tokens = self._doc_tokens[idx]
            if not tokens:
                continue

            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            score = 0.0
            matched: List[str] = []
            dl = self._doc_len[idx]
            for term in terms:
                if term not in tf:
                    continue
                matched.append(term)
                idf = self._idf(term)
                freq = tf[term]
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / max(self._avgdl, 1e-9))
                score += idf * (freq * (self.k1 + 1.0)) / denom

            # Require at least one content-term match above the floor
            if score >= min_score and matched:
                scored.append(
                    ScoredDocument(
                        id=str(doc["id"]),
                        text=doc["text"],
                        score=score,
                        matched_terms=sorted(set(matched)),
                        metadata=dict(doc.get("metadata") or {}),
                    )
                )

        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[:top_k]


def _normalize_doc(raw: Dict) -> Dict:
    meta = raw.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {"note": str(meta)}
    return {
        "id": str(raw.get("id", "")),
        "text": str(raw.get("text", "")),
        "metadata": {str(k): str(v) for k, v in meta.items()},
    }


def load_docs(path: str = "data/sample_docs.json") -> List[Dict]:
    doc_path = Path(path)
    with doc_path.open() as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list of documents in {path}")
    return [_normalize_doc(doc) for doc in raw]


def build_index(docs: Optional[Sequence[Dict]] = None, path: str = "data/sample_docs.json") -> BM25Index:
    documents = list(docs) if docs is not None else load_docs(path)
    return BM25Index(documents)


def simple_search(
    query: str,
    docs: Sequence[Dict],
    top_k: int = 5,
    min_score: float = 1.0,
) -> List[Dict]:
    """
    Backward-compatible search API returning dicts with score metadata.
    Prefer BM25Index.search for typed results.
    """
    index = BM25Index(docs)
    return [doc.as_dict() for doc in index.search(query, top_k=top_k, min_score=min_score)]


def search(
    query: str,
    docs: Optional[Sequence[Dict]] = None,
    top_k: int = 5,
    min_score: float = 1.0,
    path: str = "data/sample_docs.json",
) -> List[ScoredDocument]:
    index = build_index(docs, path=path) if docs is None else BM25Index(docs)
    return index.search(query, top_k=top_k, min_score=min_score)
