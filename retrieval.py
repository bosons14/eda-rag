"""Retrieval library: dense, BM25, hybrid (reciprocal rank fusion), reranking.

Modes:
  dense          embeddings only (Stage 1 baseline)
  bm25           keyword only
  hybrid         dense + bm25 fused with RRF (Stage 3)
  hybrid_rerank  hybrid candidates re-scored by a cross-encoder (Stage 3)
"""
import json
import pickle
import re
from functools import lru_cache

import numpy as np

from config import INDEX, EMBED_MODEL, QUERY_PREFIX, RERANK_MODEL

MODES = ["dense", "bm25", "hybrid", "hybrid_rerank"]
TOKEN = re.compile(r"-{0,2}[A-Za-z0-9_][A-Za-z0-9_.\-]*")


def tokenize(text):
    """Keep EDA identifiers intact (repair_design, -max_fanout, set_input_delay)
    and ALSO emit their parts, so 'max fanout' still matches '-max_fanout'."""
    out = []
    for t in TOKEN.findall(text.lower()):
        t = t.strip(".-") or t
        out.append(t)
        if "_" in t or "-" in t:
            out += [p for p in re.split(r"[_\-]+", t) if p]
    return out


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL, max_length=512)


class Retriever:
    def __init__(self, chunks="struct"):
        d = INDEX / chunks
        if not (d / "chunks.jsonl").exists():
            raise SystemExit(f"no index for '{chunks}' - run 03_index.py")
        self.dir = d
        self.chunks = [json.loads(l) for l in open(d / "chunks.jsonl")]
        with open(d / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        self._faiss = None

    # -- primitives: each returns [(chunk_idx, score)] best first ------------
    def _allowed(self, tools):
        if not tools:
            return None
        return np.array([c["tool"] in tools for c in self.chunks])

    def dense(self, q, k=10, tools=None):
        if self._faiss is None:
            import faiss
            if not (self.dir / "faiss.index").exists():
                raise SystemExit("no dense index - rerun 03_index.py without --no-dense")
            self._faiss = faiss.read_index(str(self.dir / "faiss.index"))
        qv = _embedder().encode([QUERY_PREFIX + q], normalize_embeddings=True).astype("float32")
        pool = k if not tools else min(len(self.chunks), k * 20)
        scores, idx = self._faiss.search(qv, pool)
        allowed = self._allowed(tools)
        res = [(int(i), float(s)) for i, s in zip(idx[0], scores[0])
               if i >= 0 and (allowed is None or allowed[i])]
        return res[:k]

    def bm25_search(self, q, k=10, tools=None):
        scores = self.bm25.get_scores(tokenize(q))
        allowed = self._allowed(tools)
        if allowed is not None:
            scores = np.where(allowed, scores, -np.inf)
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if np.isfinite(scores[i])]

    @staticmethod
    def rrf(*ranked_lists, k=60):
        """Reciprocal Rank Fusion: score = sum 1/(k + rank). Scale-free, so
        BM25 scores (0..30) and cosine scores (0..1) need no normalization."""
        fused = {}
        for lst in ranked_lists:
            for rank, (i, _) in enumerate(lst):
                fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank + 1)
        return sorted(fused.items(), key=lambda x: -x[1])

    def rerank(self, q, cands, k=10):
        if not cands:
            return []
        pairs = [(q, self.chunks[i]["text"]) for i, _ in cands]
        scores = _reranker().predict(pairs, batch_size=32)
        order = np.argsort(-np.asarray(scores))[:k]
        return [(cands[j][0], float(scores[j])) for j in order]

    # -- public --------------------------------------------------------------
    def search(self, q, mode="hybrid_rerank", k=5, tools=None, pool=50):
        if mode == "dense":
            hits = self.dense(q, k, tools)
        elif mode == "bm25":
            hits = self.bm25_search(q, k, tools)
        elif mode in ("hybrid", "hybrid_rerank"):
            fused = self.rrf(self.dense(q, pool, tools), self.bm25_search(q, pool, tools))
            hits = self.rerank(q, fused[:pool], k) if mode == "hybrid_rerank" else fused[:k]
        else:
            raise ValueError(f"mode must be one of {MODES}")
        return [dict(self.chunks[i], score=s) for i, s in hits]
