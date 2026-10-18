"""Vector store (Chroma) + BM25 hybrid retrieval with metadata filtering."""
import re
import threading
import uuid
from collections import OrderedDict

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from . import config


def _tok(s: str):
    return re.findall(r"[a-z0-9\-\.]+", s.lower())


class Store:
    def __init__(self):
        self.embedder = SentenceTransformer(config.EMBED_MODEL)
        self.client = chromadb.PersistentClient(path=f"{config.DATA_DIR}/chroma")
        self.col = self.client.get_or_create_collection("it_docs", metadata={"hnsw:space": "cosine"})
        self._bm25, self._bm25_ids, self._lock = None, [], threading.Lock()
        self._cache: OrderedDict = OrderedDict()   # retrieval results (helps partial-transcript prefetch)
        self._qcache: OrderedDict = OrderedDict()  # query -> embedding

    # ---------- embedding ----------
    def embed(self, texts):
        return self.embedder.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    def embed_query(self, q: str):
        if q in self._qcache:
            return self._qcache[q]
        v = self.embed([q])[0]
        self._qcache[q] = v
        if len(self._qcache) > 256:
            self._qcache.popitem(last=False)
        return v

    # ---------- ingestion ----------
    def add(self, source: str, chunks, meta: dict) -> int:
        self.delete_source(source)  # re-upload replaces old version
        ids = [str(uuid.uuid4()) for _ in chunks]
        docs = [c[1] for c in chunks]
        metas = [{**meta, "source": source, "section": c[0], "chunk_index": i} for i, c in enumerate(chunks)]
        self.col.add(ids=ids, documents=docs, embeddings=self.embed(docs), metadatas=metas)
        self._invalidate()
        return len(ids)

    def delete_source(self, source: str):
        self.col.delete(where={"source": source})
        self._invalidate()

    def _invalidate(self):
        self._bm25 = None
        self._cache.clear()

    def stats(self):
        return {"chunks": self.col.count()}

    # ---------- retrieval ----------
    @staticmethod
    def _where(filters: dict | None):
        items = [{k: v} for k, v in (filters or {}).items() if v]
        if not items:
            return None
        return items[0] if len(items) == 1 else {"$and": items}

    def _ensure_bm25(self):
        with self._lock:
            if self._bm25 is None:
                got = self.col.get(include=["documents"])
                self._bm25_ids = got["ids"]
                self._bm25 = BM25Okapi([_tok(d) for d in got["documents"]]) if got["ids"] else None

    def search(self, query: str, top_k=None, threshold=None, filters=None, hybrid=True):
        top_k = top_k or config.TOP_K
        threshold = config.SIM_THRESHOLD if threshold is None else threshold
        key = (query.strip().lower(), str(sorted((filters or {}).items())), top_k, threshold, hybrid)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if self.col.count() == 0:
            return []

        where = self._where(filters)
        qv = self.embed_query(query)
        n = min(top_k * 4, self.col.count())
        dense = self.col.query(query_embeddings=[qv], n_results=n, where=where)

        # Reciprocal Rank Fusion of dense + BM25 rankings
        K = 60
        fused = {}
        for r, i in enumerate(dense["ids"][0]):
            fused[i] = fused.get(i, 0) + 1 / (K + r)
        if hybrid:
            self._ensure_bm25()
            if self._bm25:
                scores = self._bm25.get_scores(_tok(query))
                order = np.argsort(scores)[::-1][: top_k * 4]
                allowed = set(self.col.get(where=where, include=[])["ids"]) if where else None
                rank = 0
                for idx in order:
                    if scores[idx] <= 0:
                        break
                    cid = self._bm25_ids[idx]
                    if allowed is not None and cid not in allowed:
                        continue
                    fused[cid] = fused.get(cid, 0) + 1 / (K + rank)
                    rank += 1

        cand = sorted(fused, key=fused.get, reverse=True)[: top_k * 2]
        got = self.col.get(ids=cand, include=["documents", "metadatas", "embeddings"])
        by_id = {i: (d, m, np.array(e)) for i, d, m, e in zip(got["ids"], got["documents"], got["metadatas"], got["embeddings"])}
        qa = np.array(qv)
        results = []
        for cid in cand:  # keep fused order
            d, m, e = by_id[cid]
            sim = float(np.dot(qa, e))  # cosine (both L2-normalized)
            if sim < threshold:
                continue
            results.append({"id": cid, "text": d, "metadata": m, "similarity": round(sim, 4), "fused_score": round(fused[cid], 5)})
            if len(results) == top_k:
                break

        self._cache[key] = results
        if len(self._cache) > 512:
            self._cache.popitem(last=False)
        return results
