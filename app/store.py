
"""Gemini embeddings + ChromaDB + BM25 hybrid retrieval."""

import os
import re
import threading
import uuid
from collections import OrderedDict

import chromadb
import numpy as np
from google import genai
from google.genai import types
from rank_bm25 import BM25Okapi

from . import config


def _tok(text: str):
    return re.findall(r"[a-z0-9'\-\.]+", text.lower())


class Store:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set GEMINI_API_KEY in your environment before starting the app."
            )

        self.embed_client = genai.Client(api_key=api_key)
        self.embed_model = config.EMBED_MODEL

        os.makedirs(config.DATA_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=os.path.join(config.DATA_DIR, "chroma")
        )

        # This collection uses Gemini embeddings, not MiniLM embeddings.
        self.col = self.client.get_or_create_collection(
            "it_docs_gemini",
            metadata={"hnsw:space": "cosine"},
        )

        self._bm25 = None
        self._bm25_ids = []
        self._lock = threading.Lock()
        self._cache = OrderedDict()
        self._qcache = OrderedDict()

    def _embed_batch(self, texts, task_type):
        if not texts:
            return []

        response = self.embed_client.models.embed_content(
            model=self.embed_model,
            contents=texts if len(texts) > 1 else texts[0],
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=768,
            ),
        )

        if not response.embeddings:
            raise RuntimeError("Gemini returned no embeddings.")

        vectors = []
        for item in response.embeddings:
            vector = np.asarray(item.values, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if norm == 0:
                raise RuntimeError("Gemini returned a zero-length embedding.")
            vectors.append((vector / norm).tolist())

        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Expected {len(texts)} embeddings, received {len(vectors)}."
            )

        return vectors

    def embed(self, texts):
        """Embed document chunks in small batches."""
        vectors = []
        batch_size = 16

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            vectors.extend(
                self._embed_batch(batch, "RETRIEVAL_DOCUMENT")
            )

        return vectors

    def embed_query(self, query: str):
        if query in self._qcache:
            self._qcache.move_to_end(query)
            return self._qcache[query]

        vector = self._embed_batch([query], "RETRIEVAL_QUERY")[0]
        self._qcache[query] = vector

        if len(self._qcache) > 64:
            self._qcache.popitem(last=False)

        return vector

    def add(self, source: str, chunks, meta: dict) -> int:
        self.delete_source(source)

        total = 0
        batch_size = 16

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            docs = [chunk[1] for chunk in batch]
            embeddings = self.embed(docs)
            ids = [str(uuid.uuid4()) for _ in batch]

            metadatas = [
                {
                    **meta,
                    "source": source,
                    "section": chunk[0],
                    "chunk_index": start + i,
                }
                for i, chunk in enumerate(batch)
            ]

            self.col.add(
                ids=ids,
                documents=docs,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            total += len(ids)

        self._invalidate()
        return total

    def delete_source(self, source: str):
        if self.col.count() > 0:
            self.col.delete(where={"source": source})
        self._invalidate()

    def _invalidate(self):
        with self._lock:
            self._bm25 = None
            self._bm25_ids = []
            self._cache.clear()

    def stats(self):
        return {"chunks": self.col.count()}

    @staticmethod
    def _where(filters: dict | None):
        items = [
            {key: value}
            for key, value in (filters or {}).items()
            if value is not None and value != ""
        ]
        if not items:
            return None
        return items[0] if len(items) == 1 else {"$and": items}

    def _ensure_bm25(self):
        with self._lock:
            if self._bm25 is None:
                got = self.col.get(include=["documents"])
                self._bm25_ids = got["ids"]
                documents = got["documents"] or []

                self._bm25 = (
                    BM25Okapi([_tok(doc) for doc in documents])
                    if documents else None
                )

    def search(
        self,
        query: str,
        top_k=None,
        threshold=None,
        filters=None,
        hybrid=True,
    ):
        top_k = top_k or config.TOP_K
        threshold = (
            config.SIM_THRESHOLD if threshold is None else threshold
        )

        key = (
            query.strip().lower(),
            str(sorted((filters or {}).items())),
            top_k,
            threshold,
            hybrid,
        )

        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        if self.col.count() == 0:
            return []

        where = self._where(filters)
        query_vector = self.embed_query(query)
        count = self.col.count()
        n_results = min(top_k * 4, count)

        dense = self.col.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            where=where,
        )

        # Reciprocal Rank Fusion: dense retrieval + BM25.
        fusion_constant = 60
        fused = {}

        for rank, chunk_id in enumerate(dense["ids"][0]):
            fused[chunk_id] = (
                fused.get(chunk_id, 0) + 1 / (fusion_constant + rank)
            )

        if hybrid:
            self._ensure_bm25()

            if self._bm25 is not None:
                scores = self._bm25.get_scores(_tok(query))
                order = np.argsort(scores)[::-1][:top_k * 4]

                allowed = (
                    set(self.col.get(where=where, include=[])["ids"])
                    if where else None
                )

                rank = 0
                for idx in order:
                    if scores[idx] <= 0:
                        break

                    chunk_id = self._bm25_ids[idx]
                    if allowed is not None and chunk_id not in allowed:
                        continue

                    fused[chunk_id] = (
                        fused.get(chunk_id, 0)
                        + 1 / (fusion_constant + rank)
                    )
                    rank += 1

        candidates = sorted(
            fused, key=fused.get, reverse=True
        )[:top_k * 2]

        if not candidates:
            return []

        got = self.col.get(
            ids=candidates,
            include=["documents", "metadatas", "embeddings"],
        )

        by_id = {
            chunk_id: (doc, metadata, np.asarray(embedding))
            for chunk_id, doc, metadata, embedding in zip(
                got["ids"],
                got["documents"],
                got["metadatas"],
                got["embeddings"],
            )
        }

        q = np.asarray(query_vector)
        results = []

        for chunk_id in candidates:
            doc, metadata, embedding = by_id[chunk_id]
            similarity = float(np.dot(q, embedding))

            if similarity < threshold:
                continue

            results.append({
                "id": chunk_id,
                "text": doc,
                "metadata": metadata,
                "similarity": round(similarity, 4),
                "fused_score": round(fused[chunk_id], 5),
            })

            if len(results) >= top_k:
                break

        self._cache[key] = results
        if len(self._cache) > 64:
            self._cache.popitem(last=False)

        return results
