# Design Note: IT Troubleshooting Voice Assistant

## 1. Goal and constraints
Spoken troubleshooting needs answers that start fast (<~1.5 s to first audio/text), are short, and are verifiably grounded. Every decision below trades off those three.

## 2. Ingestion and chunking
- Parsers: pypdf, python-docx (Word heading styles become `##`), plain TXT/MD.
- **Heading-aware chunking**: split on headings, then pack paragraphs to ~900 chars with one-paragraph overlap. Troubleshooting docs are organised as problem → steps, so a heading is the natural unit; fixed-size windows would split a numbered procedure in half. Each chunk is prefixed with its heading so "Flush DNS" embeds with its parent context.
- 900 chars (~200 tokens) keeps 5 chunks well under ~1.5k prompt tokens, which directly lowers LLM time-to-first-token.

## 3. Embeddings and vector DB
`all-MiniLM-L6-v2` (384-d, CPU-friendly, ~10-20 ms/query) with L2-normalised vectors so cosine = dot product. ChromaDB persistent HNSW (cosine) gives metadata pre-filtering without extra infrastructure. Trade-off: MiniLM is weaker than larger models on long-tail jargon; the BM25 leg compensates for exact tokens such as error codes, commands and product names.

## 4. Retrieval
1. Dense top-4K with metadata `where` filter.
2. BM25 over all chunks (same filter via allowed-id set).
3. **Reciprocal Rank Fusion** (k=60): needs no score calibration between the two systems.
4. Fused candidates are re-scored with true cosine similarity; those below the **threshold (0.30)** are dropped, then Top-K (5). Applying the threshold on cosine rather than the RRF score keeps it interpretable and lets the system answer "I don't have documentation for that" instead of hallucinating.
5. Query rewriting (Gemini Flash-Lite) runs only when there is conversation history, so first turns pay no extra latency.

## 5. Grounded generation
Numbered context passages are injected with source/section labels; the system prompt forbids answering outside them, requires inline `[n]` citations, and asks for short numbered steps suitable for speech. The `sources` SSE event is sent *before* the first token so the UI shows citations immediately.

## 6. Streaming retrieval for partial transcripts
Speech recognition emits interim hypotheses every few hundred ms. Strategy:
- **Debounced prefetch** (400 ms pause, ≥4 words): the client calls `/retrieve` with the interim text. This loads/caches the query embedding and the search result.
- **Final transcript → `/query`**: interim and final text are usually near-identical, and the embedding/result caches (LRU, keyed by normalised query + filters) cut retrieval to ~0 ms on exact matches. When the final text differs, retrieval is simply re-run; it is only tens of ms, so prefetch is purely an optimisation and never a correctness risk.
- Interim text is never sent to the LLM (cost, and unstable text would produce flip-flopping answers).
- Possible extension: stabilise on the longest common prefix of consecutive hypotheses and only re-query when it changes materially.

## 7. Latency budget (typical, warm)
| Stage | Target |
|---|---|
| Embed query (cached/prefetched) | 0-20 ms |
| Hybrid search (≤10k chunks) | 10-40 ms |
| Rewrite (follow-ups only, Flash-Lite) | 200-400 ms |
| LLM time to first token (Gemini Flash, ~1.2k prompt tokens) | 400-900 ms |
Other optimisations: embedding model loaded at startup; BM25 index built lazily and cached; SSE streaming with proxy buffering disabled; compact prompts; thinking disabled for Flash models (thinking_budget=0), bounded output (600 tokens, ~150-word style); `done` event returns per-stage timings for monitoring.

## 8. Evaluation plan (not yet automated)
Build a 30-50 question set mapped to expected source sections; track recall@5, MRR, fraction of out-of-scope questions correctly refused, and p50/p95 time-to-first-token. Ablate dense-only vs hybrid and thresholds 0.2-0.4.

## 9. Known limitations
Single-process memory (sessions, BM25, caches); no OCR; no cross-encoder reranker; browser-only STT/TTS (production would use streaming STT such as Deepgram/Whisper over WebSocket); prompt-injection from uploaded docs is mitigated only by the prompt's grounding rules.
