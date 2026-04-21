# Design Notes

Distilled from the planning chats. Authoritative spec: `Project.md`.

## Hybrid search — why
- BM25 for keyword precision.
- Vector ANN for semantic similarity.
- RRF (k=60) fuses ranked lists without score normalization.
- "Related" = kNN from the top-1 document's embedding.

## Crate choices
- **tantivy** — BM25, STORED + FAST fields.
- **arroy** — LMDB-mmap ANN, cosine, `need_build` + tree rebuild on commit.
- **moka** — TinyLFU, TTL + TTI — embedding cache + result cache.
- **fastembed** — optional ONNX, all-MiniLM-L6-v2, 384-dim.
- Considered and parked: frankensearch, usearch, LanceDB, Qdrant.

## Perf tips
- Quantized vectors (f16 / int8) when scaling.
- mmap everywhere (LMDB, tantivy).
- Two-tier: cheap first pass → heavier rerank on top-N.
- Warmup + per-mode warmup queries in the benchmark.

## Cache strategy
- **Embedding cache**: normalized query → vector. High hit-rate for repeat queries / RAG.
- **Result cache**: serialized SearchRequest → SearchResponse. 5m TTL / 2m idle.
- Invalidate result cache on commit.
- Future: semantic cache over recent queries; foyer for hot-disk tier.

## Scaling path (1M → 100M+)
- Now: single node, 4 in-process segments, rayon fan-out.
- Later: same `Segment` interface, each segment on its own container/VM, proxy stays thin.
- Route related-items via consistent hashing so the anchor's vector lives on the right node.

## Q&A decisions
- Docs = products: many varchar(255), 1–2 long text.
- Embeddings: pre-computed OK (ChatGPT or fastembed). Model = all-MiniLM-L6-v2 (384).
- Inserts: batch / message-queue friendly, minutes acceptable.
- Filters: yes, all of them; may drive future sharding.
- Field selection: optional per-query, never at the cost of speed.
- Single node for 1M; revisit segmentation at 100M.
