```
Client (HTTP)
    │
    ▼
TopProxy (axum)
  ├─ moka result cache (10k, 5m TTL, 2m idle)
  ├─ moka embedding cache (50k, 1h TTL) [optional]
  └─ rayon fan-out
        │
    ┌───┼─────┬─────┐
    ▼   ▼     ▼     ▼
  seg-0 seg-1 seg-2 seg-3
    ├─ tantivy (BM25, STORED)
    └─ arroy  (Cosine ANN, LMDB mmap)
        │
        ▼
  per-segment top-k  (over-fetch k·3)
        │
        ▼
  merge + RRF fuse (k=60)
        │
        ▼
  related-boost: anchor = top-1, cosine for ranks 2..20
        │
        ▼
  fetch stored docs → SearchHit[]
```
