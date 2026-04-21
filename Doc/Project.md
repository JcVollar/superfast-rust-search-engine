# SuperFast Product Search

Hybrid search over 1M product docs on a Pi 5 (8GB). Goal: faster than Elasticsearch.

## Capabilities
BM25 · vector ANN · hybrid (RRF) · related-boost (top-1 anchor, cosine to anchor for ranks 2–20) · metadata filters.

## Targets (Pi 5)
- <4GB RAM for 1M docs
- Indexing: minutes (batched)
- Query: <10ms cold, <1ms hot

## Stack
Rust 1.85+ · tantivy (BM25) · arroy (LMDB ANN) · moka (cache) · fastembed (optional ONNX) · rayon · axum · clap. Target `aarch64-unknown-linux-gnu`.

## Architecture
4 internal segments, rayon fan-out, RRF fusion, result cache at proxy. Shard key = `hash(product_id) % N`.

## Scoring
```
rank 1:      final = rrf
rank 2..20:  final = 0.7·rrf + 0.3·cos(doc, top1)
rank 21+:    final = rrf
```
