# SuperFast Product Search Engine

**Goal**: Build a blazingly fast, hybrid search engine for **1 million product documents** that runs efficiently on a **Raspberry Pi 5 (8GB RAM)** and is significantly faster than Elasticsearch.

## Project Overview

- **Document type**: Products with many small `varchar(255)` fields + 1-2 longer text fields.
- **Capabilities**:
  - Full-text search (BM25)
  - Vector search (semantic)
  - Hybrid search (text + vector)
  - Related documents boost in scoring (top-1 exact match, positions 2-20 boosted by relation to top result)
  - Metadata filters (future sharding key)
- **Performance targets** (on Pi 5):
  - Hold 1M documents comfortably (<4GB RAM)
  - Indexing: a few minutes (batched, queue-based)
  - Queries: <10ms cold, <1ms hot (with cache)
  - Way faster than Elastic
- **Architecture**: Single-node with 4 internal segments for parallelism. Designed for easy future scaling.
- **Scoring**: RRF (Reciprocal Rank Fusion) + 0.3 related boost (cosine similarity to top result for ranks 2-20)

## Requirements

1. SUPER FAST — prioritize low latency and high throughput.
2. 1 million rows without issues on Pi 5 8GB.
3. Document-based: full-text, vector, hybrid, related scoring.
4. Selectable features per search (text-only, vector-only, hybrid, related, filters).
5. Run well on Raspberry Pi 5 (ARM64).
6. Inserts via slow message queue / batch (indexing time tolerant).
7. Return full docs or selected fields.
8. Start single-node; scalable design.

## Tech Stack

- **Language**: Rust 1.85+
- **Core Engine**: `frankensearch` (Tantivy BM25 + f16 SIMD vectors + RRF + two-tier)
- **Vector Index fallback**: `arroy` (LMDB mmap) if needed
- **Embeddings**: `fastembed-rs` (quantized local models, e.g., all-MiniLM-L6-v2 or smaller for Pi)
- **Cache**: `moka` (embedding cache + result cache)
- **Storage**: LMDB / mmap per segment (zero-copy, low RAM)
- **Parallelism**: `rayon` for segments
- **API**: Simple Axum HTTP server (or library mode)
- **Build**: Target `aarch64-unknown-linux-gnu`

Add to `Cargo.toml`:
```toml
[dependencies]
frankensearch = "*"          # or latest
moka = { version = "0.12", features = ["future"] }
fastembed = { version = "*", features = ["model-all-MiniLM-L6-v2"] }  # choose small quantized
rayon = "1"
serde = { version = "1.0", features = ["derive"] }
axum = { version = "0.7", features = ["json"] }
tokio = { version = "1", features = ["full"] }
heed = "*"                   # for arroy if used