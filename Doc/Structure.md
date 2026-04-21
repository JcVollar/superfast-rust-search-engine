```
product-search/
├── Cargo.toml
├── src/
│   ├── main.rs       # CLI + Axum server
│   ├── proxy.rs      # TopProxy: fan-out, cache, fusion
│   ├── segment.rs    # Tantivy + arroy per segment
│   ├── schema.rs     # Product schema + doc mapping
│   ├── scoring.rs    # RRF + related boost
│   ├── embedding.rs  # moka-cached fastembed wrapper
│   ├── indexer.rs    # Batch + JSONL + synthetic
│   ├── benchmark.rs  # Latency / QPS harness
│   └── types.rs      # Request/Response DTOs
├── data/             # seg-0 … seg-N (tantivy/ + arroy/)
└── scripts/          # csv_to_jsonl.py, embed_products.py
```
