product-search/
├── Cargo.toml
├── src/
│   ├── main.rs                 # entry point + Axum server
│   ├── proxy.rs                # TopProxy with cache + feature selector
│   ├── segment.rs              # Segment wrapper (frankensearch)
│   ├── embedding.rs            # fastembed cache
│   ├── scoring.rs              # RRF + related boost
│   ├── schema.rs               # Product document schema
│   ├── indexer.rs              # Batch indexing (queue friendly)
│   └── types.rs                # Request/Response structs
├── data/                       # index dirs (seg-0, seg-1, ...)
├── README.md
└── .gitignore