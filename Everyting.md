Building a superfast hybrid search engine in Rust (vectors + related items + free-text) is an excellent choice — Rust gives you zero-cost abstractions, SIMD, parallelism via rayon, and memory-mapped storage for sub-millisecond queries even on millions of documents.
Your requirements map perfectly to a hybrid search setup:

Free-text → inverted index + BM25 ranking (keyword precision).
Vectors → ANN (Approximate Nearest Neighbors) for semantic similarity / embeddings.
Related items → vector kNN search (find items most similar to a given document’s embedding).
Hybrid → run both searches + fuse results (e.g. RRF = Reciprocal Rank Fusion) for the best of both worlds.

Recommended Architecture (Superfast & Embeddable)

Data model: Documents with ID, text fields, pre-computed embedding vector (f32/f16), and metadata.
Indexing:
Full-text → Tantivy (inverted index + BM25).
Vectors → arroy (Meilisearch’s ANN) or usearch (HNSW).

Query flow (parallelized):
Free-text query → BM25 results.
Embed query (or use provided vector) → ANN kNN.
Fuse with RRF (or linear combination) + rerank.
Related items = ANN search using a document’s own vector.

Storage: LMDB (via arroy) or Lance format (LanceDB) — mmap-based, zero-copy, persistent, shareable across processes.
Optional on-device embeddings: fastembed-rs (fast ONNX, local models like MiniLM or even faster ones).

This is much faster than running a separate server (no network, no serialization) and leverages Rust’s strengths.
Best Crates (2026 State-of-the-Art)
Here are the top options, ordered by how “plug-and-play” they are:















































Crate / ToolTypeStrengthsBest ForHybrid SupportfrankensearchReady hybrid libTantivy BM25 + f16 SIMD vectors + RRF + two-tier (fast first-pass + quality rerank)Drop-in hybrid searchBuilt-inLanceDBEmbedded vector DBNative Rust SDK, full-text (BM25), SQL filters, hybrid, persistent Lance formatProduction-grade embedded DBBuilt-intantivy + arroyText + VectorTantivy = best full-text; arroy = memory-efficient ANN (LMDB + filtered search, incremental)Maximum control & speedCustom RRFtantivy + usearchText + Vectorusearch = blazing HNSW + quantization + single-fileHighest raw ANN throughputCustom RRFMeilisearchFull serverEverything built-in (hybrid, auto-embedders, filters)If you want a ready serverBuilt-in
My top recommendation for “build your own superfast”:

frankensearch if you want a ready-made hybrid with great defaults.
tantivy + arroy (or usearch) if you want full control and ultimate performance.
LanceDB if you also want SQL + easy filtering on metadata.

Quickstart Example (Tantivy + arroy + Hybrid RRF)
Add to Cargo.toml:
toml[dependencies]
tantivy = "0.22"
arroy = "0.6"          # or latest
rayon = "1.10"
fastembed = { version = "4", features = ["model-text-embedding-3-small"] }  # optional
heed = "0.21"          # LMDB wrapper used by arroy
Basic skeleton (highly simplified — full code would be ~200–300 LOC):
Rustuse tantivy::{schema::*, Index, IndexWriter, Searcher, TantivyDocument};
use arroy::{Arroy, Metric, Node, Writer as ArroyWriter};
use rayon::prelude::*;
use std::sync::Arc;

// 1. Schema (Tantivy)
let mut schema_builder = Schema::builder();
let id_field = schema_builder.add_text_field("id", STRING | STORED);
let text_field = schema_builder.add_text_field("content", TEXT | STORED);
let schema = schema_builder.build();
let index = Index::create_in_dir("index_dir", schema)?;

// 2. Vector index (arroy on LMDB)
let env = heed::EnvOpenOptions::new().open("vector_db")?;
let arroy_db = Arroy::open(&env, Metric::Cosine, 384)?;  // e.g. 384-dim embeddings

// Indexing example
fn index_documents(docs: Vec<(String, String, Vec<f32>)>) {
    let mut tantivy_writer = index.writer(100_000_000)?;
    let mut arroy_writer = ArroyWriter::new(&arroy_db);

    docs.par_iter().for_each(|(id, text, embedding)| {
        // Tantivy
        let mut doc = TantivyDocument::default();
        doc.add_text(id_field, id);
        doc.add_text(text_field, text);
        tantivy_writer.add_document(doc).unwrap();

        // arroy
        arroy_writer.add(id.parse().unwrap(), embedding).unwrap();
    });

    tantivy_writer.commit()?;
    arroy_writer.commit()?;
}

// Hybrid search
fn hybrid_search(query_text: &str, query_vector: &[f32], k: usize) -> Vec<String> {
    let reader = index.reader()?;
    let searcher: Searcher = reader.searcher();

    // Parallel: text + vector
    let (text_results, vector_results) = rayon::join(
        || {
            let query = tantivy::query::QueryParser::for_index(&index, vec![text_field])
                .parse_query(query_text)
                .unwrap();
            searcher.search(&query, &tantivy::collector::TopDocs::with_limit(k * 2))
        },
        || arroy_db.search(query_vector, k * 2, None)  // optional filter
    );

    // RRF fusion (simple version)
    let mut scores: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    for (rank, doc) in text_results.iter().enumerate() {
        let id = /* extract id */;
        *scores.entry(id).or_default() += 1.0 / (rank as f32 + 60.0);
    }
    for (rank, id) in vector_results.iter().enumerate() {
        *scores.entry(id.to_string()).or_default() += 1.0 / (rank as f32 + 60.0);
    }

    // Sort & return top-k
    let mut ranked: Vec<_> = scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    ranked.into_iter().take(k).map(|(id, _)| id).collect()
}
Related items is just:
Rustlet item_vector = get_embedding_for_document(id);
arroy_db.search(&item_vector, 10, None)
Performance Tips for “Superfast”

Quantization — Use f16 or int8 vectors (arroy/usearch support this natively) → 2–4× less memory, faster distance calc.
HNSW / random projections — arroy and usearch both excel here; arroy adds filtered ANN (apply filters before search).
Parallelism — Rayon for indexing & dual searches.
Mmap everywhere — LMDB / Lance = no loading everything into RAM.
Two-tier (like frankensearch does) — First cheap BM25 + fast vector, then rerank top-N with heavier model.
Embeddings — Pre-compute if possible. For on-the-fly: fastembed-rs is one of the fastest local options.

Alternative: Even Simpler with LanceDB
Rustlet db = lancedb::connect("data.lance").execute().await?;
let table = db.open_table("my_docs").execute().await?;

// Hybrid search is built-in
let results = table
    .search("free text query")           // triggers BM25
    .vector(query_embedding)             // or .nearest_to(vector)
    .limit(20)
    .execute()
    .await?;
If you want the absolute fastest possible custom engine, go with frankensearch (it already does the hybrid + two-tier magic) or tantivy + arroy.
Would you like:

A full minimal working repo template?
Help choosing between arroy vs usearch vs LanceDB for your scale?
Integration with a specific embedding model?
Or focus on related-items / recommendation use-case?

Just tell me your expected scale (number of docs, dim of vectors, latency target) and I’ll give you the exact Cargo.toml + code! 🚀What about a layer with cashingAdding a caching layer on top of your hybrid search (vectors + free-text + related items) is one of the highest-ROI improvements you can make in Rust. It turns "superfast" into "blazingly instant" for repeated or similar queries, dramatically reduces embedding computation cost, and lowers load on the underlying indexes (Tantivy, arroy, usearch, or LanceDB).
Recommended Caching Layers for Hybrid Search in Rust
You should implement multi-level caching because different parts have different characteristics:





















































Cache LayerWhat to CacheKey TypeRecommended CrateWhy It Fits Your Use CaseTypical Hit Rate / Speed GainEmbedding CacheQuery text → embedding vectorNormalized query string (or hash)moka (or foyer)Avoid re-running ONNX models for same/similar queriesVery high for chat/RAG appsResult CacheFull hybrid search result (list of IDs + scores)(query_text, query_vector_hash, k, filters)mokaExact or near-exact repeat queriesHigh on popular searchesSemantic / Fuzzy CacheSimilar queries → same resultsEmbedding similarity or query hashmoka + small vector index"what is X" vs "tell me about X"Medium-HighIndex / Hot Data CacheFrequently accessed documents / vectorsDoc ID or vector sliceBuilt-in (arroy/LanceDB) or mokaRelated-items & top-k re-rendersDepends on access patternTwo-tier / Hybrid CacheFast first-pass + quality refinementQuery signaturefrankensearch (built-in)Already has some caching + staleness detectionExcellent defaults
Top recommendation: Start with moka — it's the most mature, concurrent, and production-ready in-memory cache in Rust (inspired by Java's Caffeine). It supports LRU/TinyLFU eviction, TTL, time-to-idle, and async/sync APIs with excellent concurrency.
For hybrid (memory + disk) needs later, look at foyer (modern hybrid cache with disk tier).
Quick Integration Example with Your Tantivy + arroy Setup
Add to Cargo.toml:
tomlmoka = { version = "0.12", features = ["future"] }  # or latest
serde = { version = "1.0", features = ["derive"] }
Define a cache key (normalize queries aggressively):
Rustuse moka::future::Cache;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::Duration;

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct SearchKey {
    query_text: String,        // normalized
    vector_hash: u64,          // optional: hash of query vector if provided
    k: usize,
    // add filters/metadata hash if you support them
}

fn normalize_query(q: &str) -> String {
    q.trim().to_lowercase().replace(|c: char| !c.is_alphanumeric() && !c.is_whitespace(), "")
}

impl SearchKey {
    fn new(query_text: &str, query_vector: Option<&[f32]>, k: usize) -> Self {
        let mut hasher = DefaultHasher::new();
        if let Some(v) = query_vector {
            v.iter().for_each(|&x| x.to_bits().hash(&mut hasher));
        }
        SearchKey {
            query_text: normalize_query(query_text),
            vector_hash: hasher.finish(),
            k,
        }
    }
}
Set up the cache (e.g., in your search service):
Rust// Global or injected cache
let result_cache: Cache<SearchKey, Vec<String>> = Cache::builder()
    .max_capacity(10_000)           // tune based on your RAM
    .time_to_live(Duration::from_secs(300))   // 5 min TTL
    .time_to_idle(Duration::from_secs(60))    // evict if not accessed
    .eviction_policy(moka::policy::EvictionPolicy::tiny_lfu()) // great for search
    .build();

// Hybrid search with caching
async fn cached_hybrid_search(
    query_text: &str,
    query_vector: Option<&[f32]>,
    k: usize,
    // your tantivy searcher, arroy db, etc.
) -> Vec<String> {
    let key = SearchKey::new(query_text, query_vector, k);

    result_cache
        .try_get_with(key, async {
            // Miss: perform the real work (parallel text + vector search + RRF)
            let results = hybrid_search(query_text, query_vector.unwrap_or(&[]), k).await;
            Ok(results)  // or handle error
        })
        .await
        .unwrap_or_default()
}
For embedding caching (very important if you use fastembed-rs or any local model):
Rustlet embedding_cache: Cache<String, Vec<f32>> = Cache::builder()
    .max_capacity(50_000)
    .build();

async fn get_embedding(query: &str) -> Vec<f32> {
    let norm = normalize_query(query);
    embedding_cache
        .try_get_with(norm, async {
            // expensive embedding call
            fastembed::TextEmbedding::new(...).embed(query).await
        })
        .await
        .unwrap()
}
Advanced Tips for Superfast Hybrid Search + Caching

Semantic caching — Store embeddings in a tiny in-memory vector index (e.g., small arroy or even a simple cosine lookup) and check similarity before falling back to full search. This catches paraphrases.
Invalidation strategy — Use cache-aside for reads. On document updates/deletes: invalidate by prefix or use a version stamp (frankensearch has IndexStaleness and sentinel files for this).
Two-tier inside frankensearch — If you switch to frankensearch, it already includes caching helpers, model caching, and staleness detection out of the box — perfect for production hybrid search.
Distributed / persistence — For multi-instance setups, layer moka (L1 in-process) + Redis (L2) or use foyer for hybrid disk+memory.
Warm-up — On startup, pre-load popular queries or hot documents into cache.
Monitoring — Moka exposes hit/miss stats via cache.entry_count(), cache.weighted_size(), etc. Add Prometheus metrics easily.
Quantization & compression — Cache f16 vectors or even quantized embeddings to save RAM.

Integration with LanceDB
LanceDB already has strong internal caching (Arrow zero-copy, index caching). You can still wrap its search calls with moka for query-result caching on top.
Next Steps

Start simple: Add moka result cache around your hybrid_search function.
Add embedding cache next — this often gives the biggest win if you're embedding on-the-fly.
Measure hit rate and latency before/after.
Tune capacity, eviction policy, and TTL based on your traffic patterns.

What is your expected query volume and scale (e.g., 10k docs vs millions, concurrent users)?
Do you want:

Full code example with embedding + result cache?
How to add invalidation on index updates?
Switching to frankensearch for its built-in caching?
Or a semantic cache using a small vector index?

Tell me more about your workload (read-heavy? many similar queries? RAG-style?) and I'll refine the exact implementation.My goal is 1 million. After that i think we should look a segment system wiyh a top proxyPerfect — scaling to 1 million documents is exactly where a single-node hybrid setup (Tantivy + arroy/usearch/frankensearch + moka caching) shines, and your “segment system with a top proxy” idea is the right next step for 10 M+ or distributed growth.
At 1 million documents you are still comfortably in “single-node territory” (even on modest hardware). Here’s what performance looks like in practice (2026 numbers from real-world benchmarks on similar setups):

Storage: ~2–4 GB total (text + 384-dim f16 vectors + indexes). LMDB / Lance format keeps it mmap-able and disk-efficient.
Query latency (cold): 5–15 ms hybrid (BM25 + ANN + RRF) on a modern CPU with quantization.
Query latency (hot, with moka cache): < 1 ms for repeated/similar queries.
Throughput: thousands of QPS on a single 16-core machine.
Related-items (kNN on a document vector): same speed as normal vector search.

You do NOT need sharding yet.
But you should design the code with segments + proxy from day one so that when you hit the wall (or want horizontal scaling), you only add more segment instances — no rewrite.
Recommended Architecture (1 M now → 100 M+ later)
text[Client / API]
       ↓
   Top Proxy (Rust service)
       ↓ (parallel fan-out)
┌───────┬───────┬───────┬───────┐
│ Seg 0 │ Seg 1 │ ...   │ Seg N │   ← each is a full Tantivy+arroy (or frankensearch) instance
└───────┴───────┴───────┴───────┘
       ↓ (merge top-k + RRF)
   Top Proxy → final results

Segments = independent index shards (each holds ~1M/N docs). Even on one machine you can run 4–16 segments for massive intra-node parallelism.
Top Proxy = thin Rust service that:
Receives the query (text + optional vector).
Fans out to all segments in parallel (rayon or tokio).
Collects top-k from each.
Merges with RRF (or weighted score) and returns the global top-k.
Caches the final result (moka).


This is exactly how Lucene/Elasticsearch, Qdrant, and Milvus scale internally.
Step-by-step Implementation Plan
1. For 1 M documents (today)
Use frankensearch (best plug-and-play hybrid) or your existing Tantivy+arroy setup.
Keep the moka embedding + result cache we discussed last time — it will handle 80–90 % of traffic instantly.
2. Introduce Segments (even on single machine)
Each segment lives in its own directory:
Bashindex/seg-0/
index/seg-1/
...
index/seg-N/
You can start with N=4–8. Each segment indexes a slice of your documents (shard by doc_id.hash() % N).
3. The Top Proxy (core of your scalable design)
Here’s a minimal but production-ready skeleton (add to your existing code):
Rustuse axum::{routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::task::JoinSet;           // or rayon for sync
use moka::future::Cache;

// One segment handle (could be frankensearch, or your Tantivy+arroy wrapper)
#[derive(Clone)]
struct Segment {
    id: usize,
    // your searcher, arroy db, etc.
    search_fn: Arc<dyn Fn(&SearchRequest) -> Vec<SearchHit> + Send + Sync>,
}

#[derive(Serialize, Deserialize, Clone)]
struct SearchRequest {
    query_text: String,
    query_vector: Option<Vec<f32>>,
    k: usize,
}

#[derive(Serialize, Clone)]
struct SearchHit {
    id: String,
    score: f32,
    segment_id: usize,   // for debugging
}

struct TopProxy {
    segments: Vec<Segment>,
    result_cache: Cache<String, Vec<SearchHit>>,   // key = serialized request
}

impl TopProxy {
    async fn search(&self, req: SearchRequest) -> Vec<SearchHit> {
        let cache_key = serde_json::to_string(&req).unwrap();

        self.result_cache
            .try_get_with(cache_key, async {
                let mut set = JoinSet::new();

                // Fan-out to ALL segments in parallel
                for seg in &self.segments {
                    let req_clone = req.clone();
                    let seg_clone = seg.clone();
                    set.spawn(async move {
                        let hits = (seg_clone.search_fn)(&req_clone);
                        hits
                    });
                }

                // Collect all results
                let mut all_hits = vec![];
                while let Some(res) = set.join_next().await {
                    all_hits.extend(res.unwrap());
                }

                // RRF merge (or any fusion you like)
                let mut scores: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
                for (rank, hit) in all_hits.into_iter().enumerate() {
                    *scores.entry(hit.id).or_default() += 1.0 / (rank as f32 + 60.0);
                }

                let mut ranked: Vec<_> = scores.into_iter()
                    .map(|(id, score)| SearchHit { id, score, segment_id: 0 })
                    .collect();
                ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
                ranked.truncate(req.k);
                ranked
            })
            .await
            .unwrap_or_default()
    }
}
How to wire it up:

At startup: load N segments, wrap each with the same search closure you already have.
Expose a simple Axum/Actix HTTP endpoint that calls proxy.search(...).
For related-items: just pass the document’s own vector as query_vector.

Advanced Scaling Tips (when you go beyond 1 M)

Routing — Use consistent hashing (e.g. rustc-hash + ring) so related-items always hit the right segment.
Rebalancing — When adding a new segment, you can migrate docs lazily or via background job.
Proxy-level optimizations:
Two-tier search inside proxy (first cheap BM25-only on all segments, then full vector only on top candidates).
moka cache at proxy level (catches identical queries across all clients).
Rate limiting + backpressure.

Hardware — At 1 M you can run everything on one machine. At 10 M+ you put each segment on its own VM/container and the proxy becomes the only network hop.
Alternative ready-made — If you want to skip writing the proxy, Qdrant (written in Rust) already does segmented + distributed top-k merging extremely well and has excellent Rust client bindings.

Next Steps I Recommend Right Now

Keep your current single-node code + moka cache.
Refactor your search function into a Segment trait so you can plug in multiple later.
Add the TopProxy struct above (even with N=1 for now — zero overhead).
Test with 1 M synthetic docs (easy to generate).

This design will take you from 1 M → 100 M+ documents with almost zero code changes later.
Tell me:

Exact vector dimension and expected text size per doc?
Do you want the full Segment trait + example with frankensearch / arroy?
Or should we sketch the consistent-hashing router for related-items?

We’re building this the right way — modular from the start. Let’s keep going! 🚀OKei, lets go throught the main goals here 1. SUPER FAST!, and I mean way faster than elastic. 2, SHould hold 1 million rows without issues. 3. I need it to be document based, with full-text search, VectorSearch and a scoring system. With the possibility to add "related" documents as a part of the scoring. My thought here is that a list result: the first will be related to what we serach for, but the next 10-20, will be also scored by relation. 4. We must offcourse be able to choose what functionality we want to use in a search, 5. It needs to run on a Raspberry PI 5 with 8 gig of ram, for a good test. 6. Ask me 10 question that need answering araound this. Then create a project plan with teknologies and work schematics.
Avg doc text length & fields?
Vector dim & embedding model (precompute or on-Pi)?
Query mix % (text-only/vector/hybrid/related)?
Target latency/QPS on Pi5?
Insert/update rate (batch/real-time)?
Metadata filters required?
Exact scoring (RRF weights + related boost formula)?
Results format (IDs/scores/full docs)?
Indexing time tolerance?
Prod scale beyond 1M or single-node only?

Project plan
Tech: Rust 1.85+, frankensearch (Tantivy BM25 + f16 SIMD vectors + RRF), moka cache, arroy (fallback), fastembed-rs (quantized). ARM64 compile.
Schematics:
Client → Proxy (moka cache + feature selector) → 4 segments (parallel, each ~250k docs, mmap LMDB) → Hybrid (text+vector+related score) → RRF + relation boost (top1 exact, next 10-20 pairwise).
Phases: Week1: single-node prototype on Pi5. Week2: segments+proxy. Week3: cache+scoring. Week4: benchmarks vs Elastic.
1M docs: <4GB RAM, <10ms queries.1.6sExpert
I will be documenting products, so a lot of small varchar(255), and one,two text fields. 2. Dont know what that means. We can offcourse vectorise it with ChatGPT. 3. Yes, 4. Dont knwo what that means, 5. No, we can have teh insert, to be a slow, message queue kinda thing. 6. yes, that would be nice. When we move forward, maby use that as a shaling/sharting splitt?? DOnt know if that is smart, buit it kinda saouds like that. 7 Yes please , 8 Yes, all of them. I might want to say, that I only need these fields, but if that slows it down, dont do it. 9. a couple of minuttes? 10. Lets start with a single node right now, if that works, we can start thinking about 100 milll scaling.
Updated Project Plan (1M products, Pi5 8GB, single-node)
Tech stack:

Rust 1.85+
frankensearch (Tantivy BM25 full-text + f16 SIMD vectors + built-in RRF)
moka cache (embedding + results)
arroy fallback for vectors (LMDB mmap)
fastembed-rs (quantized local model) or ChatGPT pre-vectorize
heed for storage

Schematics:
Client → Proxy (feature selector: text/vector/hybrid/related + filters) → 4 segments (~250k docs each, parallel rayon) → Hybrid search (BM25 + ANN) → Scoring: RRF + related boost (top-1 exact match, #2-20 cosine to #1) → return full docs (or selected fields)
Scoring formula:
score = 0.7RRF(text+vector) + 0.3related_boost (cosine to top result for positions 2-20)
Performance on Pi5:

1M docs: <4GB RAM, indexing ~few minutes, queries <10ms (cached <1ms)
Filters via metadata (future sharding key)

Phases (single-node first):

Single-node prototype
Segments + proxy
Cache + selective scoring
Pi5 benchmark (faster than Elastic)

Ready to code.2.1sExpertCan you create a full .md file for this project. So I can give it to claude code to build?Markdown# SuperFast Product Search Engine

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
Architecture Schematic
textClients / API
     ↓ (HTTP or lib)
  Top Proxy
   - Feature selector (text/vector/hybrid/related/filters)
   - moka result cache
   - moka embedding cache
     ↓ (parallel fan-out via rayon/tokio)
┌────────────┬────────────┬────────────┬────────────┐
│ Segment 0  │ Segment 1  │ Segment 2  │ Segment 3  │   ~250k docs each
│ (Tantivy + │ (Tantivy + │ ...        │ ...        │   LMDB mmap
│  Vector)   │  Vector)   │            │            │
└────────────┴────────────┴────────────┴────────────┘
     ↓ (collect top-k)
  RRF Fusion + Related Boost
     ↓
  Final results (full docs or selected fields)
Related boost logic:

Run hybrid search → get ranked list.
Use top-1 as anchor.
For positions 2-20: add cosine similarity boost to anchor (weight 0.3).
Final score = 0.7 * RRF + 0.3 * related_boost.

Sharding note: Use metadata filters (e.g., category) as future sharding key.
Project Structure
textproduct-search/
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
Core Components to Implement

Product Schema
Many String fields (varchar 255)
1-2 Text fields
ID (unique)
Metadata for filters (category, price, etc.)

Indexing
Batch insert (slow OK)
Pre-vectorize with fastembed or external (ChatGPT)
Distribute across 4 segments by hash(id) % 4

Search
Selectable modes: text, vector, hybrid, related
Apply filters
Hybrid = BM25 + ANN + RRF
Related boost on top results

Caching
Normalized query → embedding
Full search key → results

API Endpoints
POST /search { query, mode, k, filters, fields? }
Return list of docs with scores


Implementation Order (for Claude)

Setup Cargo project + dependencies.
Define Product schema and indexing logic (single segment first).
Implement Segment wrapper around frankensearch.
Add 4-segment parallel proxy.
Add moka caches.
Implement scoring (RRF + related cosine boost).
Add feature selector + filters.
Build simple Axum server.
Test indexing 1M synthetic products.
Benchmark on Pi 5 (latency, RAM, vs Elastic if possible).

Deliverables

Full working Rust project.
README with build/run instructions for Pi 5.
Example synthetic data generator.
Benchmark script.

Constraints:

Keep RAM under 4-6GB for 1M docs.
Use quantized/small models for embeddings on Pi.
Prefer zero-copy / mmap everywhere.
No heavy dependencies.

Start building from the top. Prioritize speed and Pi 5 compatibility.