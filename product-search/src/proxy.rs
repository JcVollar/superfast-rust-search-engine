use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use moka::sync::Cache;
use rayon::prelude::*;

use crate::embedding::{product_embed_text, EmbeddingService};
use crate::schema::{segment_for_product, tantivy_doc_to_hit};
use crate::scoring::{apply_related_boost, merge_single_source, rrf_fuse};
use crate::segment::Segment;
use crate::types::*;

/// Default number of internal segments.
pub const DEFAULT_NUM_SEGMENTS: usize = 4;

/// The top-level search proxy that fans out to segments and merges results.
pub struct TopProxy {
    pub segments: Vec<Segment>,
    pub embedding: Arc<EmbeddingService>,
    result_cache: Cache<String, Arc<SearchResponse>>,
}

impl TopProxy {
    /// Create a new TopProxy with N segments under `base_dir/data/`.
    pub fn new(
        base_dir: &Path,
        num_segments: usize,
        embedding: Arc<EmbeddingService>,
    ) -> anyhow::Result<Self> {
        let data_dir = base_dir.join("data");
        std::fs::create_dir_all(&data_dir)?;

        tracing::info!("Opening {} segments in {:?}", num_segments, data_dir);

        let mut segments = Vec::with_capacity(num_segments);
        for i in 0..num_segments {
            let seg = Segment::open(i, &data_dir)?;
            tracing::info!("Segment {} opened ({} docs)", i, seg.doc_count());
            segments.push(seg);
        }

        // Result cache: 10k entries, 5min TTL, 2min idle
        let result_cache = Cache::builder()
            .max_capacity(10_000)
            .time_to_live(Duration::from_secs(300))
            .time_to_idle(Duration::from_secs(120))
            .build();

        Ok(Self {
            segments,
            embedding,
            result_cache,
        })
    }

    /// Execute a search request.
    pub fn search(&self, req: &SearchRequest) -> anyhow::Result<SearchResponse> {
        let start = Instant::now();

        // Check result cache
        let cache_key = serde_json::to_string(req)?;
        if let Some(cached) = self.result_cache.get(&cache_key) {
            return Ok(SearchResponse {
                results: cached.results.clone(),
                took_ms: start.elapsed().as_secs_f64() * 1000.0,
                total_hits: cached.total_hits,
                cache_hit: true,
            });
        }

        // Use pre-computed query vector if provided, otherwise embed
        let query_vector = match req.mode {
            SearchMode::Vector | SearchMode::Hybrid | SearchMode::Related => {
                if let Some(ref qv) = req.query_vector {
                    Some(qv.clone())
                } else {
                    Some(self.embedding.embed(&req.query)?)
                }
            }
            SearchMode::Text => None,
        };

        // Fan out to all segments in parallel
        let fetch_k = req.k * 3; // Over-fetch for better fusion quality

        let segment_results: Vec<_> = self
            .segments
            .par_iter()
            .map(|seg| {
                let text_results = match req.mode {
                    SearchMode::Text | SearchMode::Hybrid | SearchMode::Related => {
                        seg.search_text(&req.query, fetch_k).unwrap_or_default()
                    }
                    SearchMode::Vector => vec![],
                };

                let vector_results = match req.mode {
                    SearchMode::Vector | SearchMode::Hybrid | SearchMode::Related => {
                        if let Some(ref qv) = query_vector {
                            seg.search_vector(qv, fetch_k).unwrap_or_default()
                        } else {
                            vec![]
                        }
                    }
                    SearchMode::Text => vec![],
                };

                (seg.id, text_results, vector_results)
            })
            .collect();

        // Build ranked lists with segment IDs
        let mut text_lists: Vec<Vec<(u32, usize, f32)>> = Vec::new();
        let mut vector_lists: Vec<Vec<(u32, usize, f32)>> = Vec::new();

        for (seg_id, text_res, vec_res) in &segment_results {
            if !text_res.is_empty() {
                text_lists.push(
                    text_res
                        .iter()
                        .map(|&(id, score)| (id, *seg_id, score))
                        .collect(),
                );
            }
            if !vec_res.is_empty() {
                vector_lists.push(
                    vec_res
                        .iter()
                        .map(|&(id, score)| (id, *seg_id, score))
                        .collect(),
                );
            }
        }

        // Fuse/merge based on mode
        let mut scored = match req.mode {
            SearchMode::Text => merge_single_source(&text_lists),
            SearchMode::Vector => merge_single_source(&vector_lists),
            SearchMode::Hybrid | SearchMode::Related => {
                let mut all_lists = text_lists;
                all_lists.extend(vector_lists);
                rrf_fuse(&all_lists)
            }
        };

        // Apply related boost if requested
        if req.mode == SearchMode::Related && !scored.is_empty() {
            apply_related_boost(&mut scored, |internal_id, segment_id| {
                if segment_id < self.segments.len() {
                    self.segments[segment_id]
                        .get_vector(internal_id)
                        .ok()
                        .flatten()
                } else {
                    None
                }
            });
        }

        // Deduplicate by internal_id (keep highest-scored entry)
        {
            let mut seen = std::collections::HashSet::new();
            scored.retain(|sr| seen.insert(sr.internal_id));
        }

        // Truncate to requested k
        let total_hits = scored.len() as u64;
        scored.truncate(req.k);

        // Fetch full documents
        let selected_fields = req.fields.as_deref();
        let results: Vec<SearchHit> = scored
            .iter()
            .filter_map(|sr| {
                if sr.segment_id >= self.segments.len() {
                    return None;
                }
                let seg = &self.segments[sr.segment_id];
                let doc = seg.get_document(sr.internal_id).ok()??;
                Some(tantivy_doc_to_hit(
                    &doc,
                    &seg.schema,
                    selected_fields,
                    sr.final_score,
                    Some(sr.rrf_score),
                    sr.related_boost,
                ))
            })
            .collect();

        let response = SearchResponse {
            results,
            took_ms: start.elapsed().as_secs_f64() * 1000.0,
            total_hits,
            cache_hit: false,
        };

        // Cache the response
        self.result_cache
            .insert(cache_key, Arc::new(response.clone()));

        Ok(response)
    }

    /// Index a batch of documents, distributing to the correct segments.
    pub fn index(&self, req: &IndexRequest) -> anyhow::Result<IndexResponse> {
        let start = Instant::now();

        // Partition documents by segment
        let num_segments = self.segments.len();
        let mut partitions: Vec<Vec<ProductDocument>> = vec![vec![]; num_segments];

        // Test once if embedding is available (avoid spamming warnings)
        let embedding_available = self.embedding.embed("test").is_ok();

        for mut doc in req.documents.clone() {
            // Generate embedding if not provided and embedding model is available
            if doc.embedding.is_none() && embedding_available {
                let text =
                    product_embed_text(&doc.name, &doc.brand, &doc.category, &doc.description);
                if let Ok(emb) = self.embedding.embed(&text) {
                    doc.embedding = Some(emb);
                }
            }

            let seg_idx = segment_for_product(&doc.product_id, num_segments);
            partitions[seg_idx].push(doc);
        }

        // Index each partition into its segment
        let mut total_indexed = 0;
        let mut total_failed = 0;

        for (i, partition) in partitions.iter().enumerate() {
            if partition.is_empty() {
                continue;
            }
            match self.segments[i].index_batch(partition) {
                Ok(count) => total_indexed += count,
                Err(e) => {
                    tracing::error!("Segment {} index error: {}", i, e);
                    total_failed += partition.len();
                }
            }
        }

        Ok(IndexResponse {
            indexed: total_indexed,
            failed: total_failed,
            took_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Commit all segments (tantivy + arroy tree build).
    pub fn commit(&self) -> anyhow::Result<CommitResponse> {
        let start = Instant::now();

        for (i, seg) in self.segments.iter().enumerate() {
            tracing::info!("Committing segment {}...", i);
            seg.commit()?;
            tracing::info!("Segment {} committed ({} docs)", i, seg.doc_count());
        }

        // Invalidate result cache after new data is committed
        self.result_cache.invalidate_all();

        Ok(CommitResponse {
            took_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Get total document count across all segments.
    pub fn total_doc_count(&self) -> u64 {
        self.segments.iter().map(|s| s.doc_count()).sum()
    }

    /// Get number of segments.
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Get result cache entry count.
    pub fn result_cache_entry_count(&self) -> u64 {
        self.result_cache.entry_count()
    }
}
