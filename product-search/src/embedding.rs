use std::sync::Arc;
use std::time::Duration;

use moka::sync::Cache;

/// Embedding service with moka cache.
///
/// When compiled with the `embeddings` feature, uses fastembed (ONNX) for local inference.
/// Without it, documents must have pre-computed embeddings, and query embedding returns an error.
pub struct EmbeddingService {
    #[cfg(feature = "embeddings")]
    model: fastembed::TextEmbedding,
    cache: Cache<String, Arc<Vec<f32>>>,
}

impl EmbeddingService {
    /// Initialize the embedding service.
    pub fn new() -> anyhow::Result<Self> {
        // Embedding cache: 50k entries, 1h TTL, 10m idle eviction
        let cache = Cache::builder()
            .max_capacity(50_000)
            .time_to_live(Duration::from_secs(3600))
            .time_to_idle(Duration::from_secs(600))
            .build();

        #[cfg(feature = "embeddings")]
        {
            tracing::info!("Loading embedding model (all-MiniLM-L6-v2)...");
            let model = fastembed::TextEmbedding::try_new(
                fastembed::InitOptions::new(fastembed::EmbeddingModel::AllMiniLML6V2)
                    .with_show_download_progress(true),
            )?;
            tracing::info!("Embedding model loaded successfully");
            Ok(Self { model, cache })
        }

        #[cfg(not(feature = "embeddings"))]
        {
            tracing::info!(
                "Embedding service initialized WITHOUT local model. \
                 Documents must include pre-computed embeddings. \
                 Enable the 'embeddings' feature for local ONNX inference."
            );
            Ok(Self { cache })
        }
    }

    /// Embed a single query string. Uses cache.
    pub fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let normalized = normalize_query(text);

        if let Some(cached) = self.cache.get(&normalized) {
            return Ok((*cached).clone());
        }

        #[cfg(feature = "embeddings")]
        {
            let embeddings = self.model.embed(vec![text], None)?;
            let embedding = embeddings
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("No embedding returned"))?;
            self.cache.insert(normalized, Arc::new(embedding.clone()));
            Ok(embedding)
        }

        #[cfg(not(feature = "embeddings"))]
        {
            let _ = normalized;
            Err(anyhow::anyhow!(
                "Local embedding not available. Compile with --features embeddings, \
                 or provide pre-computed embeddings in documents."
            ))
        }
    }

    /// Embed a batch of texts. Does NOT use cache (for indexing large batches).
    #[allow(dead_code)]
    pub fn embed_batch(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        #[cfg(feature = "embeddings")]
        {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let embeddings = self.model.embed(text_refs, None)?;
            Ok(embeddings)
        }

        #[cfg(not(feature = "embeddings"))]
        {
            Err(anyhow::anyhow!(
                "Local embedding not available. Compile with --features embeddings."
            ))
        }
    }

    /// Get cache entry count.
    pub fn cache_entry_count(&self) -> u64 {
        self.cache.entry_count()
    }
}

/// Normalize a query string for cache key consistency.
fn normalize_query(q: &str) -> String {
    q.trim()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Create the text to embed for a product document (concatenate searchable fields).
pub fn product_embed_text(
    name: &str,
    brand: &str,
    category: &str,
    description: &str,
) -> String {
    format!("{} {} {} {}", name, brand, category, description)
        .trim()
        .to_string()
}
