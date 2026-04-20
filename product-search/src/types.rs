use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Search mode selector — choose which capabilities to use per query.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// BM25 full-text search only
    Text,
    /// Vector ANN search only
    Vector,
    /// Hybrid: BM25 + Vector + RRF fusion
    Hybrid,
    /// Hybrid + related-document boost (top-1 anchor, 2-20 boosted by similarity)
    Related,
}

/// Filter specification for metadata-based filtering.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct FilterSpec {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub brand: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub price_min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub price_max: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub in_stock: Option<bool>,
}

/// Incoming search request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_mode")]
    pub mode: SearchMode,
    #[serde(default = "default_k")]
    pub k: usize,
    #[serde(default)]
    pub filters: FilterSpec,
    /// Optional: return only these fields. None = all fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fields: Option<Vec<String>>,
    /// Optional: pre-computed query vector. If provided, skips embedding computation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_vector: Option<Vec<f32>>,
}

fn default_mode() -> SearchMode {
    SearchMode::Hybrid
}

fn default_k() -> usize {
    20
}

/// A single search result with scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub product_id: String,
    #[serde(flatten)]
    pub fields: HashMap<String, serde_json::Value>,
    #[serde(rename = "_score")]
    pub score: f32,
    #[serde(rename = "_rrf_score", skip_serializing_if = "Option::is_none")]
    pub rrf_score: Option<f32>,
    #[serde(rename = "_related_boost", skip_serializing_if = "Option::is_none")]
    pub related_boost: Option<f32>,
}

/// Search response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchHit>,
    pub took_ms: f64,
    pub total_hits: u64,
    pub cache_hit: bool,
}

/// A product document for indexing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductDocument {
    pub product_id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub brand: String,
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub sku: String,
    #[serde(default)]
    pub color: String,
    #[serde(default)]
    pub size: String,
    #[serde(default)]
    pub material: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub specifications: String,
    #[serde(default)]
    pub price: f64,
    #[serde(default)]
    pub stock: u64,
    /// Optional extra attributes as key-value pairs.
    #[serde(default)]
    pub attributes: HashMap<String, serde_json::Value>,
    /// Pre-computed embedding vector (384-dim). If None, will be computed at index time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

/// Batch index request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexRequest {
    pub documents: Vec<ProductDocument>,
}

/// Batch index response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexResponse {
    pub indexed: usize,
    pub failed: usize,
    pub took_ms: f64,
}

/// Health check response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub segments: usize,
    pub total_documents: u64,
    pub cache_stats: CacheStats,
}

/// Cache statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub embedding_entries: u64,
    pub result_entries: u64,
}

/// Commit response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitResponse {
    pub took_ms: f64,
}

/// Internal scored result before document retrieval.
#[derive(Debug, Clone)]
pub struct ScoredResult {
    pub internal_id: u32,
    pub segment_id: usize,
    pub rrf_score: f32,
    pub final_score: f32,
    pub related_boost: Option<f32>,
}
