mod benchmark;
mod embedding;
mod indexer;
mod proxy;
mod schema;
mod scoring;
mod segment;
mod types;

use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use serde::Deserialize;
use tokio::net::TcpListener;

use crate::benchmark::BenchmarkConfig;
use crate::embedding::EmbeddingService;
use crate::proxy::{TopProxy, DEFAULT_NUM_SEGMENTS};
use crate::types::*;

/// SuperFast Product Search Engine
#[derive(Parser, Debug)]
#[command(name = "product-search")]
#[command(about = "Blazingly fast hybrid search engine for product documents")]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value = "3030")]
    port: u16,

    /// Base directory for index data
    #[arg(short = 'd', long, default_value = ".")]
    data_dir: PathBuf,

    /// Number of internal segments
    #[arg(short = 's', long, default_value_t = DEFAULT_NUM_SEGMENTS)]
    segments: usize,

    /// Generate and index synthetic documents (with random vectors for benchmarking)
    #[arg(long)]
    generate: Option<usize>,

    /// Index documents from a JSONL file
    #[arg(long)]
    index_file: Option<PathBuf>,

    /// Batch size for indexing
    #[arg(long, default_value = "1000")]
    batch_size: usize,

    /// Run benchmark after indexing (does not start HTTP server)
    #[arg(long)]
    benchmark: bool,

    /// Number of queries per mode in benchmark
    #[arg(long, default_value = "1000")]
    bench_queries: usize,

    /// Top-k for benchmark queries
    #[arg(long, default_value = "20")]
    bench_k: usize,

    /// Skip starting the HTTP server (useful with --generate or --benchmark)
    #[arg(long)]
    no_server: bool,
}

type AppState = Arc<TopProxy>;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    tracing::info!("Starting SuperFast Product Search Engine");
    tracing::info!("Data directory: {:?}", args.data_dir);
    tracing::info!("Segments: {}", args.segments);

    // Initialize embedding service
    let embedding = Arc::new(EmbeddingService::new()?);

    // Initialize the proxy with segments
    let proxy = Arc::new(TopProxy::new(&args.data_dir, args.segments, embedding)?);

    // Handle synthetic data generation (always includes random vectors)
    if let Some(count) = args.generate {
        tracing::info!("Generating {} synthetic documents with random vectors...", count);
        let docs = indexer::generate_synthetic(count, true);
        tracing::info!("Generated {} documents, starting indexing...", docs.len());
        let (indexed, failed) = indexer::index_all(&proxy, docs, args.batch_size)?;
        tracing::info!(
            "Indexing complete: {} indexed, {} failed",
            indexed,
            failed
        );
    }

    // Handle JSONL file indexing
    if let Some(ref file_path) = args.index_file {
        tracing::info!("Loading documents from {:?}...", file_path);
        let docs = indexer::load_from_jsonl(file_path)?;
        tracing::info!("Loaded {} documents, starting indexing...", docs.len());
        let (indexed, failed) = indexer::index_all(&proxy, docs, args.batch_size)?;
        tracing::info!(
            "Indexing complete: {} indexed, {} failed",
            indexed,
            failed
        );
    }

    // Run benchmark if requested
    if args.benchmark {
        let config = BenchmarkConfig {
            num_queries: args.bench_queries,
            k: args.bench_k,
            warmup_queries: 50,
        };
        benchmark::run_benchmark(&proxy, &config);

        if args.no_server {
            return Ok(());
        }
    }

    // Skip server if --no-server
    if args.no_server {
        tracing::info!("--no-server flag set, exiting.");
        return Ok(());
    }

    // Build the HTTP server
    let app = Router::new()
        .route("/search", post(search_handler).get(search_get_handler))
        .route("/index", post(index_handler))
        .route("/index/commit", post(commit_handler))
        .route("/health", get(health_handler))
        .with_state(proxy.clone());

    let addr = format!("0.0.0.0:{}", args.port);
    tracing::info!(
        "Server listening on {} ({} docs indexed)",
        addr,
        proxy.total_doc_count()
    );

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("Server shut down gracefully");
    Ok(())
}

/// Query parameters for GET /search
#[derive(Deserialize)]
struct SearchParams {
    q: String,
    #[serde(default = "default_mode_param")]
    mode: String,
    #[serde(default = "default_k_param")]
    k: usize,
    /// Optional: comma-separated list of fields to return
    fields: Option<String>,
}

fn default_mode_param() -> String {
    "text".to_string()
}

fn default_k_param() -> usize {
    20
}

/// GET /search?q=equinor&mode=text&k=10
async fn search_get_handler(
    State(proxy): State<AppState>,
    Query(params): Query<SearchParams>,
) -> Result<impl IntoResponse, AppError> {
    let mode = match params.mode.to_lowercase().as_str() {
        "text" => SearchMode::Text,
        "vector" => SearchMode::Vector,
        "hybrid" => SearchMode::Hybrid,
        "related" => SearchMode::Related,
        _ => SearchMode::Text,
    };

    let fields = params.fields.map(|f| {
        f.split(',').map(|s| s.trim().to_string()).collect::<Vec<_>>()
    });

    let req = SearchRequest {
        query: params.q,
        mode,
        k: params.k,
        filters: FilterSpec::default(),
        fields,
        query_vector: None,
    };

    let response = tokio::task::spawn_blocking(move || proxy.search(&req))
        .await
        .map_err(|e| AppError(anyhow::anyhow!("Task join error: {}", e)))??;

    Ok(Json(response))
}

/// POST /search
async fn search_handler(
    State(proxy): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<impl IntoResponse, AppError> {
    let response = tokio::task::spawn_blocking(move || proxy.search(&req))
        .await
        .map_err(|e| AppError(anyhow::anyhow!("Task join error: {}", e)))??;

    Ok(Json(response))
}

/// POST /index
async fn index_handler(
    State(proxy): State<AppState>,
    Json(req): Json<IndexRequest>,
) -> Result<impl IntoResponse, AppError> {
    let response = tokio::task::spawn_blocking(move || proxy.index(&req))
        .await
        .map_err(|e| AppError(anyhow::anyhow!("Task join error: {}", e)))??;

    Ok(Json(response))
}

/// POST /index/commit
async fn commit_handler(
    State(proxy): State<AppState>,
) -> Result<impl IntoResponse, AppError> {
    let response = tokio::task::spawn_blocking(move || proxy.commit())
        .await
        .map_err(|e| AppError(anyhow::anyhow!("Task join error: {}", e)))??;

    Ok(Json(response))
}

/// GET /health
async fn health_handler(State(proxy): State<AppState>) -> impl IntoResponse {
    let health = HealthResponse {
        status: "ok".to_string(),
        segments: proxy.num_segments(),
        total_documents: proxy.total_doc_count(),
        cache_stats: CacheStats {
            embedding_entries: proxy.embedding.cache_entry_count(),
            result_entries: proxy.result_cache_entry_count(),
        },
    };
    Json(health)
}

/// Graceful shutdown signal handler.
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C signal handler");
    tracing::info!("Received shutdown signal");
}

/// Simple error wrapper for Axum handlers.
struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let body = serde_json::json!({
            "error": self.0.to_string()
        });
        (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
    }
}

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}
