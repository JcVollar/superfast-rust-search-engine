use std::time::{Duration, Instant};

use rand::Rng;

use crate::proxy::TopProxy;
use crate::segment::VECTOR_DIMENSIONS;
use crate::types::{FilterSpec, SearchMode, SearchRequest};

/// Benchmark configuration.
pub struct BenchmarkConfig {
    pub num_queries: usize,
    pub k: usize,
    pub warmup_queries: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_queries: 1000,
            k: 20,
            warmup_queries: 50,
        }
    }
}

/// Results from a single benchmark run.
#[derive(Debug)]
pub struct BenchmarkResult {
    pub mode: String,
    pub num_queries: usize,
    pub avg_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub qps: f64,
    pub avg_results: f64,
}

/// Sample queries for benchmarking.
const BENCHMARK_QUERIES: &[&str] = &[
    "red leather jacket",
    "premium headphones",
    "cotton shirt blue",
    "outdoor backpack",
    "steel watch",
    "wireless keyboard",
    "organic cotton",
    "portable speaker",
    "vintage wallet",
    "running shoes",
    "glass lamp modern",
    "bamboo desk organizer",
    "titanium sunglasses",
    "wool scarf premium",
    "compact charger",
    "leather belt brown",
    "silicone phone case",
    "aluminum bottle",
    "ceramic mug",
    "nylon bag lightweight",
    "professional keyboard",
    "budget mouse",
    "smart lamp",
    "classic pants",
    "deluxe gloves",
    "essential notebook",
    "advanced cable",
    "rugged backpack outdoor",
    "sleek modern table",
    "heavy duty shelf",
];

/// Generate a random normalized query vector.
fn random_query_vector() -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut vec: Vec<f32> = (0..VECTOR_DIMENSIONS)
        .map(|_| rng.gen_range(-1.0f32..1.0))
        .collect();
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vec.iter_mut().for_each(|x| *x /= norm);
    }
    vec
}

/// Run a full benchmark suite.
pub fn run_benchmark(proxy: &TopProxy, config: &BenchmarkConfig) -> Vec<BenchmarkResult> {
    let total_docs = proxy.total_doc_count();
    let num_segments = proxy.num_segments();

    println!("\n{}", "=".repeat(60));
    println!("  BENCHMARK: SuperFast Product Search Engine");
    println!("{}", "=".repeat(60));
    println!("  Documents indexed: {}", total_docs);
    println!("  Segments: {}", num_segments);
    println!("  Queries per mode: {}", config.num_queries);
    println!("  Top-k: {}", config.k);
    println!("  Warmup queries: {}", config.warmup_queries);
    println!("{}", "=".repeat(60));

    let modes = vec![
        ("text", SearchMode::Text),
        ("vector", SearchMode::Vector),
        ("hybrid", SearchMode::Hybrid),
        ("related", SearchMode::Related),
    ];

    let mut results = Vec::new();

    for (mode_name, mode) in &modes {
        println!("\n  Benchmarking mode: {} ...", mode_name.to_uppercase());

        let needs_vector = matches!(
            mode,
            SearchMode::Vector | SearchMode::Hybrid | SearchMode::Related
        );

        // Warmup (fills caches, warms mmap pages)
        for i in 0..config.warmup_queries {
            let query = BENCHMARK_QUERIES[i % BENCHMARK_QUERIES.len()];
            let req = make_request(query, *mode, config.k, needs_vector);
            let _ = proxy.search(&req);
        }

        let mut latencies = Vec::with_capacity(config.num_queries);
        let mut total_results = 0u64;
        let start_all = Instant::now();

        for i in 0..config.num_queries {
            let query_idx = i % BENCHMARK_QUERIES.len();
            let base_query = BENCHMARK_QUERIES[query_idx];
            // Add variation to avoid result cache hits
            let query = if i < BENCHMARK_QUERIES.len() {
                base_query.to_string()
            } else {
                format!("{} {}", base_query, i)
            };

            let req = make_request(&query, *mode, config.k, needs_vector);

            let start = Instant::now();
            match proxy.search(&req) {
                Ok(resp) => {
                    let elapsed = start.elapsed();
                    latencies.push(elapsed);
                    total_results += resp.results.len() as u64;
                }
                Err(e) => {
                    tracing::warn!("Query {} failed: {}", i, e);
                }
            }
        }

        let total_time = start_all.elapsed();

        if latencies.is_empty() {
            println!("    [SKIPPED - all queries failed]");
            continue;
        }

        // Sort for percentiles
        latencies.sort();

        let result = BenchmarkResult {
            mode: mode_name.to_string(),
            num_queries: latencies.len(),
            avg_ms: latencies.iter().map(|d| d.as_secs_f64() * 1000.0).sum::<f64>()
                / latencies.len() as f64,
            p50_ms: percentile(&latencies, 50.0),
            p95_ms: percentile(&latencies, 95.0),
            p99_ms: percentile(&latencies, 99.0),
            min_ms: latencies.first().unwrap().as_secs_f64() * 1000.0,
            max_ms: latencies.last().unwrap().as_secs_f64() * 1000.0,
            qps: latencies.len() as f64 / total_time.as_secs_f64(),
            avg_results: total_results as f64 / latencies.len() as f64,
        };

        print_result(&result);
        results.push(result);
    }

    // Summary
    println!("\n{}", "=".repeat(60));
    println!("  SUMMARY");
    println!("{}", "=".repeat(60));
    println!(
        "  {:<12} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Mode", "Avg(ms)", "P50(ms)", "P95(ms)", "P99(ms)", "QPS"
    );
    println!("  {}", "-".repeat(58));
    for r in &results {
        println!(
            "  {:<12} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.0}",
            r.mode, r.avg_ms, r.p50_ms, r.p95_ms, r.p99_ms, r.qps
        );
    }
    println!("{}", "=".repeat(60));

    // Memory estimate
    print_memory_info(proxy);

    results
}

fn make_request(query: &str, mode: SearchMode, k: usize, with_vector: bool) -> SearchRequest {
    SearchRequest {
        query: query.to_string(),
        mode,
        k,
        filters: FilterSpec::default(),
        fields: None,
        query_vector: if with_vector {
            Some(random_query_vector())
        } else {
            None
        },
    }
}

fn percentile(sorted_latencies: &[Duration], pct: f64) -> f64 {
    if sorted_latencies.is_empty() {
        return 0.0;
    }
    let idx = ((pct / 100.0) * (sorted_latencies.len() - 1) as f64).round() as usize;
    let idx = idx.min(sorted_latencies.len() - 1);
    sorted_latencies[idx].as_secs_f64() * 1000.0
}

fn print_result(r: &BenchmarkResult) {
    println!("    Queries: {}", r.num_queries);
    println!("    Avg latency:  {:.2} ms", r.avg_ms);
    println!("    P50 latency:  {:.2} ms", r.p50_ms);
    println!("    P95 latency:  {:.2} ms", r.p95_ms);
    println!("    P99 latency:  {:.2} ms", r.p99_ms);
    println!("    Min latency:  {:.2} ms", r.min_ms);
    println!("    Max latency:  {:.2} ms", r.max_ms);
    println!("    Throughput:   {:.0} QPS", r.qps);
    println!("    Avg results:  {:.1}", r.avg_results);
}

fn print_memory_info(proxy: &TopProxy) {
    println!("\n  MEMORY USAGE (estimated):");
    let docs = proxy.total_doc_count();
    let tantivy_est_mb = (docs as f64 * 0.5) / 1024.0;
    let arroy_est_mb = (docs as f64 * 384.0 * 4.0) / (1024.0 * 1024.0);
    let cache_emb = proxy.embedding.cache_entry_count() as f64 * 384.0 * 4.0 / (1024.0 * 1024.0);
    let cache_res = proxy.result_cache_entry_count() as f64 * 5.0 / 1024.0;

    println!(
        "    Tantivy indexes: ~{:.0} MB (mmap, not all in RSS)",
        tantivy_est_mb
    );
    println!(
        "    Arroy vectors:   ~{:.0} MB (mmap, not all in RSS)",
        arroy_est_mb
    );
    println!(
        "    Embedding cache: ~{:.1} MB ({} entries)",
        cache_emb,
        proxy.embedding.cache_entry_count()
    );
    println!(
        "    Result cache:    ~{:.1} MB ({} entries)",
        cache_res,
        proxy.result_cache_entry_count()
    );
    println!("    Total on-disk:   ~{:.0} MB", tantivy_est_mb + arroy_est_mb);

    // Get actual process RSS on Windows
    #[cfg(target_os = "windows")]
    {
        if let Ok(output) = std::process::Command::new("powershell")
            .args([
                "-Command",
                &format!(
                    "(Get-Process -Id {}).WorkingSet64 / 1MB",
                    std::process::id()
                ),
            ])
            .output()
        {
            if let Ok(rss) = String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse::<f64>()
            {
                println!("    Actual RSS:      ~{:.0} MB", rss);
            }
        }
    }

    println!("{}", "=".repeat(60));
}
