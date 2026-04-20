use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

use rand::Rng;

use crate::segment::VECTOR_DIMENSIONS;
use crate::types::{IndexRequest, ProductDocument};
use crate::proxy::TopProxy;

/// Word lists for synthetic data generation.
const ADJECTIVES: &[&str] = &[
    "Premium", "Classic", "Ultra", "Deluxe", "Essential", "Professional", "Vintage",
    "Modern", "Sleek", "Rugged", "Lightweight", "Heavy-Duty", "Compact", "Portable",
    "Luxury", "Budget", "Smart", "Organic", "Natural", "Advanced",
];

const MATERIALS: &[&str] = &[
    "Leather", "Cotton", "Steel", "Aluminum", "Wood", "Bamboo", "Plastic",
    "Ceramic", "Glass", "Silicone", "Nylon", "Polyester", "Titanium", "Carbon Fiber",
    "Copper", "Brass", "Rubber", "Wool", "Linen", "Denim",
];

const PRODUCTS: &[&str] = &[
    "Jacket", "Shirt", "Pants", "Shoes", "Watch", "Bag", "Wallet", "Belt",
    "Hat", "Gloves", "Scarf", "Sunglasses", "Backpack", "Laptop Stand", "Phone Case",
    "Headphones", "Speaker", "Keyboard", "Mouse", "Cable", "Charger", "Lamp",
    "Mug", "Bottle", "Notebook", "Pen", "Desk Organizer", "Chair", "Table", "Shelf",
];

const COLORS: &[&str] = &[
    "Red", "Blue", "Green", "Black", "White", "Navy", "Gray", "Brown",
    "Beige", "Olive", "Burgundy", "Teal", "Coral", "Gold", "Silver",
];

const BRANDS: &[&str] = &[
    "NordicCraft", "UrbanEdge", "PeakGear", "EcoLine", "TechVault", "SkyBound",
    "IronForge", "CrystalWave", "ThunderBolt", "SilverStream", "GreenHaven",
    "BlueCrest", "RedFox", "GoldenPath", "OakHeart", "StormPeak", "CoralReef",
    "ArcticWind", "SunForge", "MoonStone",
];

const CATEGORIES: &[&str] = &[
    "Electronics", "Clothing", "Accessories", "Home & Garden", "Sports",
    "Office", "Kitchen", "Outdoor", "Travel", "Fitness",
];

const SIZES: &[&str] = &[
    "XS", "S", "M", "L", "XL", "XXL", "One Size", "Small", "Medium", "Large",
];

/// Generate synthetic product documents for testing.
/// If `with_vectors` is true, includes random 384-dim embeddings (for benchmarking).
pub fn generate_synthetic(count: usize, with_vectors: bool) -> Vec<ProductDocument> {
    let mut rng = rand::thread_rng();
    let mut docs = Vec::with_capacity(count);

    for i in 0..count {
        let adj = ADJECTIVES[rng.gen_range(0..ADJECTIVES.len())];
        let material = MATERIALS[rng.gen_range(0..MATERIALS.len())];
        let product = PRODUCTS[rng.gen_range(0..PRODUCTS.len())];
        let color = COLORS[rng.gen_range(0..COLORS.len())];
        let brand = BRANDS[rng.gen_range(0..BRANDS.len())];
        let category = CATEGORIES[rng.gen_range(0..CATEGORIES.len())];
        let size = SIZES[rng.gen_range(0..SIZES.len())];

        let name = format!("{} {} {} {}", adj, color, material, product);
        let price: f64 = (rng.gen_range(5.0f64..500.0) * 100.0).round() / 100.0;
        let stock = rng.gen_range(0..1000);
        let sku = format!("SKU-{:08}", i);

        let description = format!(
            "The {} by {} is a {} {} made from high-quality {}. \
             Available in {} color, this {} is perfect for everyday use. \
             Features include durable construction, comfortable design, and modern aesthetics. \
             Ideal for {} enthusiasts looking for {} quality gear.",
            name, brand, adj.to_lowercase(), product.to_lowercase(), material.to_lowercase(),
            color.to_lowercase(), product.to_lowercase(),
            category.to_lowercase(), adj.to_lowercase(),
        );

        let specifications = format!(
            "Material: {}. Color: {}. Size: {}. Weight: {:.1}kg. \
             Warranty: {} months. Origin: Imported.",
            material, color, size,
            rng.gen_range(0.1..5.0),
            rng.gen_range(6..36),
        );

        let mut attributes = HashMap::new();
        attributes.insert(
            "rating".to_string(),
            serde_json::json!(rng.gen_range(1.0..5.0)),
        );
        attributes.insert(
            "reviews".to_string(),
            serde_json::json!(rng.gen_range(0..500)),
        );

        docs.push(ProductDocument {
            product_id: sku.clone(),
            name,
            brand: brand.to_string(),
            category: category.to_string(),
            sku,
            color: color.to_string(),
            size: size.to_string(),
            material: material.to_string(),
            description,
            specifications,
            price,
            stock,
            attributes,
            embedding: if with_vectors {
                // Random normalized vector for benchmarking
                let mut vec: Vec<f32> = (0..VECTOR_DIMENSIONS)
                    .map(|_| rng.gen_range(-1.0f32..1.0))
                    .collect();
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    vec.iter_mut().for_each(|x| *x /= norm);
                }
                Some(vec)
            } else {
                None
            }
        });
    }

    docs
}

/// Load product documents from a JSON lines file.
pub fn load_from_jsonl(path: &Path) -> anyhow::Result<Vec<ProductDocument>> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut docs = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let doc: ProductDocument = serde_json::from_str(&line)?;
        docs.push(doc);
    }

    Ok(docs)
}

/// Index all documents via the proxy in batches with progress logging.
pub fn index_all(
    proxy: &TopProxy,
    docs: Vec<ProductDocument>,
    batch_size: usize,
) -> anyhow::Result<(usize, usize)> {
    let total = docs.len();
    let mut total_indexed = 0;
    let mut total_failed = 0;

    for (batch_idx, chunk) in docs.chunks(batch_size).enumerate() {
        let req = IndexRequest {
            documents: chunk.to_vec(),
        };

        let resp = proxy.index(&req)?;
        total_indexed += resp.indexed;
        total_failed += resp.failed;

        let progress = ((batch_idx + 1) * batch_size).min(total);
        tracing::info!(
            "Indexed {}/{} documents ({} failed) - {:.1}ms",
            progress,
            total,
            total_failed,
            resp.took_ms
        );
    }

    // Commit after all batches
    tracing::info!("Committing all segments...");
    let commit_resp = proxy.commit()?;
    tracing::info!("Commit completed in {:.1}ms", commit_resp.took_ms);

    Ok((total_indexed, total_failed))
}
