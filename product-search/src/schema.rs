use std::collections::{BTreeMap, HashMap};
use tantivy::schema::*;
use tantivy::TantivyDocument;

use crate::types::{ProductDocument, SearchHit};

/// All tantivy field handles for the product schema.
#[derive(Clone)]
pub struct ProductFields {
    pub internal_id: Field,
    pub product_id: Field,
    pub name: Field,
    pub brand: Field,
    pub category: Field,
    pub sku: Field,
    pub color: Field,
    pub size: Field,
    pub material: Field,
    pub description: Field,
    pub specifications: Field,
    pub price: Field,
    pub stock: Field,
    pub attributes: Field,
}

impl ProductFields {
    /// Fields used for BM25 full-text search.
    pub fn text_search_fields(&self) -> Vec<Field> {
        vec![
            self.name,
            self.brand,
            self.description,
            self.specifications,
            self.category,
            self.color,
            self.material,
            self.sku,
            self.attributes, // includes address, email, phone, etc.
        ]
    }

    /// All field names for mapping.
    pub fn field_names() -> &'static [&'static str] {
        &[
            "product_id",
            "name",
            "brand",
            "category",
            "sku",
            "color",
            "size",
            "material",
            "description",
            "specifications",
            "price",
            "stock",
            "attributes",
        ]
    }
}

/// Build the tantivy schema for products.
pub fn build_product_schema() -> (Schema, ProductFields) {
    let mut builder = Schema::builder();

    // Internal numeric ID for cross-referencing with arroy (u32 stored as u64)
    let internal_id = builder.add_u64_field("internal_id", INDEXED | STORED | FAST);

    // Product's external string ID (exact match)
    let product_id = builder.add_text_field("product_id", STRING | STORED);

    // Small varchar(255) fields — TEXT for BM25 + STORED for retrieval
    let name = builder.add_text_field("name", TEXT | STORED);
    let brand = builder.add_text_field("brand", TEXT | STORED);
    let category = builder.add_text_field("category", TEXT | STORED | FAST);
    let sku = builder.add_text_field("sku", STRING | STORED);
    let color = builder.add_text_field("color", TEXT | STORED);
    let size = builder.add_text_field("size", STRING | STORED);
    let material = builder.add_text_field("material", TEXT | STORED);

    // Long text fields — TEXT for full BM25
    let description = builder.add_text_field("description", TEXT | STORED);
    let specifications = builder.add_text_field("specifications", TEXT | STORED);

    // Numeric fields — FAST for range filtering
    let price = builder.add_f64_field("price", INDEXED | STORED | FAST);
    let stock = builder.add_u64_field("stock", INDEXED | STORED | FAST);

    // JSON field for arbitrary extra attributes
    let attributes = builder.add_json_field("attributes", TEXT | STORED);

    let schema = builder.build();

    let fields = ProductFields {
        internal_id,
        product_id,
        name,
        brand,
        category,
        sku,
        color,
        size,
        material,
        description,
        specifications,
        price,
        stock,
        attributes,
    };

    (schema, fields)
}

/// Convert a ProductDocument to a TantivyDocument for indexing.
pub fn product_to_tantivy_doc(
    product: &ProductDocument,
    fields: &ProductFields,
    internal_id: u32,
) -> TantivyDocument {
    let mut doc = TantivyDocument::default();

    doc.add_u64(fields.internal_id, internal_id as u64);
    doc.add_text(fields.product_id, &product.product_id);
    doc.add_text(fields.name, &product.name);
    doc.add_text(fields.brand, &product.brand);
    doc.add_text(fields.category, &product.category);
    doc.add_text(fields.sku, &product.sku);
    doc.add_text(fields.color, &product.color);
    doc.add_text(fields.size, &product.size);
    doc.add_text(fields.material, &product.material);
    doc.add_text(fields.description, &product.description);
    doc.add_text(fields.specifications, &product.specifications);
    doc.add_f64(fields.price, product.price);
    doc.add_u64(fields.stock, product.stock);

    if !product.attributes.is_empty() {
        // Convert HashMap<String, serde_json::Value> to BTreeMap<String, OwnedValue>
        let mut obj = BTreeMap::new();
        for (k, v) in &product.attributes {
            let owned = json_to_owned_value(v);
            obj.insert(k.clone(), owned);
        }
        doc.add_object(fields.attributes, obj);
    }

    doc
}

/// Extract a SearchHit from a retrieved TantivyDocument.
pub fn tantivy_doc_to_hit(
    doc: &TantivyDocument,
    schema: &Schema,
    selected_fields: Option<&[String]>,
    score: f32,
    rrf_score: Option<f32>,
    related_boost: Option<f32>,
) -> SearchHit {
    let mut fields_map = HashMap::new();
    let mut product_id = String::new();

    // Extract all named values from the document
    for field_value in doc.field_values() {
        let field = field_value.field();
        let value = field_value.value();
        let field_name = schema.get_field_name(field).to_string();

        // Skip internal_id from output
        if field_name == "internal_id" {
            continue;
        }

        // Apply field selection filter
        if let Some(selected) = selected_fields {
            if field_name != "product_id" && !selected.contains(&field_name) {
                continue;
            }
        }

        if field_name == "product_id" {
            if let Some(text) = value.as_str() {
                product_id = text.to_string();
            }
            continue;
        }

        let json_value = match value {
            OwnedValue::Str(s) => serde_json::Value::String(s.to_string()),
            OwnedValue::U64(n) => serde_json::Value::Number((*n).into()),
            OwnedValue::F64(f) => serde_json::json!(*f),
            OwnedValue::Object(obj) => object_to_json(obj),
            _ => continue,
        };

        fields_map.insert(field_name, json_value);
    }

    SearchHit {
        product_id,
        fields: fields_map,
        score,
        rrf_score,
        related_boost,
    }
}

/// Convert a serde_json::Value to a tantivy OwnedValue.
fn json_to_owned_value(v: &serde_json::Value) -> OwnedValue {
    match v {
        serde_json::Value::String(s) => OwnedValue::Str(s.clone()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_u64() {
                OwnedValue::U64(i)
            } else if let Some(i) = n.as_i64() {
                OwnedValue::I64(i)
            } else if let Some(f) = n.as_f64() {
                OwnedValue::F64(f)
            } else {
                OwnedValue::Null
            }
        }
        serde_json::Value::Bool(b) => OwnedValue::Bool(*b),
        serde_json::Value::Object(map) => {
            let mut btree = BTreeMap::new();
            for (k, v) in map {
                btree.insert(k.clone(), json_to_owned_value(v));
            }
            OwnedValue::Object(btree)
        }
        serde_json::Value::Array(arr) => {
            OwnedValue::Array(arr.iter().map(json_to_owned_value).collect())
        }
        serde_json::Value::Null => OwnedValue::Null,
    }
}

/// Convert a tantivy BTreeMap object to a serde_json::Value.
fn object_to_json(entries: &BTreeMap<String, OwnedValue>) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (key, value) in entries {
        let json_val = match value {
            OwnedValue::Str(s) => serde_json::Value::String(s.to_string()),
            OwnedValue::U64(n) => serde_json::Value::Number((*n).into()),
            OwnedValue::I64(n) => serde_json::Value::Number((*n).into()),
            OwnedValue::F64(f) => serde_json::json!(*f),
            OwnedValue::Bool(b) => serde_json::Value::Bool(*b),
            OwnedValue::Object(nested) => object_to_json(nested),
            _ => serde_json::Value::Null,
        };
        map.insert(key.clone(), json_val);
    }
    serde_json::Value::Object(map)
}

/// Compute a stable u32 internal ID from a product_id string.
/// If the ID is numeric (e.g. Norwegian org number), parse it directly.
/// Otherwise fall back to hashing.
pub fn product_id_to_internal(product_id: &str) -> u32 {
    // Try parsing as a number first (org numbers, SKUs, etc.)
    if let Ok(n) = product_id.trim().parse::<u32>() {
        return n;
    }
    // Fallback: hash for non-numeric IDs
    use std::hash::{Hash, Hasher};
    let mut hasher = ahash::AHasher::default();
    product_id.hash(&mut hasher);
    hasher.finish() as u32
}

/// Determine which segment a document belongs to based on its product_id.
pub fn segment_for_product(product_id: &str, num_segments: usize) -> usize {
    (product_id_to_internal(product_id) as usize) % num_segments
}
