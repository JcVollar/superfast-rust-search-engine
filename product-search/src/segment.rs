use std::path::Path;
use std::sync::Mutex;

use arroy::distances::Cosine;
use arroy::{Database as ArroyDb, Reader as ArroyReader, Writer as ArroyWriter};
use heed::EnvOpenOptions;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tantivy::collector::TopDocs;
use tantivy::query::{QueryParser, TermQuery};
use tantivy::schema::*;
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term};

use crate::schema::{
    build_product_schema, product_id_to_internal, product_to_tantivy_doc, ProductFields,
};
use crate::types::ProductDocument;

/// Vector dimensions. Currently sized for OpenAI text-embedding-3-large (3072-dim).
/// Change to 384 for all-MiniLM-L6-v2 when targeting Pi 5 production.
pub const VECTOR_DIMENSIONS: usize = 3072;

/// arroy index identifier (u16, we use a single index per segment).
const ARROY_INDEX: u16 = 0;

/// A single search segment containing a tantivy index + arroy vector database.
pub struct Segment {
    pub id: usize,

    // Tantivy
    pub schema: Schema,
    pub fields: ProductFields,
    index: Index,
    reader: IndexReader,
    writer: Mutex<IndexWriter>,

    // Arroy (vector)
    arroy_env: heed::Env,
    arroy_db: ArroyDb<Cosine>,
}

/// Result from a single-segment search: (internal_id, score).
pub type SegmentSearchResult = Vec<(u32, f32)>;

impl Segment {
    /// Open or create a segment at `base_dir/seg-{id}/`.
    pub fn open(id: usize, base_dir: &Path) -> anyhow::Result<Self> {
        let seg_dir = base_dir.join(format!("seg-{}", id));
        let tantivy_dir = seg_dir.join("tantivy");
        let arroy_dir = seg_dir.join("arroy");

        std::fs::create_dir_all(&tantivy_dir)?;
        std::fs::create_dir_all(&arroy_dir)?;

        // Build schema
        let (schema, fields) = build_product_schema();

        // Open or create tantivy index
        let index = match Index::open_in_dir(&tantivy_dir) {
            Ok(idx) => idx,
            Err(_) => Index::create_in_dir(&tantivy_dir, schema.clone())?,
        };

        // Manual reload — commit() explicitly reloads. Avoids per-segment background
        // file_watcher threads that poll meta.json and warn loudly when paths vanish
        // (e.g. during dev resets).
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        reader.reload()?;

        // 200MB writer budget, single worker thread to reduce concurrent file writes
        // per segment (Windows Defender real-time scanning fights multi-threaded writes).
        let writer = index.writer_with_num_threads(1, 200_000_000)?;

        // Open LMDB environment for arroy
        let arroy_env = unsafe {
            EnvOpenOptions::new()
                .map_size(2 * 1024 * 1024 * 1024) // 2GB virtual
                .max_dbs(10)
                .open(&arroy_dir)?
        };

        // Create arroy database
        let mut wtxn = arroy_env.write_txn()?;
        let arroy_db: ArroyDb<Cosine> = arroy_env.create_database(&mut wtxn, Some("vectors"))?;
        wtxn.commit()?;

        Ok(Segment {
            id,
            schema,
            fields,
            index,
            reader,
            writer: Mutex::new(writer),
            arroy_env,
            arroy_db,
        })
    }

    /// Index a batch of product documents into this segment (upsert).
    /// Deletes any existing doc with the same internal_id before inserting.
    /// Does NOT commit — call `commit()` separately after batching.
    pub fn index_batch(&self, docs: &[ProductDocument]) -> anyhow::Result<usize> {
        let writer = self.writer.lock().unwrap();
        let mut wtxn = self.arroy_env.write_txn()?;
        let arroy_writer = ArroyWriter::new(self.arroy_db, ARROY_INDEX, VECTOR_DIMENSIONS);
        let mut count = 0;

        for doc in docs {
            let internal_id = product_id_to_internal(&doc.product_id);

            // Delete existing doc with same ID (upsert) — ensures uniqueness
            let delete_term = Term::from_field_u64(self.fields.internal_id, internal_id as u64);
            writer.delete_term(delete_term);

            // Add to tantivy
            let tantivy_doc = product_to_tantivy_doc(doc, &self.fields, internal_id);
            writer.add_document(tantivy_doc)?;

            // Add/overwrite vector in arroy (add_item overwrites by ItemId)
            if let Some(ref embedding) = doc.embedding {
                if embedding.len() == VECTOR_DIMENSIONS {
                    arroy_writer.add_item(&mut wtxn, internal_id, embedding)?;
                }
            }

            count += 1;
        }

        wtxn.commit()?;
        Ok(count)
    }

    /// Commit the tantivy index and build arroy trees.
    pub fn commit(&self) -> anyhow::Result<()> {
        // Commit tantivy
        {
            let mut writer = self.writer.lock().unwrap();
            writer.commit()?;
        }

        // Build arroy tree index
        let mut wtxn = self.arroy_env.write_txn()?;
        let arroy_writer = ArroyWriter::new(self.arroy_db, ARROY_INDEX, VECTOR_DIMENSIONS);

        // Check if we need to build (has vectors and needs rebuild)
        let rtxn = self.arroy_env.read_txn()?;
        let needs_build = arroy_writer.need_build(&rtxn)?;
        drop(rtxn);

        if needs_build {
            let mut rng = StdRng::seed_from_u64(self.id as u64);
            arroy_writer.builder(&mut rng).build(&mut wtxn)?;
            wtxn.commit()?;
        }

        // Reload tantivy reader to pick up new docs
        self.reader.reload()?;

        Ok(())
    }

    /// BM25 full-text search. Returns (internal_id, bm25_score) pairs.
    /// Uses AND (conjunction) mode: all terms must match.
    /// Supports phrase queries with quotes: "Markveien 3B" matches exact phrase.
    pub fn search_text(&self, query: &str, k: usize) -> anyhow::Result<SegmentSearchResult> {
        let searcher = self.reader.searcher();
        let mut query_parser =
            QueryParser::for_index(&self.index, self.fields.text_search_fields());
        // AND mode: all search terms must be present in the document
        query_parser.set_conjunction_by_default();
        let parsed = query_parser.parse_query(query)?;

        let top_docs = searcher.search(&parsed, &TopDocs::with_limit(k))?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_addr) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_addr)?;
            if let Some(id) = self.extract_internal_id(&doc) {
                results.push((id, score));
            }
        }

        Ok(results)
    }

    /// Vector ANN search. Returns (internal_id, distance) pairs.
    /// Lower distance = more similar for cosine.
    pub fn search_vector(&self, vector: &[f32], k: usize) -> anyhow::Result<SegmentSearchResult> {
        let rtxn = self.arroy_env.read_txn()?;

        // Try to open the reader — if no tree is built yet, return empty
        let arroy_reader = match ArroyReader::<Cosine>::open(&rtxn, ARROY_INDEX, self.arroy_db) {
            Ok(r) => r,
            Err(_) => return Ok(vec![]),
        };

        let results = arroy_reader.nns(k).by_vector(&rtxn, vector)?;

        // Convert distance to similarity score (1 - distance for cosine)
        Ok(results
            .into_iter()
            .map(|(id, dist)| (id, 1.0 - dist))
            .collect())
    }

    /// Get the stored vector for a given internal_id.
    pub fn get_vector(&self, internal_id: u32) -> anyhow::Result<Option<Vec<f32>>> {
        let rtxn = self.arroy_env.read_txn()?;
        let arroy_writer = ArroyWriter::new(self.arroy_db, ARROY_INDEX, VECTOR_DIMENSIONS);
        Ok(arroy_writer.item_vector(&rtxn, internal_id)?)
    }

    /// Retrieve a full document by internal_id from tantivy.
    pub fn get_document(&self, internal_id: u32) -> anyhow::Result<Option<TantivyDocument>> {
        let searcher = self.reader.searcher();
        let term = Term::from_field_u64(self.fields.internal_id, internal_id as u64);
        let query = TermQuery::new(term, IndexRecordOption::Basic);
        let top_docs = searcher.search(&query, &TopDocs::with_limit(1))?;

        if let Some((_score, doc_addr)) = top_docs.first() {
            let doc: TantivyDocument = searcher.doc(*doc_addr)?;
            Ok(Some(doc))
        } else {
            Ok(None)
        }
    }

    /// Get the total number of documents in this segment's tantivy index.
    pub fn doc_count(&self) -> u64 {
        let searcher = self.reader.searcher();
        searcher.num_docs()
    }

    /// Extract the internal_id from a retrieved tantivy document.
    fn extract_internal_id(&self, doc: &TantivyDocument) -> Option<u32> {
        for field_value in doc.field_values() {
            if field_value.field() == self.fields.internal_id {
                if let tantivy::schema::OwnedValue::U64(id) = field_value.value() {
                    return Some(*id as u32);
                }
            }
        }
        None
    }
}
