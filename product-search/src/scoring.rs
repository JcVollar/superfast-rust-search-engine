use std::collections::HashMap;

use crate::types::ScoredResult;

/// RRF constant (standard value from the original paper).
const RRF_K: f32 = 60.0;

/// Weight for the RRF component in final scoring.
const RRF_WEIGHT: f32 = 0.7;

/// Weight for the related-document boost component.
const RELATED_WEIGHT: f32 = 0.3;

/// Number of positions (after #1) to apply related boost to.
const RELATED_BOOST_POSITIONS: usize = 20;

/// RRF fusion across multiple ranked lists.
///
/// Each ranked list contains (internal_id, segment_id, raw_score).
/// Items are ranked by their position in each list.
/// Returns fused results sorted by RRF score descending.
pub fn rrf_fuse(ranked_lists: &[Vec<(u32, usize, f32)>]) -> Vec<ScoredResult> {
    let mut scores: HashMap<(u32, usize), f32> = HashMap::new();

    for list in ranked_lists {
        for (rank, &(internal_id, segment_id, _raw_score)) in list.iter().enumerate() {
            let key = (internal_id, segment_id);
            *scores.entry(key).or_default() += 1.0 / (rank as f32 + RRF_K);
        }
    }

    let mut results: Vec<ScoredResult> = scores
        .into_iter()
        .map(|((internal_id, segment_id), rrf_score)| ScoredResult {
            internal_id,
            segment_id,
            rrf_score,
            final_score: rrf_score,
            related_boost: None,
        })
        .collect();

    results.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap());
    results
}

/// Merge results from a single source (text-only or vector-only) across segments.
///
/// Simply sorts by raw score descending.
pub fn merge_single_source(ranked_lists: &[Vec<(u32, usize, f32)>]) -> Vec<ScoredResult> {
    let mut results: Vec<ScoredResult> = ranked_lists
        .iter()
        .flat_map(|list| {
            list.iter().map(|&(internal_id, segment_id, score)| ScoredResult {
                internal_id,
                segment_id,
                rrf_score: score,
                final_score: score,
                related_boost: None,
            })
        })
        .collect();

    results.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap());
    results
}

/// Apply related-document boost to RRF-ranked results.
///
/// - Position 1: stays as-is (the anchor).
/// - Positions 2-20: final_score = 0.7 * rrf_score + 0.3 * cosine_similarity_to_anchor.
/// - Positions 21+: keep pure RRF score.
///
/// `get_vector` retrieves the embedding for a given (internal_id, segment_id).
/// After boosting, positions 2-20 are re-sorted by final_score.
pub fn apply_related_boost<F>(results: &mut Vec<ScoredResult>, get_vector: F)
where
    F: Fn(u32, usize) -> Option<Vec<f32>>,
{
    if results.is_empty() {
        return;
    }

    // Get anchor vector (top-1 result)
    let anchor_vec = get_vector(results[0].internal_id, results[0].segment_id);

    if anchor_vec.is_none() {
        return; // No vector for anchor — skip related boost
    }
    let anchor_vec = anchor_vec.unwrap();

    // Apply boost to positions 2 through RELATED_BOOST_POSITIONS
    let boost_end = results.len().min(RELATED_BOOST_POSITIONS + 1); // +1 because index 0 is anchor
    for result in results[1..boost_end].iter_mut() {
        if let Some(doc_vec) = get_vector(result.internal_id, result.segment_id) {
            let sim = cosine_similarity(&anchor_vec, &doc_vec).max(0.0);
            result.related_boost = Some(sim);
            result.final_score = RRF_WEIGHT * result.rrf_score + RELATED_WEIGHT * sim;
        }
    }

    // Re-sort positions 2-20 by final_score (anchor stays pinned at #1)
    if boost_end > 2 {
        results[1..boost_end].sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap());
    }
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_rrf_fuse_single_list() {
        let lists = vec![vec![
            (1u32, 0usize, 10.0f32),
            (2, 0, 8.0),
            (3, 0, 5.0),
        ]];
        let results = rrf_fuse(&lists);
        assert_eq!(results.len(), 3);
        // First item should have highest RRF score
        assert_eq!(results[0].internal_id, 1);
    }

    #[test]
    fn test_rrf_fuse_two_lists() {
        // Doc 2 appears first in both lists — should rank highest
        let lists = vec![
            vec![(2, 0, 10.0), (1, 0, 8.0), (3, 0, 5.0)],
            vec![(2, 0, 9.0), (3, 0, 7.0), (1, 0, 3.0)],
        ];
        let results = rrf_fuse(&lists);
        assert_eq!(results[0].internal_id, 2);
    }

    #[test]
    fn test_related_boost_no_vectors() {
        let mut results = vec![
            ScoredResult {
                internal_id: 1,
                segment_id: 0,
                rrf_score: 0.1,
                final_score: 0.1,
                related_boost: None,
            },
        ];
        // get_vector returns None — should not crash
        apply_related_boost(&mut results, |_, _| None);
        assert_eq!(results[0].final_score, 0.1);
    }
}
