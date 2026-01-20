use std::cmp::Ordering;

use serde::{
  Deserialize,
  Serialize
};

use crate::embedder::SparseVector;

#[derive(
  Debug, Clone, Serialize, Deserialize,
)]
pub struct IndexEntry {
  pub chunk_id: String,
  pub doc_id:   String,
  pub vector:   SparseVector
}

pub struct VectorIndex {
  entries: Vec<IndexEntry>
}

impl VectorIndex {
  pub fn from_entries(
    entries: Vec<IndexEntry>
  ) -> Self {
    Self {
      entries
    }
  }

  pub fn add_chunk(
    &mut self,
    chunk_id: String,
    doc_id: String,
    vector: SparseVector
  ) {
    self.entries.push(IndexEntry {
      chunk_id,
      doc_id,
      vector
    });
  }

  pub fn entries(
    &self
  ) -> &[IndexEntry] {
    &self.entries
  }

  pub fn search(
    &self,
    query: &SparseVector,
    top_k: usize
  ) -> Vec<(usize, f32)> {
    if query.is_empty() {
      return Vec::new();
    }
    let mut scored: Vec<(usize, f32)> =
      self
        .entries
        .iter()
        .enumerate()
        .map(|(idx, entry)| {
          (
            idx,
            cosine_similarity(
              query,
              &entry.vector
            )
          )
        })
        .filter(|(_, score)| {
          *score > 0.0
        })
        .collect();
    scored.sort_by(|a, b| {
      b.1
        .partial_cmp(&a.1)
        .unwrap_or(Ordering::Equal)
    });
    scored.truncate(top_k);
    scored
  }
}

fn cosine_similarity(
  a: &SparseVector,
  b: &SparseVector
) -> f32 {
  let mut dot = 0.0;
  let mut norm_a = 0.0;
  let mut norm_b = 0.0;
  for value in a.values() {
    norm_a += value * value;
  }
  for value in b.values() {
    norm_b += value * value;
  }
  if norm_a == 0.0 || norm_b == 0.0 {
    return 0.0;
  }
  for (token, a_val) in a {
    if let Some(b_val) = b.get(token) {
      dot += a_val * b_val;
    }
  }
  dot / (norm_a.sqrt() * norm_b.sqrt())
}
