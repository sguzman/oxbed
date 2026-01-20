use std::collections::HashMap;

use unicode_segmentation::UnicodeSegmentation;

pub type SparseVector =
  HashMap<String, f32>;

#[derive(Default)]
pub struct TfEmbedder;

impl TfEmbedder {
  pub fn new() -> Self {
    Self
  }

  pub fn embed(
    &self,
    text: &str
  ) -> SparseVector {
    let tokens = tokenize(text);
    let total =
      tokens.len().max(1) as f32;
    let mut counts =
      SparseVector::new();
    for token in tokens {
      *counts
        .entry(token)
        .or_insert(0.0) += 1.0;
    }
    for value in counts.values_mut() {
      *value /= total;
    }
    counts
  }

  pub fn token_count(
    text: &str
  ) -> usize {
    text.unicode_words().count()
  }
}

fn tokenize(text: &str) -> Vec<String> {
  text
    .unicode_words()
    .map(|word| word.to_lowercase())
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn embedder_normalizes_counts() {
    let embedder = TfEmbedder::new();
    let vector =
      embedder.embed("foo foo bar");
    let foo = vector
      .get("foo")
      .copied()
      .unwrap_or_default();
    let bar = vector
      .get("bar")
      .copied()
      .unwrap_or_default();
    assert!(
      (foo - (2.0 / 3.0)).abs() < 1e-6
    );
    assert!(
      (bar - (1.0 / 3.0)).abs() < 1e-6
    );
    assert!(
      !vector.contains_key("missing")
    );
  }
}
