use std::collections::HashMap;

use unicode_segmentation::UnicodeSegmentation;

pub type SparseVector =
  HashMap<String, f32>;

pub struct TfEmbedder {
  min_freq: usize
}

impl TfEmbedder {
  pub fn new(min_freq: usize) -> Self {
    Self {
      min_freq: min_freq.max(1)
    }
  }

  pub fn embed(
    &self,
    text: &str
  ) -> SparseVector {
    let tokens = tokenize(text);
    let mut counts: std::collections::HashMap<
      String,
      usize
    > = std::collections::HashMap::new();
    for token in tokens {
      *counts
        .entry(token)
        .or_insert(0) += 1;
    }
    let entries: Vec<_> = counts
      .into_iter()
      .filter(|(_, count)| {
        *count >= self.min_freq
      })
      .collect();
    let total: f32 = entries
      .iter()
      .map(|(_, count)| *count as f32)
      .sum();
    if total == 0.0 {
      return SparseVector::new();
    }
    let mut vector =
      SparseVector::new();
    for (token, count) in entries {
      vector.insert(
        token,
        count as f32 / total
      );
    }
    vector
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
    let embedder = TfEmbedder::new(1);
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
