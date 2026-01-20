use std::collections::HashMap;

use unicode_segmentation::UnicodeSegmentation;

use crate::config::EmbedderKind;

pub type SparseVector =
  HashMap<String, f32>;

pub trait Embedder {
  fn name(&self) -> &'static str;
  fn embed(
    &self,
    text: &str
  ) -> SparseVector;
  fn token_count(
    &self,
    text: &str
  ) -> usize;
}

pub fn build_embedder(
  kind: EmbedderKind,
  min_freq: usize
) -> Box<dyn Embedder> {
  match kind {
    | EmbedderKind::Tf => {
      Box::new(TfEmbedder::new(
        min_freq
      ))
    }
    | EmbedderKind::BagOfWords => {
      Box::new(
        BagOfWordsEmbedder::default()
      )
    }
  }
}

#[derive(Default)]
pub struct BagOfWordsEmbedder;

impl Embedder for BagOfWordsEmbedder {
  fn name(&self) -> &'static str {
    "bag-of-words"
  }

  fn embed(
    &self,
    text: &str
  ) -> SparseVector {
    let tokens = tokenize(text);
    let mut counts = HashMap::new();
    for token in tokens {
      *counts
        .entry(token)
        .or_insert(0) += 1;
    }
    let total: f32 = counts
      .values()
      .map(|count| *count as f32)
      .sum();
    if total == 0.0 {
      return SparseVector::new();
    }
    let mut vector =
      SparseVector::new();
    for (token, count) in counts {
      vector.insert(
        token,
        count as f32 / total
      );
    }
    vector
  }

  fn token_count(
    &self,
    text: &str
  ) -> usize {
    text.unicode_words().count()
  }
}

pub struct TfEmbedder {
  min_freq: usize
}

impl TfEmbedder {
  pub fn new(min_freq: usize) -> Self {
    Self {
      min_freq: min_freq.max(1)
    }
  }

  pub fn token_count(
    text: &str
  ) -> usize {
    text.unicode_words().count()
  }
}

impl Embedder for TfEmbedder {
  fn name(&self) -> &'static str {
    "tf"
  }

  fn embed(
    &self,
    text: &str
  ) -> SparseVector {
    let tokens = tokenize(text);
    let mut counts: HashMap<
      String,
      usize
    > = HashMap::new();
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

  fn token_count(
    &self,
    text: &str
  ) -> usize {
    Self::token_count(text)
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
