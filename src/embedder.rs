use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{
  Path,
  PathBuf
};

use anyhow::{
  Context,
  Result
};
use unicode_segmentation::UnicodeSegmentation;

use crate::config::EmbedderKind;
use crate::stage4::ModelManifest;

pub type SparseVector =
  HashMap<String, f32>;

pub trait Embedder {
  fn name(&self) -> String;
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
  config: &crate::config::Config
) -> Result<Box<dyn Embedder>> {
  match kind {
    | EmbedderKind::Tf => {
      Ok(Box::new(TfEmbedder::new(
        config
          .stage1
          .embedder
          .tfidf_min_freq
      )))
    }
    | EmbedderKind::BagOfWords => {
      Ok(Box::new(
        BagOfWordsEmbedder::default()
      ))
    }
    | EmbedderKind::Custom {
      name,
      version
    } => {
      let dir = Path::new(
        &config.stage4.models_dir
      );
      Ok(Box::new(
        CustomEmbedder::load(
          dir,
          &name,
          version.as_deref()
        )?
      ))
    }
  }
}

#[derive(Default)]
pub struct BagOfWordsEmbedder;

impl Embedder for BagOfWordsEmbedder {
  fn name(&self) -> String {
    "bag-of-words".into()
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
    normalize_counts(counts)
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
  fn name(&self) -> String {
    "tf".into()
  }

  fn embed(
    &self,
    text: &str
  ) -> SparseVector {
    let mut counts = HashMap::new();
    for token in tokenize(text) {
      *counts
        .entry(token)
        .or_insert(0) += 1;
    }
    counts.retain(|_, &mut count| {
      count >= self.min_freq
    });
    normalize_counts(counts)
  }

  fn token_count(
    &self,
    text: &str
  ) -> usize {
    Self::token_count(text)
  }
}

pub struct CustomEmbedder {
  weights: HashMap<String, f32>,
  name:    String,
  version: String
}

impl CustomEmbedder {
  pub fn load(
    models_dir: &Path,
    name: &str,
    version: Option<&str>
  ) -> Result<Self> {
    let base = models_dir.join(name);
    let target =
      if let Some(ver) = version {
        base.join(ver)
      } else {
        find_latest_model(&base)?
      };
    let manifest_path =
      target.join("manifest.json");
    let manifest: ModelManifest =
      serde_json::from_reader(
        BufReader::new(
          File::open(&manifest_path)
            .context("open manifest")?
        )
      )
      .context("parse manifest")?;
    Ok(Self {
      weights: manifest
        .token_weights
        .clone(),
      name:    manifest.name,
      version: manifest.version
    })
  }
}

impl Embedder for CustomEmbedder {
  fn name(&self) -> String {
    format!(
      "custom:{}:{}",
      self.name, self.version
    )
  }

  fn embed(
    &self,
    text: &str
  ) -> SparseVector {
    let mut vector =
      SparseVector::new();
    for token in tokenize(text) {
      if let Some(weight) =
        self.weights.get(&token)
      {
        vector.insert(token, *weight);
      }
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

fn normalize_counts(
  counts: HashMap<String, usize>
) -> SparseVector {
  let total: f32 = counts
    .values()
    .map(|count| *count as f32)
    .sum();
  if total == 0.0 {
    return SparseVector::new();
  }
  let mut vector = SparseVector::new();
  for (token, count) in counts {
    vector.insert(
      token,
      count as f32 / total
    );
  }
  vector
}

fn find_latest_model(
  base: &Path
) -> Result<PathBuf> {
  let mut candidates: Vec<_> =
    std::fs::read_dir(base)?
      .filter_map(Result::ok)
      .filter(|entry| {
        entry.path().is_dir()
      })
      .collect();
  candidates
    .sort_by_key(|entry| entry.path());
  let entry = candidates
    .into_iter()
    .last()
    .context(
      "no custom models found"
    )?;
  Ok(entry.path())
}

fn tokenize(text: &str) -> Vec<String> {
  text
    .unicode_words()
    .map(|word| word.to_lowercase())
    .collect()
}
