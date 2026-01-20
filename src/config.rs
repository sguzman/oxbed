#![allow(dead_code)]

use std::fs;

use anyhow::{
  Context,
  Result
};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
  #[serde(default)]
  pub stage1: Stage1Config,
  #[serde(default)]
  pub stage2: Stage2Config,
  #[serde(default)]
  pub stage3: Stage3Config,
  #[serde(default)]
  pub stage4: Stage4Config
}

impl Default for Config {
  fn default() -> Self {
    Self {
      stage1: Stage1Config::default(),
      stage2: Stage2Config::default(),
      stage3: Stage3Config::default(),
      stage4: Stage4Config::default()
    }
  }
}

impl Config {
  pub fn load<
    P: AsRef<std::path::Path>
  >(
    path: P
  ) -> Result<Self> {
    let path_ref = path.as_ref();
    if path_ref.exists() {
      let contents =
        fs::read_to_string(path_ref)
          .with_context(|| {
            format!(
              "read config {:?}",
              path_ref
            )
          })?;
      toml::from_str(&contents)
        .with_context(|| {
          format!(
            "parse config {:?}",
            path_ref
          )
        })
    } else {
      Ok(Self::default())
    }
  }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Stage1Config {
  #[serde(default = "default_true")]
  pub enabled:  bool,
  #[serde(default)]
  pub ingest:   Stage1Ingest,
  #[serde(default)]
  pub chunk:    Stage1Chunk,
  #[serde(default)]
  pub embedder: Stage1Embedder,
  #[serde(default)]
  pub search:   Stage1Search,
  #[serde(default)]
  pub storage:  Stage1Storage
}

impl Default for Stage1Config {
  fn default() -> Self {
    Self {
      enabled:  true,
      ingest:   Stage1Ingest::default(),
      chunk:    Stage1Chunk::default(),
      embedder: Stage1Embedder::default(
      ),
      search:   Stage1Search::default(),
      storage:  Stage1Storage::default(
      )
    }
  }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Stage1Ingest {
  #[serde(
    default = "default_extensions"
  )]
  pub extensions:        Vec<String>,
  #[serde(default = "default_true")]
  pub skip_duplicates:   bool,
  #[serde(default = "default_true")]
  pub verbose_documents: bool
}

impl Default for Stage1Ingest {
  fn default() -> Self {
    Self {
      extensions:
        default_extensions(),
      skip_duplicates:   true,
      verbose_documents: true
    }
  }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Stage1Chunk {
  #[serde(
    default = "default_max_tokens"
  )]
  pub max_tokens:              usize,
  #[serde(default = "default_overlap")]
  pub overlap:                 usize,
  #[serde(default = "default_true")]
  pub split_on_double_newline: bool,
  #[serde(default = "default_true")]
  pub dedupe_segments:         bool,
  #[serde(
    default = "default_chunk_separators"
  )]
  pub chunk_separators: Vec<String>
}

impl Default for Stage1Chunk {
  fn default() -> Self {
    Self {
      max_tokens:
        default_max_tokens(),
      overlap:
        default_overlap(),
      split_on_double_newline: true,
      dedupe_segments:         true,
      chunk_separators:
        default_chunk_separators()
    }
  }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Stage1Embedder {
  #[serde(
    default = "default_min_freq"
  )]
  pub tfidf_min_freq:  usize,
  #[serde(default = "default_true")]
  pub normalize_query: bool
}

impl Default for Stage1Embedder {
  fn default() -> Self {
    Self {
      tfidf_min_freq:  default_min_freq(
      ),
      normalize_query: true
    }
  }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Stage1Search {
  #[serde(default = "default_top_k")]
  pub top_k:           usize,
  #[serde(default)]
  pub score_threshold: f32,
  #[serde(default = "default_false")]
  pub rerank_enabled:  bool
}

impl Default for Stage1Search {
  fn default() -> Self {
    Self {
      top_k:           default_top_k(),
      score_threshold: 0.0,
      rerank_enabled:  false
    }
  }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Stage1Storage {
  #[serde(
    default = "default_state_file"
  )]
  pub state_file:  String,
  #[serde(
    default = "default_chunks_file"
  )]
  pub chunks_file: String
}

impl Default for Stage1Storage {
  fn default() -> Self {
    Self {
      state_file:  default_state_file(),
      chunks_file: default_chunks_file(
      )
    }
  }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Stage2Config {
  #[serde(default = "default_false")]
  pub enabled:        bool,
  #[serde(default = "default_true")]
  pub log_evaluation: bool,
  #[serde(default = "default_true")]
  pub run_baselines:  bool
}

impl Default for Stage2Config {
  fn default() -> Self {
    Self {
      enabled:        false,
      log_evaluation: true,
      run_baselines:  true
    }
  }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Stage3Config {
  #[serde(default = "default_false")]
  pub enabled:        bool,
  #[serde(
    default = "default_context_budget"
  )]
  pub context_budget: usize
}

impl Default for Stage3Config {
  fn default() -> Self {
    Self {
      enabled:        false,
      context_budget:
        default_context_budget()
    }
  }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Stage4Config {
  #[serde(default = "default_false")]
  pub enabled:        bool,
  #[serde(
    default = "default_checkpoint_dir"
  )]
  pub checkpoint_dir: String
}

impl Default for Stage4Config {
  fn default() -> Self {
    Self {
      enabled:        false,
      checkpoint_dir:
        default_checkpoint_dir()
    }
  }
}

fn default_true() -> bool {
  true
}

fn default_false() -> bool {
  false
}

fn default_extensions() -> Vec<String> {
  vec!["txt".into(), "md".into()]
}

fn default_max_tokens() -> usize {
  200
}

fn default_overlap() -> usize {
  32
}

fn default_chunk_separators()
-> Vec<String> {
  vec![
    "\n\n".into(),
    "\r\n\r\n".into(),
    "\n-\n".into(),
    "\n*\n".into(),
  ]
}

fn default_min_freq() -> usize {
  1
}

fn default_top_k() -> usize {
  5
}

fn default_state_file() -> String {
  "data/state.json".into()
}

fn default_chunks_file() -> String {
  "data/chunks.jsonl".into()
}

fn default_context_budget() -> usize {
  1024
}

fn default_checkpoint_dir() -> String {
  "models/".into()
}
