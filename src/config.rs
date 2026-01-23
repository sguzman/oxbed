#![allow(dead_code)]

use std::{
  fmt,
  fs
};

use anyhow::{
  Context,
  Result
};
use serde::{
  Deserialize,
  Deserializer
};

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
#[derive(Clone, Debug)]
pub enum EmbedderKind {
  Tf,
  BagOfWords,
  Custom {
    name:    String,
    version: Option<String>
  }
}

impl<'de> Deserialize<'de>
  for EmbedderKind
{
  fn deserialize<D>(
    deserializer: D
  ) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>
  {
    struct EmbedderKindVisitor;

    impl<'de> serde::de::Visitor<'de>
      for EmbedderKindVisitor
    {
      type Value = EmbedderKind;

      fn expecting(
        &self,
        formatter: &mut fmt::Formatter<
          '_
        >
      ) -> fmt::Result {
        formatter.write_str(
          "tf, bag-of-words, or \
           custom:<name>[:<version>]"
        )
      }

      fn visit_str<E>(
        self,
        value: &str
      ) -> Result<Self::Value, E>
      where
        E: serde::de::Error
      {
        let normalized =
          value.trim().to_lowercase();
        match normalized.as_str() {
          | "tf" => {
            Ok(EmbedderKind::Tf)
          }
          | "bag-of-words" => {
            Ok(EmbedderKind::BagOfWords)
          }
          | _ if normalized
            .starts_with("custom:") =>
          {
            let parts: Vec<_> =
              normalized
                .splitn(3, ':')
                .collect();
            let name = parts
              .get(1)
              .cloned()
              .unwrap_or_default();
            if name.is_empty() {
              return Err(
                serde::de::Error::custom(
                  "custom embedder needs a name"
                )
              );
            }
            let version = parts
              .get(2)
              .and_then(|v| {
                if v.is_empty() {
                  None
                } else {
                  Some(v.to_string())
                }
              });
            Ok(EmbedderKind::Custom {
              name: name.to_string(),
              version
            })
          }
          | _ => {
            Err(
              serde::de::Error::custom(
                format!(
                  "unknown embedder \
                   kind '{}'",
                  value
                )
              )
            )
          }
        }
      }
    }

    deserializer.deserialize_str(
      EmbedderKindVisitor
    )
  }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Stage1Embedder {
  #[serde(
    default = "default_min_freq"
  )]
  pub tfidf_min_freq:  usize,
  #[serde(default = "default_true")]
  pub normalize_query: bool,
  #[serde(
    default = "default_embedder_kind"
  )]
  pub kind:            EmbedderKind
}

impl Default for Stage1Embedder {
  fn default() -> Self {
    Self {
      tfidf_min_freq:  default_min_freq(
      ),
      normalize_query: true,
      kind:
        default_embedder_kind()
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
  pub state_file:   String,
  #[serde(
    default = "default_chunks_file"
  )]
  pub chunks_file:  String,
  #[serde(
    default = "default_artifact_dir"
  )]
  pub artifact_dir: String
}

impl Default for Stage1Storage {
  fn default() -> Self {
    Self {
      state_file:   default_state_file(
      ),
      chunks_file:  default_chunks_file(
      ),
      artifact_dir:
        default_artifact_dir()
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
  pub run_baselines:  bool,
  #[serde(
    default = "default_stage2_runs_dir"
  )]
  pub runs_dir:       String,
  #[serde(
    default = "default_stage2_embedder_kinds"
  )]
  pub embedder_kinds: Vec<EmbedderKind>,
  #[serde(default)]
  pub evaluation:     Stage2Evaluation
}

impl Default for Stage2Config {
  fn default() -> Self {
    Self {
      enabled:        false,
      log_evaluation: true,
      run_baselines:  true,
      runs_dir:
        default_stage2_runs_dir(),
      embedder_kinds:
        default_stage2_embedder_kinds(),
      evaluation:
        Stage2Evaluation::default()
    }
  }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Stage2Evaluation {
  #[serde(default)]
  pub queries: Vec<EvaluationQuery>
}

impl Default for Stage2Evaluation {
  fn default() -> Self {
    Self {
      queries: Vec::new()
    }
  }
}

#[derive(Clone, Debug, Deserialize)]
pub struct EvaluationQuery {
  pub name:           String,
  pub query:          String,
  #[serde(default)]
  pub expected_terms: Vec<String>,
  #[serde(default)]
  pub top_k:          Option<usize>
}

#[derive(Clone, Debug, Deserialize)]
pub struct Stage3Config {
  #[serde(default = "default_false")]
  pub enabled:         bool,
  #[serde(
    default = "default_context_budget"
  )]
  pub context_budget:  usize,
  #[serde(
    default = "default_stage3_prompt_template"
  )]
  pub prompt_template: String,
  #[serde(default)]
  pub reranker: Stage3RerankerConfig
}

impl Default for Stage3Config {
  fn default() -> Self {
    Self {
      enabled:         false,
      context_budget:
        default_context_budget(),
      prompt_template:
        default_stage3_prompt_template(),
      reranker:
        Stage3RerankerConfig::default()
    }
  }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Stage3RerankerConfig {
  #[serde(
    default = "default_stage3_strategies"
  )]
  pub strategies:
    Vec<Stage3RerankerStrategyConfig>
}

impl Default for Stage3RerankerConfig {
  fn default() -> Self {
    Self {
      strategies:
        default_stage3_strategies()
    }
  }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Stage3RerankerStrategyConfig
{
  pub name:          String,
  #[serde(
    default = "default_stage3_rerank_mode"
  )]
  pub mode:          Stage3RerankMode,
  #[serde(default)]
  pub boost_terms:   Vec<String>,
  #[serde(
    default = "default_stage3_boost_factor"
  )]
  pub boost_factor:  f32,
  #[serde(
    default = "default_stage3_threshold"
  )]
  pub threshold:     f32,
  #[serde(
    default = "default_stage3_hybrid_weight"
  )]
  pub hybrid_weight: f32
}

impl Default
  for Stage3RerankerStrategyConfig
{
  fn default() -> Self {
    Self {
      name:          "embedding-only"
        .into(),
      mode:
        default_stage3_rerank_mode(),
      boost_terms:   Vec::new(),
      boost_factor:
        default_stage3_boost_factor(),
      threshold:
        default_stage3_threshold(),
      hybrid_weight:
        default_stage3_hybrid_weight()
    }
  }
}

#[derive(
  Clone, Copy, Debug, Deserialize,
)]
#[serde(rename_all = "kebab-case")]
pub enum Stage3RerankMode {
  None,
  TermOverlap,
  Hybrid
}

#[derive(Clone, Debug, Deserialize)]
pub struct Stage4Config {
  #[serde(default = "default_false")]
  pub enabled:    bool,
  #[serde(
    default = "default_stage4_models_dir"
  )]
  pub models_dir: String,
  #[serde(default)]
  pub training:   Stage4TrainingConfig
}

impl Default for Stage4Config {
  fn default() -> Self {
    Self {
      enabled:    false,
      models_dir:
        default_stage4_models_dir(),
      training:
        Stage4TrainingConfig::default()
    }
  }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Stage4TrainingConfig {
  #[serde(
    default = "default_stage4_context_budget"
  )]
  pub context_budget: usize,
  #[serde(
    default = "default_stage4_sample_limit"
  )]
  pub sample_limit:   usize
}

impl Default for Stage4TrainingConfig {
  fn default() -> Self {
    Self {
      context_budget:
        default_stage4_context_budget(),
      sample_limit:
        default_stage4_sample_limit()
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

fn default_embedder_kind()
-> EmbedderKind {
  EmbedderKind::Tf
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

fn default_artifact_dir() -> String {
  "data".into()
}

fn default_stage2_runs_dir() -> String {
  "runs".into()
}

fn default_stage2_embedder_kinds()
-> Vec<EmbedderKind> {
  vec![
    EmbedderKind::Tf,
    EmbedderKind::BagOfWords,
  ]
}

fn default_stage4_models_dir() -> String
{
  "models".into()
}

fn default_stage4_context_budget()
-> usize {
  512
}

fn default_stage4_sample_limit() -> usize
{
  10_000
}

fn default_stage3_prompt_template()
-> String {
  "Question: {query}\nContext:\\
   n{context}\nAnswer:"
    .into()
}

fn default_stage3_strategies()
-> Vec<Stage3RerankerStrategyConfig> {
  vec![Stage3RerankerStrategyConfig::default()]
}

fn default_stage3_rerank_mode()
-> Stage3RerankMode {
  Stage3RerankMode::None
}

fn default_stage3_boost_factor() -> f32
{
  1.0
}

fn default_stage3_threshold() -> f32 {
  0.0
}

fn default_stage3_hybrid_weight() -> f32
{
  0.5
}

fn default_context_budget() -> usize {
  1024
}

fn default_checkpoint_dir() -> String {
  "models/".into()
}
