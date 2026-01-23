use std::collections::HashMap;
use std::fs::{
  self,
  File
};
use std::io::{
  BufRead,
  BufReader,
  BufWriter,
  Write
};
use std::path::{
  Path,
  PathBuf
};

use anyhow::{
  Context,
  Result
};
use chrono::Utc;
use serde::{
  Deserialize,
  Serialize
};
use serde_json;
use unicode_segmentation::UnicodeSegmentation;

use crate::chunk::Chunk;
use crate::config::Config;

#[derive(
  Clone, Debug, Deserialize, Serialize,
)]
pub struct ModelManifest {
  pub name:          String,
  pub version:       String,
  pub trained_at:    String,
  pub example_count: usize,
  pub token_weights:
    HashMap<String, f32>
}

pub struct TrainResult {
  pub manifest:      ModelManifest,
  pub manifest_path: PathBuf,
  pub training_data: PathBuf
}

pub fn train_model(
  config: &Config,
  name: &str,
  version: Option<&str>,
  chunks_override: Option<&Path>
) -> Result<TrainResult> {
  if name.trim().is_empty() {
    anyhow::bail!(
      "model name must be provided"
    );
  }
  let chunks_file = chunks_override
    .map(PathBuf::from)
    .unwrap_or_else(|| {
      PathBuf::from(
        &config
          .stage1
          .storage
          .chunks_file
      )
    });
  let models_dir = Path::new(
    &config.stage4.models_dir
  );
  let version_str = version
    .map(|v| v.to_string())
    .unwrap_or_else(|| {
      format!(
        "v{}",
        Utc::now()
          .format("%Y%m%dT%H%M%S")
      )
    });
  let model_dir = models_dir
    .join(name)
    .join(&version_str);
  fs::create_dir_all(&model_dir)?;
  let training_path = model_dir
    .join("training-data.jsonl");
  let mut training_writer =
    BufWriter::new(File::create(
      &training_path
    )?);

  let file = File::open(&chunks_file)
    .with_context(|| {
    format!(
      "open chunks file {:?}",
      chunks_file
    )
  })?;
  let reader = BufReader::new(file);
  let limit =
    config.stage4.training.sample_limit;
  let mut counts = HashMap::new();
  let mut examples = 0usize;
  for line in reader.lines() {
    let line = line?;
    if line.trim().is_empty() {
      continue;
    }
    let chunk: Chunk =
      serde_json::from_str(&line)
        .context("parse chunk json")?;
    accumulate_counts(
      &chunk.text,
      &mut counts
    );
    if examples < limit {
      serde_json::to_writer(
        &mut training_writer,
        &chunk
      )?;
      writeln!(training_writer)?;
      examples += 1;
    }
  }
  if examples == 0 {
    anyhow::bail!(
      "no chunks read from {:?}",
      chunks_file
    );
  }
  let total: f32 = counts
    .values()
    .map(|count| *count as f32)
    .sum();
  let mut weights = HashMap::new();
  if total > 0.0 {
    for (token, count) in counts {
      weights.insert(
        token,
        count as f32 / total
      );
    }
  }
  let manifest = ModelManifest {
    name:          name.into(),
    version:       version_str.clone(),
    trained_at:    Utc::now()
      .to_rfc3339(),
    example_count: examples,
    token_weights: weights
  };
  let manifest_path =
    model_dir.join("manifest.json");
  let mut manifest_file =
    File::create(&manifest_path)?;
  serde_json::to_writer_pretty(
    &mut manifest_file,
    &manifest
  )?;
  writeln!(manifest_file)?;
  Ok(TrainResult {
    manifest,
    manifest_path,
    training_data: training_path
  })
}

fn accumulate_counts(
  text: &str,
  counts: &mut HashMap<String, usize>
) {
  for word in text
    .unicode_words()
    .map(|word| word.to_lowercase())
  {
    *counts.entry(word).or_insert(0) +=
      1;
  }
}

#[cfg(test)]
mod tests {
  use std::fs::File;
  use std::io::Write;

  use tempfile::TempDir;

  use super::*;
  use crate::chunk::Chunk;
  use crate::config::Config;

  #[test]
  fn train_model_writes_manifest()
  -> Result<()> {
    let temp = TempDir::new()?;
    let data_dir =
      temp.path().join("data");
    fs::create_dir_all(&data_dir)?;
    let chunk_file =
      data_dir.join("chunks.jsonl");
    let mut file =
      File::create(&chunk_file)?;
    let chunk = Chunk {
      id: "c".into(),
      doc_id: "d".into(),
      text: "alpha beta".into(),
      start: 0,
      end: 0,
      strategy: crate::chunk::ChunkStrategy::Structured
    };
    serde_json::to_writer(
      &mut file, &chunk
    )?;
    writeln!(file)?;
    let mut config = Config::default();
    config.stage1.storage.chunks_file =
      chunk_file
        .to_string_lossy()
        .into();
    config.stage4.models_dir = temp
      .path()
      .join("models")
      .to_string_lossy()
      .into();
    let result = train_model(
      &config,
      "test-model",
      Some("v1"),
      None
    )?;
    assert_eq!(
      result.manifest.name,
      "test-model"
    );
    assert_eq!(
      result.manifest.version,
      "v1"
    );
    assert!(
      result
        .manifest
        .token_weights
        .contains_key("alpha")
    );
    assert!(
      result.manifest_path.exists()
    );
    Ok(())
  }
}
