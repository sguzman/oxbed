use std::collections::HashMap;
use std::fs::{
  self,
  File
};
use std::io::Write;
use std::path::{
  Path,
  PathBuf
};

use anyhow::{
  Context,
  Result
};
use sha2::{
  Digest,
  Sha256
};
use walkdir::WalkDir;

use crate::args::Command;
use crate::chunk::{
  Chunk,
  ChunkStrategy,
  Chunker
};
use crate::config::Config;
use crate::embedder::{
  Embedder,
  build_embedder
};
use crate::index::VectorIndex;
use crate::search::search_hits;
use crate::state::{
  Document,
  State
};
use crate::{
  evaluation,
  normalization,
  stage3
};

pub fn run(
  command: Command,
  config: Config
) -> Result<()> {
  let state_path = PathBuf::from(
    &config.stage1.storage.state_file
  );
  let mut state =
    State::load_from(&state_path)?;
  let mut index =
    VectorIndex::from_entries(
      state.index_entries.clone()
    );
  let embedder = build_embedder(
    config.stage1.embedder.kind,
    config
      .stage1
      .embedder
      .tfidf_min_freq
  );
  match command {
    | Command::Ingest {
      path,
      strategy,
      emit_word_tally,
      emit_normalized
    } => {
      ingest(
        &path,
        strategy,
        emit_word_tally,
        emit_normalized,
        &config,
        &mut state,
        &mut index,
        embedder.as_ref()
      )?;
      state.index_entries =
        index.entries().to_vec();
      emit_chunks_jsonl(
        &state.chunks,
        &PathBuf::from(
          &config
            .stage1
            .storage
            .chunks_file
        )
      )?;
      state.save_to(&state_path)?;
      println!(
        "Ingested {} documents ({} \
         chunks total).",
        state.documents.len(),
        state.chunks.len()
      );
    }
    | Command::Search {
      query,
      top_k
    } => {
      let resolved_top_k = top_k
        .unwrap_or(
          config.stage1.search.top_k
        );
      search(
        &query,
        resolved_top_k,
        &state,
        &index,
        embedder.as_ref(),
        &config
      )?;
    }
    | Command::Rag {
      query,
      top_k
    } => {
      let resolved_top_k = top_k
        .unwrap_or(
          config.stage1.search.top_k
        );
      stage3::run_stage3(
        &query,
        resolved_top_k,
        &config,
        &state,
        &index,
        embedder.as_ref()
      )?;
    }
    | Command::Evaluate => {
      evaluation::run_evaluation(
        &config, &state, &index
      )?;
    }
    | Command::Status => {
      status(&state)?;
    }
  }
  Ok(())
}

fn ingest(
  path: &Path,
  strategy: ChunkStrategy,
  emit_word_tally: bool,
  emit_normalized: bool,
  config: &Config,
  state: &mut State,
  index: &mut VectorIndex,
  embedder: &dyn Embedder
) -> Result<()> {
  let source_files = collect_sources(
    path,
    &config.stage1.ingest.extensions
  )?;
  if source_files.is_empty() {
    println!(
      "No text or Markdown files \
       found at {:?}",
      path
    );
    return Ok(());
  }
  let chunk_cfg = &config.stage1.chunk;
  let chunker = Chunker::with_config(
    strategy,
    chunk_cfg.max_tokens,
    chunk_cfg.overlap,
    chunk_cfg.split_on_double_newline,
    chunk_cfg.dedupe_segments,
    chunk_cfg.chunk_separators.clone()
  );
  let artifacts_dir = PathBuf::from(
    &config.stage1.storage.artifact_dir
  );
  let normalized_path =
    if emit_normalized {
      Some(
        artifacts_dir
          .join("normalized.txt")
      )
    } else {
      None
    };
  let word_tally_path =
    if emit_word_tally {
      Some(
        artifacts_dir
          .join("word_tally.csv")
      )
    } else {
      None
    };
  let mut normalized_writer =
    if let Some(path) =
      normalized_path.as_ref()
    {
      ensure_parent(path)?;
      Some(File::create(path)?)
    } else {
      None
    };
  let mut word_counts =
    if word_tally_path.is_some() {
      Some(HashMap::new())
    } else {
      None
    };
  for file in source_files {
    let content =
      fs::read_to_string(&file)
        .with_context(|| {
          format!(
            "read file {:?}",
            file
          )
        })?;
    let normalized =
      normalization::normalize(
        &content
      );
    if let Some(writer) =
      normalized_writer.as_mut()
    {
      writeln!(
        writer,
        "### {}\n",
        file.display()
      )?;
      writeln!(
        writer,
        "{}\n",
        normalized
      )?;
    }
    if let Some(counts) =
      word_counts.as_mut()
    {
      accumulate_word_counts(
        counts,
        &normalized
      );
    }
    let hash = hash_text(&normalized);
    if state.has_document(&hash) {
      println!(
        "Skipping already ingested \
         {:?}",
        file
      );
      if config
        .stage1
        .ingest
        .skip_duplicates
      {
        continue;
      }
    }
    let doc_id =
      uuid::Uuid::new_v4().to_string();
    let doc_path =
      fs::canonicalize(&file)
        .map(|p| {
          p.to_string_lossy().into()
        })
        .unwrap_or_else(|_| {
          file.to_string_lossy().into()
        });
    let document = Document {
      id:          doc_id.clone(),
      path:        doc_path,
      hash:        hash.clone(),
      token_count: embedder
        .token_count(&normalized)
    };
    let chunks = chunker
      .chunk(&doc_id, &normalized);
    if chunks.is_empty() {
      println!(
        "No chunks produced for {:?}",
        file
      );
      continue;
    }
    for chunk in chunks {
      let vector =
        embedder.embed(&chunk.text);
      index.add_chunk(
        chunk.id.clone(),
        doc_id.clone(),
        vector
      );
      state.chunks.push(chunk);
    }
    state.documents.push(document);
    if config
      .stage1
      .ingest
      .verbose_documents
    {
      println!(
        "Document {} → {} tokens",
        file.display(),
        embedder
          .token_count(&normalized)
      );
    }
  }
  if let (Some(path), Some(counts)) = (
    word_tally_path.as_ref(),
    word_counts.as_ref()
  ) {
    ensure_parent(path)?;
    emit_word_tally_csv(path, counts)?;
  }
  Ok(())
}

fn collect_sources(
  path: &Path,
  allowed_exts: &[String]
) -> Result<Vec<PathBuf>> {
  let mut files = Vec::new();
  if path.is_file() {
    files.push(path.to_path_buf());
  } else {
    for entry in WalkDir::new(path)
      .into_iter()
      .filter_map(Result::ok)
    {
      if !entry.file_type().is_file() {
        continue;
      }
      if let Some(ext) =
        entry.path().extension()
      {
        let candidate = ext
          .to_string_lossy()
          .to_lowercase();
        if allowed_exts.iter().any(
          |allowed| {
            allowed.to_lowercase()
              == candidate
          }
        ) {
          files.push(entry.into_path());
        }
      }
    }
  }
  Ok(files)
}

fn accumulate_word_counts(
  counts: &mut HashMap<String, usize>,
  text: &str
) {
  for token in text
    .split(|c: char| {
      !c.is_alphanumeric()
    })
    .filter(|part| !part.is_empty())
  {
    let word = token.to_lowercase();
    *counts.entry(word).or_insert(0) +=
      1;
  }
}

fn emit_word_tally_csv(
  path: &Path,
  counts: &HashMap<String, usize>
) -> Result<()> {
  let mut entries: Vec<_> =
    counts.iter().collect();
  entries.sort_by(|a, b| {
    b.1
      .cmp(a.1)
      .then_with(|| a.0.cmp(b.0))
  });
  let mut file = File::create(path)?;
  writeln!(file, "word,count")?;
  for (word, count) in entries {
    writeln!(
      file,
      "{},{}",
      word, count
    )?;
  }
  Ok(())
}

fn ensure_parent(
  path: &Path
) -> Result<()> {
  if let Some(parent) = path.parent() {
    fs::create_dir_all(parent)
      .with_context(|| {
        format!(
          "create directory {:?}",
          parent
        )
      })?;
  }
  Ok(())
}

fn hash_text(text: &str) -> String {
  let mut hasher = Sha256::new();
  hasher.update(text.as_bytes());
  format!("{:x}", hasher.finalize())
}

fn emit_chunks_jsonl(
  chunks: &[Chunk],
  path: &Path
) -> Result<()> {
  if let Some(parent) = path.parent() {
    fs::create_dir_all(parent)
      .with_context(|| {
        format!(
          "create chunk data \
           directory {:?}",
          parent
        )
      })?;
  }
  let mut file = File::create(&path)?;
  for chunk in chunks {
    let line =
      serde_json::to_string(chunk)?;
    writeln!(file, "{}", line)?;
  }
  Ok(())
}

fn search(
  query: &str,
  top_k: usize,
  state: &State,
  index: &VectorIndex,
  embedder: &dyn Embedder,
  config: &Config
) -> Result<()> {
  if state.index_entries.is_empty() {
    println!(
      "No indexed chunks yet. Run \
       `oxbed ingest` first."
    );
    return Ok(());
  }
  let hits = search_hits(
    embedder, query, top_k, config,
    state, index
  )?;
  if hits.is_empty() {
    println!(
      "No matching chunks found for \
       query."
    );
    return Ok(());
  }
  for (rank, hit) in
    hits.into_iter().enumerate()
  {
    println!(
      "Result {} (score: {:.3})",
      rank + 1,
      hit.score
    );
    println!(
      " → Document: {}",
      hit.document.path
    );
    println!(
      " → Chunk: {}",
      hit.chunk.text.trim()
    );
    println!("----------");
  }
  Ok(())
}

fn status(state: &State) -> Result<()> {
  println!(
    "Documents: {}",
    state.documents.len()
  );
  println!(
    "Chunks: {}",
    state.chunks.len()
  );
  if let Some(last) =
    state.documents.last()
  {
    println!(
      "Latest document: {}",
      last.path
    );
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use std::fs::File;
  use std::io::Write;
  use std::path::{
    Path,
    PathBuf
  };

  use tempfile::TempDir;
  use walkdir::WalkDir;

  use super::*;
  use crate::chunk::ChunkStrategy;
  use crate::config::{
    Config,
    EmbedderKind,
    EvaluationQuery
  };
  use crate::embedder::build_embedder;
  use crate::evaluation;
  use crate::index::VectorIndex;
  use crate::state::State;

  fn with_temp_data_dir(
    test: impl FnOnce(
      &Path,
      Config
    ) -> Result<()>
  ) -> Result<()> {
    let temp = TempDir::new()?;
    let mut config = Config::default();
    config.stage1.storage.state_file =
      temp
        .path()
        .join("state.json")
        .to_string_lossy()
        .into_owned();
    config.stage1.storage.chunks_file =
      temp
        .path()
        .join("chunks.jsonl")
        .to_string_lossy()
        .into_owned();
    test(temp.path(), config)
  }

  #[test]
  fn ingest_populates_state_and_chunks_jsonl()
  -> Result<()> {
    with_temp_data_dir(
      |path, config| {
        let corpus =
          path.join("doc.txt");
        let mut file =
          File::create(&corpus)?;
        writeln!(
          file,
          "alpha\n\nbeta\n\nalpha"
        )?;
        run(
          Command::Ingest {
            path:            corpus
              .clone(),
            strategy:
              ChunkStrategy::Structured,
            emit_word_tally: false,
            emit_normalized: false
          },
          config.clone()
        )?;
        let state_path = PathBuf::from(
          &config
            .stage1
            .storage
            .state_file
        );
        let state = State::load_from(
          &state_path
        )?;
        assert_eq!(
          state.documents.len(),
          1
        );
        assert!(
          state.chunks.len() >= 1
        );
        let chunk_file = PathBuf::from(
          &config
            .stage1
            .storage
            .chunks_file
        );
        assert!(chunk_file.exists());
        Ok(())
      }
    )
  }

  #[test]
  fn search_finds_matching_results()
  -> Result<()> {
    with_temp_data_dir(
      |path, config| {
        let corpus =
          path.join("doc2.txt");
        let mut file =
          File::create(&corpus)?;
        writeln!(file, "gamma delta")?;
        run(
          Command::Ingest {
            path:            corpus
              .clone(),
            strategy:
              ChunkStrategy::Fixed,
            emit_word_tally: false,
            emit_normalized: false
          },
          config.clone()
        )?;
        let state_path = PathBuf::from(
          &config
            .stage1
            .storage
            .state_file
        );
        let state = State::load_from(
          &state_path
        )?;
        let index =
          VectorIndex::from_entries(
            state.index_entries.clone()
          );
        let embedder = build_embedder(
          config.stage1.embedder.kind,
          config
            .stage1
            .embedder
            .tfidf_min_freq
        );
        search(
          "gamma",
          3,
          &state,
          &index,
          embedder.as_ref(),
          &config
        )?;
        Ok(())
      }
    )
  }

  #[test]
  fn evaluation_logs_runs() -> Result<()>
  {
    with_temp_data_dir(
      |path, mut config| {
        config.stage2.enabled = true;
        config.stage2.log_evaluation =
          true;
        config.stage2.runs_dir = path
          .join("runs")
          .to_string_lossy()
          .into_owned();
        config.stage2.embedder_kinds =
          vec![EmbedderKind::Tf];
        config
          .stage2
          .evaluation
          .queries =
          vec![EvaluationQuery {
            name:           "doc"
              .into(),
            query:          "alpha"
              .into(),
            expected_terms: vec![
              "alpha".into(),
            ],
            top_k:          Some(1)
          }];
        let corpus =
          path.join("doc3.txt");
        let mut file =
          File::create(&corpus)?;
        writeln!(file, "alpha beta")?;
        run(
          Command::Ingest {
            path:            corpus
              .clone(),
            strategy:
              ChunkStrategy::Fixed,
            emit_word_tally: false,
            emit_normalized: false
          },
          config.clone()
        )?;
        let state_path = PathBuf::from(
          &config
            .stage1
            .storage
            .state_file
        );
        let state = State::load_from(
          &state_path
        )?;
        let index =
          VectorIndex::from_entries(
            state.index_entries.clone()
          );
        evaluation::run_evaluation(
          &config, &state, &index
        )?;
        let mut found = false;
        for entry in WalkDir::new(
          &config.stage2.runs_dir
        )
        .into_iter()
        .filter_map(Result::ok)
        {
          if entry
            .path()
            .extension()
            .and_then(|ext| {
              ext.to_str()
            })
            == Some("json")
          {
            found = true;
            break;
          }
        }
        assert!(found);
        Ok(())
      }
    )
  }
}
