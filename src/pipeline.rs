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
use crate::embedder::TfEmbedder;
use crate::index::VectorIndex;
use crate::normalization;
use crate::state::{
  Document,
  State
};

const CHUNKS_JSONL: &str =
  "data/chunks.jsonl";

pub fn run(
  command: Command
) -> Result<()> {
  let mut state = State::load()?;
  let mut index =
    VectorIndex::from_entries(
      state.index_entries.clone()
    );
  match command {
    | Command::Ingest {
      path,
      strategy
    } => {
      ingest(
        &path, strategy, &mut state,
        &mut index
      )?;
      state.index_entries =
        index.entries().to_vec();
      emit_chunks_jsonl(&state.chunks)?;
      state.save()?;
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
      search(
        &query, top_k, &state, &index
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
  state: &mut State,
  index: &mut VectorIndex
) -> Result<()> {
  let source_files =
    collect_sources(path)?;
  if source_files.is_empty() {
    println!(
      "No text or Markdown files \
       found at {:?}",
      path
    );
    return Ok(());
  }
  let chunker = Chunker::new(strategy);
  let embedder = TfEmbedder::new();
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
    let hash = hash_text(&normalized);
    if state.has_document(&hash) {
      println!(
        "Skipping already ingested \
         {:?}",
        file
      );
      continue;
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
      token_count:
        TfEmbedder::token_count(
          &normalized
        )
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
  }
  Ok(())
}

fn collect_sources(
  path: &Path
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
        if matches!(
          ext
            .to_string_lossy()
            .to_lowercase()
            .as_str(),
          "txt" | "md"
        ) {
          files.push(entry.into_path());
        }
      }
    }
  }
  Ok(files)
}

fn hash_text(text: &str) -> String {
  let mut hasher = Sha256::new();
  hasher.update(text.as_bytes());
  format!("{:x}", hasher.finalize())
}

fn emit_chunks_jsonl(
  chunks: &[Chunk]
) -> Result<()> {
  let path =
    PathBuf::from(CHUNKS_JSONL);
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
  index: &VectorIndex
) -> Result<()> {
  if state.index_entries.is_empty() {
    println!(
      "No indexed chunks yet. Run \
       `oxbed ingest` first."
    );
    return Ok(());
  }
  let embedder = TfEmbedder::new();
  let query_vector =
    embedder.embed(query);
  let matches =
    index.search(&query_vector, top_k);
  if matches.is_empty() {
    println!(
      "No matching chunks found for \
       query."
    );
    return Ok(());
  }
  for (rank, (idx, score)) in
    matches.into_iter().enumerate()
  {
    let entry = index
      .entries()
      .get(idx)
      .context(
        "missing index entry for \
         search result"
      )?;
    let chunk = state
      .find_chunk(
        entry.chunk_id.as_str()
      )
      .context(
        "chunk metadata missing"
      )?;
    let document = state
      .find_document(
        entry.doc_id.as_str()
      )
      .context(
        "document metadata missing"
      )?;
    println!(
      "Result {} (score: {:.3})",
      rank + 1,
      score
    );
    println!(
      " → Document: {}",
      document.path
    );
    println!(
      " → Chunk: {}",
      chunk.text.trim()
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
