#![allow(dead_code)]

use std::path::{
  Path,
  PathBuf
};
use std::{
  env,
  fs
};

use anyhow::Context;
use serde::{
  Deserialize,
  Serialize
};

use crate::chunk::Chunk;
use crate::index::IndexEntry;

pub fn data_dir() -> PathBuf {
  if let Ok(override_dir) =
    env::var("OXBED_DATA_DIR")
  {
    PathBuf::from(override_dir)
  } else {
    PathBuf::from("data")
  }
}

#[derive(
  Debug, Serialize, Deserialize, Default,
)]
pub struct State {
  pub documents:     Vec<Document>,
  pub chunks:        Vec<Chunk>,
  pub index_entries: Vec<IndexEntry>
}

#[derive(
  Debug, Clone, Serialize, Deserialize,
)]
pub struct Document {
  pub id:          String,
  pub path:        String,
  pub hash:        String,
  pub token_count: usize
}

impl State {
  pub fn load() -> anyhow::Result<Self>
  {
    Self::load_from(Self::path())
  }

  pub fn load_from(
    path: impl AsRef<Path>
  ) -> anyhow::Result<Self> {
    let path = path.as_ref();
    if path.exists() {
      let contents =
        fs::read_to_string(&path)
          .with_context(|| {
            format!(
              "read state from {:?}",
              path
            )
          })?;
      serde_json::from_str(&contents)
        .context(
          "parse saved Oxbed state"
        )
    } else {
      Ok(Self::default())
    }
  }

  pub fn save(
    &self
  ) -> anyhow::Result<()> {
    self.save_to(Self::path())
  }

  pub fn save_to(
    &self,
    path: impl AsRef<Path>
  ) -> anyhow::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
    {
      fs::create_dir_all(parent)
        .with_context(|| {
          format!(
            "create state directory \
             {:?}",
            parent
          )
        })?;
    }
    let serialized =
      serde_json::to_string_pretty(
        self
      )
      .context(
        "serialize Oxbed corpus state"
      )?;
    fs::write(path, serialized)
      .with_context(|| {
        format!(
          "write state to {:?}",
          path
        )
      })?;
    Ok(())
  }

  pub fn path() -> PathBuf {
    data_dir().join("state.json")
  }

  pub fn has_document(
    &self,
    hash: &str
  ) -> bool {
    self
      .documents
      .iter()
      .any(|doc| doc.hash == hash)
  }

  pub fn find_chunk(
    &self,
    chunk_id: &str
  ) -> Option<&Chunk> {
    self.chunks.iter().find(|chunk| {
      chunk.id == chunk_id
    })
  }

  pub fn find_document(
    &self,
    doc_id: &str
  ) -> Option<&Document> {
    self
      .documents
      .iter()
      .find(|doc| doc.id == doc_id)
  }
}
