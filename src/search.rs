use anyhow::{
  Context,
  Result
};

use crate::chunk::Chunk;
use crate::config::Config;
use crate::embedder::Embedder;
use crate::index::VectorIndex;
use crate::normalization;
use crate::state::{
  Document,
  State
};

#[derive(Debug)]
pub struct SearchHit {
  pub chunk:    Chunk,
  pub document: Document,
  pub score:    f32
}

pub fn search_hits(
  embedder: &dyn Embedder,
  query: &str,
  top_k: usize,
  config: &Config,
  state: &State,
  index: &VectorIndex
) -> Result<Vec<SearchHit>> {
  let query_text = if config
    .stage1
    .embedder
    .normalize_query
  {
    normalization::normalize(query)
  } else {
    query.to_string()
  };
  let query_vector =
    embedder.embed(&query_text);
  let matches =
    index.search(&query_vector, top_k);
  let filtered: Vec<_> = matches
    .into_iter()
    .filter(|(_, score)| {
      *score
        >= config
          .stage1
          .search
          .score_threshold
    })
    .collect();
  let mut results = Vec::new();
  for (idx, score) in filtered {
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
    results.push(SearchHit {
      chunk: chunk.clone(),
      document: document.clone(),
      score
    });
  }
  Ok(results)
}
