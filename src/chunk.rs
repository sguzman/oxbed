use std::collections::HashSet;
use std::fmt;

use clap::ValueEnum;
use serde::{
  Deserialize,
  Serialize
};
use uuid::Uuid;

use crate::normalization;

#[derive(
  Debug,
  Clone,
  Copy,
  Serialize,
  Deserialize,
  ValueEnum,
  PartialEq,
  Eq,
)]
#[serde(rename_all = "lowercase")]
pub enum ChunkStrategy {
  Structured,
  Fixed
}

impl fmt::Display for ChunkStrategy {
  fn fmt(
    &self,
    f: &mut fmt::Formatter<'_>
  ) -> fmt::Result {
    match self {
      | ChunkStrategy::Structured => {
        f.write_str("structured")
      }
      | ChunkStrategy::Fixed => {
        f.write_str("fixed")
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn structured_chunks_split_paragraphs_and_dedup()
   {
    let chunker = Chunker::with_config(
      ChunkStrategy::Structured,
      200,
      32,
      true,
      true,
      vec!["\n\n".into()]
    );
    let input =
      "alpha\n\nbeta\n\nalpha";
    let chunks =
      chunker.chunk("doc", input);
    assert_eq!(chunks.len(), 2);
    assert!(chunks.iter().any(|c| {
      c.text.contains("alpha")
    }));
    assert!(chunks.iter().any(|c| {
      c.text.contains("beta")
    }));
  }

  #[test]
  fn fixed_chunks_obey_overlap_and_max()
  {
    let chunker = Chunker::with_config(
      ChunkStrategy::Fixed,
      200,
      32,
      true,
      true,
      vec!["\n\n".into()]
    );
    let input = "word ".repeat(500);
    let chunks =
      chunker.chunk("doc", &input);
    assert!(chunks.len() >= 2);
    for chunk in &chunks {
      assert!(chunk.text.len() > 0);
    }
    let start_positions: Vec<_> =
      chunks
        .iter()
        .map(|c| c.start)
        .collect();
    assert!(
      start_positions
        .windows(2)
        .all(|w| w[1] > w[0])
    );
  }
}

#[derive(
  Clone, Debug, Serialize, Deserialize,
)]
pub struct Chunk {
  pub id:       String,
  pub doc_id:   String,
  pub text:     String,
  pub start:    usize,
  pub end:      usize,
  pub strategy: ChunkStrategy
}

pub struct Chunker {
  strategy: ChunkStrategy,
  max_tokens:              usize,
  overlap:                 usize,
  split_on_double_newline: bool,
  dedupe_segments:         bool,
  chunk_separators:        Vec<String>
}

impl Chunker {
  pub fn with_config(
    strategy: ChunkStrategy,
    max_tokens: usize,
    overlap: usize,
    split_on_double_newline: bool,
    dedupe_segments: bool,
    chunk_separators: Vec<String>
  ) -> Self {
    Self {
      strategy,
      max_tokens,
      overlap,
      split_on_double_newline,
      dedupe_segments,
      chunk_separators
    }
  }

  pub fn chunk(
    &self,
    doc_id: &str,
    input: &str
  ) -> Vec<Chunk> {
    match self.strategy {
      | ChunkStrategy::Structured => {
        self.structured(doc_id, input)
      }
      | ChunkStrategy::Fixed => {
        self.fixed(doc_id, input)
      }
    }
  }

  fn structured(
    &self,
    doc_id: &str,
    input: &str
  ) -> Vec<Chunk> {
    let mut cursor = 0;
    let mut results = Vec::new();
    let mut seen =
      if self.dedupe_segments {
        Some(HashSet::new())
      } else {
        None
      };
    while cursor < input.len() {
      let remaining = &input[cursor..];
      let (split_len, sep_len) = if self
        .split_on_double_newline
      {
        find_split_length(
          remaining,
          &self.chunk_separators
        )
      } else {
        (remaining.len(), 0)
      };
      let segment = &input
        [cursor..cursor + split_len];
      if let Some(chunk) = self.segment(
        cursor,
        doc_id,
        segment,
        ChunkStrategy::Structured,
        seen.as_mut()
      ) {
        results.push(chunk);
      }
      if split_len == remaining.len() {
        break;
      }
      cursor += split_len;
      cursor += sep_len;
      cursor +=
        skip_newlines(&input[cursor..]);
    }
    results
  }

  fn fixed(
    &self,
    doc_id: &str,
    input: &str
  ) -> Vec<Chunk> {
    let tokens = token_positions(input);
    if tokens.is_empty() {
      return Vec::new();
    }
    let mut results = Vec::new();
    let mut seen =
      if self.dedupe_segments {
        Some(HashSet::new())
      } else {
        None
      };
    let step = (self
      .max_tokens
      .saturating_sub(self.overlap))
    .max(1);
    let mut cursor = 0;
    while cursor < tokens.len() {
      let end = (cursor
        + self.max_tokens)
        .min(tokens.len());
      let start_pos =
        tokens[cursor].start;
      let end_pos = tokens[end - 1].end;
      let candidate =
        &input[start_pos..end_pos];
      if let Some(chunk) = self.segment(
        start_pos,
        doc_id,
        candidate,
        ChunkStrategy::Fixed,
        seen.as_mut()
      ) {
        results.push(chunk);
      }
      if end == tokens.len() {
        break;
      }
      cursor = cursor + step;
    }
    results
  }

  fn segment(
    &self,
    absolute_start: usize,
    doc_id: &str,
    segment: &str,
    strategy: ChunkStrategy,
    seen: Option<&mut HashSet<String>>
  ) -> Option<Chunk> {
    let trimmed = segment.trim();
    if trimmed.is_empty() {
      return None;
    }
    if let Some(seen) = seen {
      if !seen
        .insert(trimmed.to_string())
      {
        return None;
      }
    }
    let trimmed_start = segment.len()
      - segment.trim_start().len();
    let trimmed_end =
      segment.trim_end().len();
    let start =
      absolute_start + trimmed_start;
    let end =
      absolute_start + trimmed_end;
    Some(Chunk {
      id: Uuid::new_v4().to_string(),
      doc_id: doc_id.to_string(),
      text: normalization::normalize(
        trimmed
      ),
      start,
      end,
      strategy
    })
  }
}

fn skip_newlines(
  remainder: &str
) -> usize {
  remainder
    .chars()
    .take_while(|c| {
      c.is_ascii_whitespace()
    })
    .map(|c| c.len_utf8())
    .sum()
}

fn find_split_length(
  remaining: &str,
  separators: &[String]
) -> (usize, usize) {
  let mut best = None;
  for sep in separators {
    if sep.is_empty() {
      continue;
    }
    if let Some(idx) =
      remaining.find(sep)
    {
      match best {
        | Some((best_idx, _))
          if best_idx <= idx => {}
        | _ => {
          best = Some((idx, sep.len()))
        }
      }
    }
  }
  best.unwrap_or((remaining.len(), 0))
}

struct TokenBoundary {
  start: usize,
  end:   usize
}

fn token_positions(
  input: &str
) -> Vec<TokenBoundary> {
  let mut positions = Vec::new();
  let mut start = None;
  for (idx, ch) in input.char_indices()
  {
    if ch.is_whitespace() {
      if let Some(s) = start.take() {
        positions.push(TokenBoundary {
          start: s,
          end:   idx
        });
      }
    } else if start.is_none() {
      start = Some(idx);
    }
  }
  if let Some(s) = start {
    positions.push(TokenBoundary {
      start: s,
      end:   input.len()
    });
  }
  positions
}
