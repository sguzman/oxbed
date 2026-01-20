use std::cmp::Ordering;
use std::collections::HashSet;

use anyhow::Result;

use crate::config::{
  Config,
  Stage3RerankMode,
  Stage3RerankerStrategyConfig
};
use crate::embedder::Embedder;
use crate::index::VectorIndex;
use crate::search::{
  SearchHit,
  search_hits
};
use crate::state::State;

pub fn run_stage3(
  query: &str,
  top_k: usize,
  config: &Config,
  state: &State,
  index: &VectorIndex,
  embedder: &dyn Embedder
) -> Result<()> {
  if !config.stage3.enabled {
    println!(
      "Stage 3 is disabled in config."
    );
    return Ok(());
  }
  let hits = search_hits(
    embedder, query, top_k, config,
    state, index
  )?;
  if hits.is_empty() {
    println!(
      "No hits found for query."
    );
    return Ok(());
  }
  let deduped = dedupe_hits(hits);
  for strategy in
    &config.stage3.reranker.strategies
  {
    let reranked =
      rerank_hits(&deduped, strategy);
    if reranked.is_empty() {
      println!(
        "Strategy {} produced no \
         reranked hits.",
        strategy.name
      );
      continue;
    }
    println!(
      "=== Strategy: {} ===",
      strategy.name
    );
    for (rank, entry) in
      reranked.iter().enumerate()
    {
      println!(
        "Result {} [score: {:.3}] → {}",
        rank + 1,
        entry.score,
        entry
          .hit
          .chunk
          .text
          .lines()
          .next()
          .unwrap_or("")
          .trim()
      );
      println!(
        "  Document: {} [{}-{}/{}]",
        entry.hit.document.path,
        entry.hit.chunk.start,
        entry.hit.chunk.end,
        entry.hit.chunk.strategy
      );
    }
    let context = build_context(
      &reranked,
      config.stage3.context_budget
    );
    let prompt = format_prompt(
      &config.stage3.prompt_template,
      query,
      &context
    );
    println!("Prompt:\n{}", prompt);
  }
  Ok(())
}

fn dedupe_hits(
  hits: Vec<SearchHit>
) -> Vec<SearchHit> {
  let mut seen = HashSet::new();
  hits
    .into_iter()
    .filter(|hit| {
      let fingerprint =
        hit.chunk.text.to_lowercase();
      seen.insert(fingerprint)
    })
    .collect()
}

struct RerankedHit<'a> {
  hit:   &'a SearchHit,
  score: f32
}

fn rerank_hits<'a>(
  hits: &'a [SearchHit],
  strategy: &Stage3RerankerStrategyConfig
) -> Vec<RerankedHit<'a>> {
  let lower_boost: Vec<String> =
    strategy
      .boost_terms
      .iter()
      .map(|term| term.to_lowercase())
      .collect();
  let mut scored = Vec::new();
  for hit in hits {
    let base = hit.score;
    let term_score = lower_boost
      .iter()
      .filter(|term| {
        hit
          .chunk
          .text
          .to_lowercase()
          .contains(term.as_str())
      })
      .count()
      as f32;
    let boost = term_score
      * strategy.boost_factor;
    let total = match strategy.mode {
      Stage3RerankMode::None => base,
      Stage3RerankMode::TermOverlap => base + boost,
      Stage3RerankMode::Hybrid => {
        base * (1.0 - strategy.hybrid_weight)
          + boost * strategy.hybrid_weight
      }
    };
    if total >= strategy.threshold {
      scored.push(RerankedHit {
        hit,
        score: total
      });
    }
  }
  scored.sort_by(|a, b| {
    b.score
      .partial_cmp(&a.score)
      .unwrap_or(Ordering::Equal)
  });
  scored
}

fn build_context(
  hits: &[RerankedHit],
  budget: usize
) -> String {
  let mut context = String::new();
  for entry in hits {
    if context.len() >= budget {
      break;
    }
    let addition =
      entry.hit.chunk.text.trim();
    if addition.is_empty() {
      continue;
    }
    if !context.is_empty() {
      context.push_str("\n---\n");
    }
    context.push_str(&truncate(
      addition,
      budget - context.len()
    ));
  }
  context
}

fn truncate(
  text: &str,
  max: usize
) -> String {
  if max == 0 {
    return String::new();
  }
  text.chars().take(max).collect()
}

fn format_prompt(
  template: &str,
  query: &str,
  context: &str
) -> String {
  template
    .replace("{query}", query)
    .replace("{context}", context)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::chunk::{
    Chunk,
    ChunkStrategy
  };
  use crate::search::SearchHit;
  use crate::state::Document;

  #[test]
  fn build_context_respects_budget() {
    let chunk = Chunk {
      id:       "c".into(),
      doc_id:   "d".into(),
      text:     "alpha beta".into(),
      start:    0,
      end:      0,
      strategy:
        ChunkStrategy::Structured
    };
    let document = Document {
      id:          "d".into(),
      path:        "doc".into(),
      hash:        "h".into(),
      token_count: 0
    };
    let hit = SearchHit {
      chunk,
      document,
      score: 1.0
    };
    let hits = vec![RerankedHit {
      hit:   &hit,
      score: 1.0
    }];
    assert_eq!(
      build_context(&hits, 3),
      "alp"
    );
  }
}
