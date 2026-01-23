use std::fs::{
  self,
  File
};
use std::io::Write;
use std::path::PathBuf;
use std::time::{
  Duration,
  Instant
};

use anyhow::{
  Context,
  Result
};
use chrono::Utc;
use serde::Serialize;

use crate::config::{
  Config,
  EvaluationQuery
};
use crate::embedder::build_embedder;
use crate::index::VectorIndex;
use crate::search::search_hits;
use crate::state::State;

pub fn run_evaluation(
  config: &Config,
  state: &State,
  index: &VectorIndex
) -> Result<()> {
  if !config.stage2.enabled {
    println!(
      "Stage 2 evaluation is disabled."
    );
    return Ok(());
  }
  let queries =
    &config.stage2.evaluation.queries;
  if queries.is_empty() {
    println!(
      "No evaluation queries \
       configured."
    );
    return Ok(());
  }
  if state.index_entries.is_empty() {
    println!(
      "No indexed chunks yet. Run \
       `oxbed ingest` before \
       evaluating."
    );
    return Ok(());
  }
  for kind in
    &config.stage2.embedder_kinds
  {
    let embedder = build_embedder(
      kind.clone(),
      config
    )?;
    let embedder_name = embedder.name();
    let mut query_reports = Vec::new();
    let mut latencies = Vec::new();
    for query in queries {
      let resolved_top_k =
        query.top_k.unwrap_or(
          config.stage1.search.top_k
        );
      let start = Instant::now();
      let hits = search_hits(
        embedder.as_ref(),
        &query.query,
        resolved_top_k,
        config,
        state,
        index
      )?;
      let duration = start.elapsed();
      let mut report = evaluate_query(
        query,
        &hits,
        resolved_top_k
      );
      report.latency_ms =
        duration.as_secs_f32() * 1000.0;
      query_reports.push(report);
      latencies.push(duration);
    }
    let aggregated = aggregate_metrics(
      &query_reports,
      &latencies,
      index.entries().len()
    );
    if config.stage2.log_evaluation {
      let run_path = persist_run(
        config,
        &embedder_name,
        &aggregated,
        &query_reports
      )?;
      println!(
        "Logged evaluation run to {}",
        run_path.display()
      );
    }
    println!(
      "Evaluation {} → recall={:.3}, \
       mrr={:.3}, nDCG={:.3}, \
       latency={:.1}ms, index={} \
       entries",
      embedder_name,
      aggregated.recall,
      aggregated.mrr,
      aggregated.ndcg,
      aggregated.avg_latency_ms,
      aggregated.index_size
    );
  }
  Ok(())
}

#[derive(Clone, Serialize)]
struct AggregatedMetrics {
  recall:         f32,
  mrr:            f32,
  ndcg:           f32,
  avg_latency_ms: f32,
  index_size:     usize
}

#[derive(Clone, Serialize)]
struct QueryReport {
  name:       String,
  top_k:      usize,
  recall:     f32,
  mrr:        f32,
  ndcg:       f32,
  hits:       usize,
  expected:   usize,
  latency_ms: f32
}

#[derive(Serialize)]
struct EvaluationRun {
  timestamp: String,
  embedder:  String,
  metrics:   AggregatedMetrics,
  queries:   Vec<QueryReport>
}

fn evaluate_query(
  query: &EvaluationQuery,
  hits: &[crate::search::SearchHit],
  top_k: usize
) -> QueryReport {
  let normalized_terms: Vec<_> = query
    .expected_terms
    .iter()
    .map(|term| term.to_lowercase())
    .collect();
  let expected_count =
    normalized_terms.len();
  let mut satisfied =
    vec![false; expected_count];
  let mut relevance_flags = Vec::new();
  let mut first_relevant_rank = None;
  for (rank, hit) in
    hits.iter().enumerate()
  {
    let chunk_text =
      hit.chunk.text.to_lowercase();
    let mut relevant = false;
    for (idx, term) in normalized_terms
      .iter()
      .enumerate()
    {
      if !satisfied[idx]
        && chunk_text.contains(term)
      {
        satisfied[idx] = true;
        relevant = true;
      }
    }
    if relevant
      && first_relevant_rank.is_none()
    {
      first_relevant_rank =
        Some(rank + 1);
    }
    relevance_flags.push(relevant);
  }
  let matched = satisfied
    .iter()
    .filter(|&&v| v)
    .count();
  let recall = if expected_count == 0 {
    0.0
  } else {
    matched as f32
      / expected_count as f32
  };
  let mrr = first_relevant_rank
    .map(|rank| 1.0 / rank as f32)
    .unwrap_or(0.0);
  let ndcg = compute_ndcg(
    &relevance_flags,
    matched
  );
  QueryReport {
    name: query.name.clone(),
    top_k,
    recall,
    mrr,
    ndcg,
    hits: hits.len(),
    expected: expected_count,
    latency_ms: 0.0
  }
}

fn compute_ndcg(
  flags: &[bool],
  relevant: usize
) -> f32 {
  if relevant == 0 {
    return 0.0;
  }
  let actual: f32 = flags
    .iter()
    .enumerate()
    .map(|(idx, &flag)| {
      if !flag {
        return 0.0;
      }
      let rank = idx + 1;
      1.0 / (rank as f32 + 1.0).log2()
    })
    .sum();
  let ideal: f32 = (0..relevant)
    .map(|idx| {
      let rank = idx + 1;
      1.0 / (rank as f32 + 1.0).log2()
    })
    .sum();
  if ideal == 0.0 {
    0.0
  } else {
    actual / ideal
  }
}

fn aggregate_metrics(
  reports: &[QueryReport],
  latencies: &[Duration],
  index_size: usize
) -> AggregatedMetrics {
  if reports.is_empty()
    || latencies.is_empty()
  {
    return AggregatedMetrics {
      recall: 0.0,
      mrr: 0.0,
      ndcg: 0.0,
      avg_latency_ms: 0.0,
      index_size
    };
  }
  let total = reports.len() as f32;
  let recall = reports
    .iter()
    .map(|r| r.recall)
    .sum::<f32>()
    / total;
  let mrr = reports
    .iter()
    .map(|r| r.mrr)
    .sum::<f32>()
    / total;
  let ndcg = reports
    .iter()
    .map(|r| r.ndcg)
    .sum::<f32>()
    / total;
  let avg_latency_ms = latencies
    .iter()
    .map(|duration| {
      duration.as_secs_f32() * 1000.0
    })
    .sum::<f32>()
    / total;
  AggregatedMetrics {
    recall,
    mrr,
    ndcg,
    avg_latency_ms,
    index_size
  }
}

fn persist_run(
  config: &Config,
  embedder_name: &str,
  metrics: &AggregatedMetrics,
  queries: &[QueryReport]
) -> Result<PathBuf> {
  let timestamp = Utc::now();
  let date_dir = PathBuf::from(
    &config.stage2.runs_dir
  )
  .join(
    timestamp
      .format("%Y-%m-%d")
      .to_string()
  );
  fs::create_dir_all(&date_dir)
    .with_context(|| {
      format!(
        "create run directory {:?}",
        date_dir
      )
    })?;
  let filename = format!(
    "run-{}-{}.json",
    timestamp.format("%Y%m%dT%H%M%SZ"),
    embedder_name
  );
  let path = date_dir
    .join(filename.replace('/', "-"));
  let run = EvaluationRun {
    timestamp: timestamp.to_rfc3339(),
    embedder:  embedder_name
      .to_string(),
    metrics:   metrics.clone(),
    queries:   queries.to_vec()
  };
  let mut file = File::create(&path)
    .with_context(|| {
      format!(
        "create run file {:?}",
        path
      )
    })?;
  serde_json::to_writer_pretty(
    &mut file, &run
  )
  .with_context(|| {
    format!("write run file {:?}", path)
  })?;
  writeln!(file)?;
  Ok(path)
}
