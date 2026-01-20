use std::path::PathBuf;

use clap::{
  Parser,
  Subcommand
};

use crate::chunk::ChunkStrategy;

#[derive(Debug, Parser)]
#[command(
  name = "oxbed",
  about = "Oxbed text + embedding \
           toolbox"
)]
pub struct Cli {
  #[command(subcommand)]
  pub command: Command
}

#[derive(Debug, Subcommand)]
pub enum Command {
  /// Ingest text/Markdown files into
  /// the local corpus plus vector index
  Ingest {
    /// Path to a file or directory to
    /// ingest
    path:            PathBuf,
    /// Chunking strategy to apply
    /// (default: structured)
    #[arg(long, default_value_t = ChunkStrategy::Structured)]
    strategy:        ChunkStrategy,
    /// Optional path to emit a CSV
    /// tally of normalized word counts
    #[arg(long)]
    emit_word_tally: Option<PathBuf>,
    /// Optional path to emit the fully
    /// normalized text
    #[arg(long)]
    emit_normalized: Option<PathBuf>
  },
  /// Search the corpus with a query
  /// string
  Search {
    /// Query text
    query: String,
    /// Number of results to return
    #[arg(long)]
    top_k: Option<usize>
  },
  /// Show corpus status (documents,
  /// chunks)
  Status
}
