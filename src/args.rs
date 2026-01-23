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
    /// Emit a CSV tally of normalized
    /// word counts to the artifact dir
    #[arg(long)]
    emit_word_tally: bool,
    /// Emit the fully normalized text
    /// to the artifact dir
    #[arg(long)]
    emit_normalized: bool
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
  Status,

  /// Run the Stage 2 evaluation harness
  Evaluate,

  /// Train a Stage 4 custom embedder
  Train {
    /// Name of the model to generate
    model:   String,
    /// Optional explicit version for
    /// the model
    #[arg(long)]
    version: Option<String>,
    /// Optional override for the
    /// chunks source
    #[arg(long)]
    chunks:  Option<PathBuf>
  },

  /// Run the Stage 3 RAG workflow
  Rag {
    /// Query text
    query: String,
    /// Limit on retrieval hits
    /// (default per stage1 search)
    #[arg(long)]
    top_k: Option<usize>
  }
}
