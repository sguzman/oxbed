mod args;
mod chunk;
mod embedder;
mod index;
mod normalization;
mod pipeline;
mod state;

use anyhow::Result;
use clap::Parser;

use crate::args::Cli;

fn main() -> Result<()> {
  let cli = Cli::parse();
  pipeline::run(cli.command)
}
