mod args;
mod chunk;
mod config;
mod embedder;
mod evaluation;
mod index;
mod normalization;
mod pipeline;
mod search;
mod stage3;
mod state;

use anyhow::Result;
use clap::Parser;

use crate::args::Cli;
use crate::config::Config;

fn main() -> Result<()> {
  let config =
    Config::load("oxbed-config.toml")
      .unwrap_or_default();
  let cli = Cli::parse();
  pipeline::run(cli.command, config)
}
