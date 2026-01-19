# Roadmap

This project follows the “Layered Embedding Builder” vision captured in `tmp/text.md`. Complete Stage 1 to reach **v1.0.0**, then publish each subsequent stage as the next major release so that a full Stage 4 implementation corresponds to **v4.0.0**. (Stage 5 features are post‑v4 enhancements and do not have a dedicated major version target yet.)

## Stage 1 – Core Pipeline → v1.0.0
- **Goal**: Deliver the “boring” pipeline that can feed any embedder. This includes ingestion/normalization, chunking, vector storage, and the end-to-end query path (`tmp/text.md:36-74`).
- **Key work**:
  - Build an ingestion module that reads `.txt`/`.md`, normalizes text, deduplicates repeated paragraphs, and emits canonical JSONL chunks with metadata.
  - Implement at least two chunking strategies (fixed window + structure-aware packing) and store chunk offsets for provenance.
  - Define a `VectorIndex` interface with in-memory backend (cosine similarity, metadata filtering, incremental inserts, `add/search/persist`).
  - Wire a query pipeline that embeds a query, hits the index, optionally reranks, and returns snippets plus metadata.
- **Definition of done**: `tk`-style CLI can ingest, chunk, store metadata, and answer queries; `cargo build` succeeds; manual checks outlined in `docs/ai/RUST.md` and `docs/ai/POST-CHANGES.md` have been considered for subsequent verification.

## Stage 2 – Measurement + Instrumentation → v2.0.0
- **Goal**: Treat the system as a measurement machine before training custom models (`tmp/text.md:74-154`).
- **Key work**:
  - Add multiple embedder backends (local model adapters, optional API, and a TF-IDF baseline).
  - Build an evaluation harness (intrinsic/extrinsic metrics, recall@k, MRR, nDCG, latency, index size).
  - Record runs under `runs/YYYY-MM-DD/` and optionally visualize embeddings (PCA/UMAP) or clustering.
- **Definition of done**: Embedders are swappable, evaluation + metrics persist in versioned runs, and visualization / logging hooks are available for future stages.

## Stage 3 – LLM + RAG → v3.0.0
- **Goal**: Layer LLM-driven retrieval over the stage 2 groundwork (`tmp/text.md:154-200`).
- **Key work**:
  - Assemble context budgets, deduped chunk sets, and citation-aware prompts for RAG.
  - Add reranking diagnostics with comparisons (embedding-only, rerank, hybrid BM25+embedding).
  - Produce QA flows that cite retrieved chunks, track whether evidence exists, and surface reranking gains.
- **Definition of done**: A RAG pipeline can answer user questions with citations, diagnostics show retrieval quality, and rerankers can be toggled for impact studies.

## Stage 4 – Custom Embeddings → v4.0.0
- **Goal**: Train and integrate bespoke embedding models while leveraging the existing evaluation loop (`tmp/text.md:200-296`).
- **Key work**:
  - Pick an objective (Word2Vec/contrastive). Prepare data (raw tokens or positive pairs) and ship a deterministic training loop with logging, checkpoints, and manifest metadata.
  - Register model artifacts under `models/<name>/<version>/manifest.json` and report dimension/training data hash.
  - Plug custom embedders back into the harness to compare them against Stage 2 baselines (`runs/…`), so you can answer questions like “Did my new embedder beat baseline X?”
- **Definition of done**: Custom embedding training exists, artifacts are versioned, and evaluation harness compares them to previous models.

## Beyond v4.0.0 – Product Features (Stage 5)
- **Stretch goal**: Real product niceties such as incremental indexing, ingestion queues, caching, metadata filters, multi-index support, exports/imports, and a knowledge-extraction overlay that stores claims with provenance (`tmp/text.md:296-379`).
- **How to proceed**: Treat these as post-v4 improvements; each could eventually become their own major release once the scope justifies it.

## Release & Workflow Notes
- Use Conventional Commits and follow `docs/RELEASE.md`/`docs/ai/RUST.md` before suggesting a version bump.
- Manual post-change checks (formatting, validation, linting, spelling, tests) are documented in `docs/ai/POST-CHANGES.md`.
- Routine tooling is exposed via `just build`, `just fmt`, `just validate`, `just typos`, `just links`, and `just test` per `justfile`.
