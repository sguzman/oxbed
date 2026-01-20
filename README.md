# Oxbed

![Oxbed primary lockup](branding/oxbed-primary-lockup.png)

Oxbed is a text analysis + embedding research platform that begins with a clean, layer-by-layer pipeline and scales toward knowledge extraction, custom embedding training, and RAG workflows. The repository stores the foundations of a “text swiss-army knife”: metadata-aware ingestion, configurable chunking, vector indexing, search, and instrumentation that lets you evaluate and swap embedding models without revamping the surrounding infrastructure.

## Ambition

The project’s ambition is to move from a “desert of text tools” to a cohesive system that supports:

- fast ingestion/normalization of `.txt`/`.md` corpora,
- fine-grained chunking plus provenance-friendly metadata,
- reusable embeddings/vector stores with simple search pipelines,
- evaluation harnesses, reranking, and LLM layers,
- and, eventually, custom-trained embedders, knowledge extraction, and polished product features.

Complete Stage 1 (the core pipeline) to reach **v1.0.0**, then follow each subsequent stage as the next major version, with Stage 4 culminating in **v4.0.0**; see `ROADMAP.md` for the full breakdown of stages, goals, and release criteria.

## Getting started

1. **Understand the foundations** – `tmp/text.md` captures a detailed, multi-stage vision; Stage 1 focuses on ingestion, chunking, vector store design, and a simple query pipeline.
2. **Follow the roadmap** – `ROADMAP.md` documents each stage, the work items, and the definition of done, plus the version target for that stage.
3. **Build & test** – run `cargo build`, followed by the manual checklist in `docs/ai/POST-CHANGES.md` (formatting with `cargo fmt`, `taplo fmt`, `biome format --write .`, validation via `taplo validate`, link checks with `lychee`, spelling with `typos`, and unit tests via `cargo test`).

## Tooling & workflow

- `just build` – compiles the workspace.
- `just fmt`, `taplo fmt`, `biome format --write .` – format code, TOML, and JSON artifacts.
- `just validate` – runs `taplo validate`.
- `just typos` – runs the spell checker (`typos --config typos.toml`).
- `just links` – runs `lychee --config lychee.toml .`.
- `just test` – runs `cargo test`.

Release & contribution processes follow `docs/RELEASE.md` and `docs/ai/RUST.md`: use Conventional Commits, respect the SemVer policy, generate changelogs with git-cliff, and keep large Rust files modularized if they grow beyond 600 lines.

## Next steps

- Implement the ingestion + chunking + query path described in Stage 1 so the CLI and vector store can deliver meaningful search results.
- Once Stage 1 is stable (v1.0.0), iterate through the later stages documented in `ROADMAP.md`, using the evaluation harness, rerankers, and custom embedder workflows to advance the platform.
- After Stage 2, enable `stage3.enabled = true` and use `oxbed rag "<your question>"` to rerank hits, build context-limited prompts, and compare multiple reranking strategies before moving on to the LLM/RAG flows of Stage 3.
- Enable Stage 2 instrumentation by running `oxbed evaluate` (once `stage2.enabled = true` in `oxbed-config.toml`) so you can capture recall@k/MRR/nDCG/latency metrics and write run summaries under `runs/YYYY-MM-DD/`.

## Branding

- The full-color visual lockup (`branding/oxbed-primary-lockup.png`) sets the tone—use it at the top of documents and marketing materials to show the playful Rust crab quickly settling onto a bed.
- The monomark version (`branding/oxbed-monomark.png`) is a simplified abstract that works well in footers, badges, or any place where a smaller, monochrome graphic is preferable (for example, it appears at the bottom of this README).
- Text-based variants live in `branding/dev.txt` (two-line version), `branding/tight.txt` (exact textual logo without spacing), and `branding/compact.txt` (single-line form); embed them in terminal UIs, release notes, or README subsections that benefit from ASCII flair.

![Oxbed monomark](branding/oxbed-monomark.png)
