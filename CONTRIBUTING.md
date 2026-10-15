# Contributing

Thanks for your interest! This is a personal project, but issues and PRs are
welcome.

## Dev Setup

```bash
git clone https://github.com/drintoul/extractor.git
cd extractor
cp .env.example .env          # fill in POSTGRES_PASSWORD + MONGO_PASSWORD
docker compose up -d --build  # first boot pulls ~10 GB of models
# UI at http://localhost:${APP_PORT}
```

Requires Docker + `nvidia-container-toolkit` (Ollama needs a GPU; without one
remove the `deploy` block and use a smaller `LLM_MODEL`).

## Project Layout

```
docker-compose.yaml     # the whole stack; env from .env only, no fallbacks
app/pipeline/
  graph.py              # LangGraph topology + SqliteSaver checkpointer
  nodes.py              # map / triage / process_page / dedupe / chunk_embed / persist
  state.py              # CorpusState schema + reducers
  clients.py            # raw REST clients for Firecrawl + Ollama
  triage.py, denoise.py # deterministic URL filter + markdown cleaning rules
  dedupe.py, chunking.py# MinHash/LSH + heading-aware chunker
  store.py              # Mongo upserts (docs/chunks/pages/runs)
  cancel.py             # cooperative cancellation registry
  ui.py                 # FastAPI UI + JSON API
  main.py               # CLI: `run <url>` / `serve`
```

## Conventions That Matter Here

- **Determinism is the point.** Graph topology and routing must stay pure
  functions of state — never let an LLM choose edges. LLM calls stay pinned:
  `temperature=0`, fixed `LLM_SEED`, `top_k=1`, JSON mode, schema-validated.
- **Sorted, keyed state.** Sort URL lists; merge fan-out results into dicts
  keyed by URL — never rely on completion order.
- **Checkpoint discipline.** Expensive operations (scrapes) and LLM outputs
  land in the SQLite checkpoint so resume never re-pays. Prefer small nodes.
- **`.env` is the single source of config.** No `${VAR:-default}` in
  docker-compose.yaml, no hardcoded secrets — ever.
- **Internal-only services.** No new host port bindings; the only published
  port is the UI. Inter-container traffic stays on `extractor-internal`.
- **Python style** — compact, idiomatic; validate at boundaries, don't
  over-guard internals.

## Testing a Change

```bash
python3 -m py_compile app/pipeline/*.py   # fast syntax check
docker compose up -d --build app          # rebuild + restart app
# then run a small site through the UI or:
docker compose run --rm app run https://example.com --run-id test-<name>
```

Check `extractor.pages` in Mongo for the audit trail of a test run.

## Pull Requests

- Keep PRs focused; describe *why*, not just *what*
- Include a note on how you verified (which site, expected counts)
- Don't bump image tags or add dependencies without a reason — this stack
  pins `latest` for upstream images deliberately, but new Python deps should
  be pinned with a lower bound in `requirements.txt`
