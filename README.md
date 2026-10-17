# extractor

Deterministic website → RAG corpus pipeline. Give it a seed URL and it
produces chunked, embedded, metadata-rich JSON documents — stored in MongoDB
and as JSONL — ready to load into a vector store for a chatbot.

Everything runs locally in Docker:

- **Discovery + scraping** — self-hosted [Firecrawl](https://github.com/firecrawl/firecrawl)
  (`/v1/map` for URL inventory, `/v1/scrape` with `onlyMainContent` per page)
- **Classification + embeddings** — Ollama on your GPU: Qwen 2.5 14B (Q4_K_M)
  judges keep/drop per page; BGE-large-en-v1.5 embeds chunks. No external API
  calls, no keys.
- **Orchestration** — LangGraph with SQLite checkpointing: deterministic
  topology, resumable runs, content-hash IDs, idempotent Mongo upserts.
- **Storage** — MongoDB holds docs, chunks (with embeddings), a per-URL audit
  trail, and run manifests.

## Stack

| Service | Image | Role |
|---|---|---|
| `ollama` | `ollama/ollama` | qwen2.5:14b (Q4_K_M) classify + bge-large-en-v1.5 embed, GPU |
| `ollama-pull` | `ollama/ollama` | one-shot model puller (~9 GB LLM + 671 MB embed) |
| `firecrawl-api` | `ghcr.io/firecrawl/firecrawl` | map / scrape / crawl / extract API (internal only) |
| `firecrawl-playwright` | `ghcr.io/firecrawl/playwright-service` | JS rendering (stock, headless) |
| `stealth-playwright` | local build `./stealth-playwright` | JS rendering (real headed Chrome + patchright under Xvfb — slower, defeats some anti-bot walls) |
| `redis` / `rabbitmq` / `nuq-postgres` | official | Firecrawl queue backend |
| `mongo` | `mongo:7` | corpus store (internal only) |
| `app` | `./app` | LangGraph pipeline + web UI on :8090 |

## Prerequisites

- Docker + `nvidia-container-toolkit` (for the `deploy` GPU reservation on `ollama`)
- **NVIDIA GPU with 24 GB VRAM** — the defaults are sized for this:
  Qwen 2.5 14B Q4_K_M (~9 GB) + bge-large (~0.7 GB) stay resident
  (`OLLAMA_MAX_LOADED_MODELS=2`), leaving headroom for parallel classify
  calls (`OLLAMA_NUM_PARALLEL=4`) and KV-cache context.
  Smaller GPU? Lower `LLM_MODEL` in `.env` (e.g. `qwen2.5:7b-instruct-q4_K_M`).
- ~12 GB disk for the models, plus a few GB for images

## Run

```bash
cp .env.example .env          # optional; defaults work
docker compose up -d          # starts infra, pulls models (slow first time)
docker compose logs -f ollama-pull   # watch model pull progress
```

Then open the UI at **http://localhost:8090** — paste a seed URL, hit Run,
and watch job status + per-page keep/drop results.

CLI equivalent (same graph, same checkpoints):

```bash
docker compose run --rm --service-ports=false app run https://docs.example.com
```

## Configuration (.env)

All settings live in `.env` (copy `.env.example`). docker-compose.yaml has no
fallbacks — a missing key fails fast.

| Variable | Purpose |
|---|---|
| `PLAYWRIGHT_MICROSERVICE_URL` | JS renderer Firecrawl calls — `http://firecrawl-playwright:3000/scrape` (stock) or `http://stealth-playwright:3000/scrape` (slower, real Chrome, better vs. bot walls) |
| `LLM_MODEL` | Ollama tag for keep/drop classification + Firecrawl `/extract` |
| `EMBED_MODEL` | Ollama embedding tag (`bge-large` = bge-large-en-v1.5, 1024-dim) |
| `LLM_SEED` | Fixed seed; combined with `temperature=0`/`top_k=1` for reproducible verdicts |
| `POSTGRES_*` | Credentials for Firecrawl's internal `nuq-postgres` queue DB |
| `NUM_WORKERS_PER_QUEUE` / `CRAWL_CONCURRENT_REQUESTS` / `MAX_CONCURRENT_JOBS` | Firecrawl worker concurrency |
| `MONGO_USER` / `MONGO_PASSWORD` / `MONGO_DB` | Corpus store credentials + database name |
| `APP_PORT` | Host port for the web UI/API — the only published port (default `8090`) |
| `MIN_WORDS` | Word-count floor before a page reaches the LLM classifier |
| `SITEMAP_SKIP_PATTERNS` | Substrings — child sitemaps matching one are skipped in sitemap-fallback discovery (drops whole product catalogs). Skipped sitemaps are still counted: if catalog URLs outnumber content pages, the run card flags `catalog_heavy` — a fit signal for RAG-chatbot prospecting |
| `EXCLUDE_PATH_REGEX` | Optional regex rejected at triage as `user-excluded` (e.g. `/(products?)(/|$)` for Shopify/Woo-style paths) |

## UI & API

The `app` service exposes a web UI and JSON API on `APP_PORT`:

| Endpoint | Purpose |
|---|---|
| `GET /` | Web UI — run form, job status, run history, page audit, chunk previews |
| `POST /api/runs` | Start a run: `{"url": "https://…", "run_id": "optional"}` |
| `GET /api/jobs` | In-flight run status + live stage progress (this process) |
| `POST /api/jobs/{run_id}/kill` | Cooperative cancel — aborts at next node; partial work stays checkpointed |
| `POST /api/runs/{run_id}/rerun` | Fresh execution under a new hash-suffixed run_id (re-scrapes) |
| `DELETE /api/runs/{run_id}` | Delete run: mongo rows, checkpoints, JSONL/manifest files |
| `GET /api/runs` | Completed run manifests (from `extractor.runs`) |
| `GET /api/pages?run_id=…&status=…` | Per-URL audit: status, reason, word count |
| `GET /api/docs?run_id=…` | Kept documents for a run |
| `GET /api/chunks?run_id=…[&doc_id=…]` | Chunk text/metadata (embeddings excluded) |
| `POST /api/runs/{run_id}/keep?url=…` | Manual keep: chunks+embeds the page into the corpus |
| `POST /api/runs/{run_id}/remove?url=…` | Manual remove: pulls the page's chunks out of the corpus |
| `GET /api/runs/{run_id}/overrides` | List stored keep/remove curation for the run's site |
| `DELETE /api/runs/{run_id}/overrides` | Forget all site curation — takes effect on the next run |
| `GET /api/runs/{run_id}/export.pkl` / `.jsonl` | Self-describing corpus export (metadata + chunks) |

Resubmitting an existing `run_id` resumes its checkpoint instead of restarting —
that's also how a cancelled/failed job's **Resume** button works. **Kill** is
cooperative: the graph aborts at the next node boundary, completed pages stay
checkpointed, and resuming continues where it left off. **Re-run** on a
finished run creates a fresh checkpoint under a new hash-suffixed run_id
(full re-scrape).

Output lands in `./data/` and MongoDB:

- `corpus-<run_id>.jsonl` — one chunk per line: `chunk_id`, `doc_id`,
  `source_url`, `title`, `section`, `text`, `embedding` (1024-dim, bge-large),
  `archived`
- `manifest-<run_id>.json` — counts, model pins, seed, status breakdown,
  `settings` snapshot (`min_words`, `map_limit`, `scrape_timeout_ms`,
  `sitemap_skip_patterns`, `exclude_path_regex`)
- `extractor.docs` — `{_id: doc_id, url, title, markdown, archived, run_ids[]}`
- `extractor.chunks` — `{_id: chunk_id, doc_id, source_url, title, section,
  text, embedding, archived, run_id}`
- `extractor.pages` — per-URL audit: `{url, status, reason, word_count}`
  (query this to see why a page was dropped)
- `extractor.runs` — `{_id: run_id, ...manifest}`
- `extractor.overrides` — manual curation: `{seed_url, url, action, run_id,
  at}` — keep/remove selections recorded per site and re-applied at persist
  time on re-runs (the classifier verdict never overrides your judgment)

**Exports** (`pkl`/`jsonl` links on each run card, or
`GET /api/runs/<id>/export.pkl|jsonl`) are self-describing:

```python
corpus = pickle.load(open("corpus-<run>.pkl", "rb"))
# corpus = {"run_id", "seed_url", "models": {llm, embed},
#           "settings": {min_words, ...}, "counts": {...},
#           "overrides": [{url, action, at}], "chunks": [...]}
for c in corpus["chunks"]:  # chunk_id, doc_id, source_url, title,
    ...                     # section, text, embedding (1024-dim), archived
```

The JSONL export is the same shape as a header line: first line is
`{"_meta": {run_id, seed_url, models, settings, counts, overrides}}`,
remaining lines are chunks — skip `line["_meta"]` when streaming.

`archived: true` is set on docs/chunks whose URL path contains `archive`
(e.g. `/beer-archive/<product>`) — historical content that could read as
currently available. Archived chunks also carry an in-text note
(`[archived content — may no longer be current or available]`) so a RAG
layer hedges availability claims even if it ignores the flag. Filter or
qualify on `archived` in your retriever.

All writes are upserts on content-hash IDs, so re-runs are idempotent.
Mongo is internal-only — query it via the container:
`docker compose exec mongo mongosh 'mongodb://$MONGO_INITDB_ROOT_USERNAME:$MONGO_INITDB_ROOT_PASSWORD@localhost:27017/extractor?authSource=admin'`

## Networking

All services share a single private bridge network (`extractor-internal`) with
outbound internet access. The **only** host-published port is the app UI on
`${APP_PORT}` → container `:8000`. Firecrawl, Ollama, Mongo, Redis,
RabbitMQ, and Postgres are unreachable from the host — all inter-container
traffic (firecrawl-api:3002, ollama:11434, mongo:27017, …) stays on the
internal network. Use `docker compose exec` for debugging access.

## Determinism & resume

- `thread_id = run_id` → SQLite checkpoints in `./checkpoints/corpus.db`.
  Re-running the same `--run-id` resumes without re-scraping.
- URL inventory is sorted at every stage; fan-out results merge into a dict
  keyed by URL, so parallel order never affects output.
- All Ollama chat calls use `temperature=0`, fixed `seed`, `top_k=1`, JSON
  mode. Parse failures fall back to `review` status (non-destructive).
- `doc_id = sha256(markdown)` — same content → same ID across runs.
- The website itself is the only mutable input; raw scrape results are pinned
  in the checkpoint, so everything downstream of `process_page` is replayable.

## Pipeline

```mermaid
flowchart TD
    S([START]) --> map["map<br/>Firecrawl /v1/map"]
    map --> triage["triage<br/>regex URL filter + normalize"]
    triage --> dispatch["dispatch<br/>advance batch cursor"]
    dispatch -- "Send × ≤BATCH_SIZE<br/>(default 200)" --> pp
    pp --> dispatch
    dispatch -- "all batches done" --> dedupe

    subgraph pp["process_page — one instance per URL"]
        scrape["/v1/scrape<br/>onlyMainContent"] --> clean["rule-based denoise"]
        clean --> thin{"words ≥ MIN_WORDS?"}
        thin -- "no" --> drop(["dropped_thin"])
        thin -- "yes" --> llm{"Qwen 14B keep/drop<br/>temp=0 · seed · JSON"}
        llm -- "no" --> drop2(["dropped_llm"])
        llm -- "parse fail" --> review(["review"])
        llm -- "yes" --> keep(["kept<br/>doc_id = sha256(md)"])
    end

    pp --> dedupe["dedupe<br/>MinHash + LSH @ 0.85"]
    dedupe --> ce["chunk_embed<br/>heading-aware ~600-word chunks<br/>bge-large embeddings"]
    ce --> persist["persist<br/>Mongo upserts + corpus.jsonl + manifest"]
    persist --> E([END])
```

- `triage`: drops tag/author/category/pagination/cart/login paths, binary
  extensions, tracking params — before any paid scrape. Also **locale-folds**:
  URLs differing only by a leading locale segment (`/en/…`, `/usa/…`, `/gbr/…`)
  collapse to one representative (naked path preferred, then English locales),
  since per-locale copies defeat content dedupe — rejected as `locale-fold`,
  counted in the manifest as `locale_folded`
- `process_page`: `onlyMainContent` scrape → rule-based denoise → thin-page
  floor (`MIN_WORDS`) → Qwen keep/drop verdict (judge, never rewriter)
- `dedupe`: MinHash+LSH at 0.85 similarity
- `chunk_embed`: heading-aware ~600-word chunks with overlap; embeddings via
  Ollama `/api/embed`
- `persist`: idempotent upserts into `docs`, `chunks`, `pages`, `runs`

### Page statuses

Every processed URL lands in `extractor.pages` with one of:

| Status | Meaning |
|---|---|
| `kept` | Passed all filters + LLM verdict; chunked and embedded |
| `review` | LLM output failed to parse — kept for manual inspection |
| `dropped_thin` | Under `MIN_WORDS` after denoising (stubs, nav pages) |
| `dropped_llm` | Qwen judged it non-informational (reason is recorded) |
| `scrape_error` | Firecrawl failed to fetch/render the page |
| `rejected` | Never scraped — killed by a `triage.py` URL rule; `reason` says which |

## Using /extract

Firecrawl's env points at Ollama (`OLLAMA_BASE_URL`, `MODEL_NAME`), so
`POST /v1/extract` on the internal `firecrawl-api:3002` runs on your local GPU
for structured
entities — keep it out of the bulk corpus path; use it for schema'd records
(FAQs, directories, product specs).

## Migrating to a VPS

Everything is containerized and self-contained — copy the repo (compose, `app/`,
`.env`) plus the data you want to keep:

```bash
# Backup ollama models (~10 GB) and mongo corpus
docker run --rm -v extractor_ollama-models:/v -v $(pwd):/backup alpine \
  tar czf /backup/ollama-models.tar.gz -C /v .
docker run --rm -v extractor_mongo-data:/v -v $(pwd):/backup alpine \
  tar czf /backup/mongo-data.tar.gz -C /v .
# scp *.tar.gz + repo to the VPS, restore into same-named volumes, then:
docker compose up -d --build
```

- **No GPU on the VPS?** Remove the `deploy` block on `ollama` (CPU fallback) and
  consider `LLM_MODEL=qwen2.5:7b` — 14B classification is very slow on CPU.
- Skip the model archive if the VPS has good bandwidth — `ollama-pull`
  re-downloads on first boot.
- Checkpoints (`./checkpoints/corpus.db`) and output (`./data/`) are plain
  files — copy them to preserve run history and resume capability.

## Troubleshooting

- **UI port conflict** — the host already runs many services; if `APP_PORT` is
  taken the app container can start with no bound port (or a different server
  answers). Pick a free port in `.env` and `docker compose up -d app`.
- **`mongo` unhealthy** — the healthcheck must use `-u`/`-p` flags; putting
  credentials in a single-quoted URI inside `test:` prevents `$VAR` expansion.
- **First boot is slow** — `ollama-pull` downloads ~10 GB of models before
  `app` starts (`service_completed_successfully`). Watch:
  `docker compose logs -f ollama-pull`.
- **Everything dropped** — check `extractor.pages`: `dropped_thin` → lower
  `MIN_WORDS`; `dropped_llm` → the classify prompt in `nodes.py`
  (`_CLASSIFY_SYSTEM`) may be too strict for business sites (menus, events).
- **Resume a failed/interrupted run** — resubmit the same `run_id`; the
  SQLite checkpoint replays completed nodes and only re-runs the rest.

## Notes

- No API key needed: `USE_DB_AUTHENTICATION=false` (self-hosted mode). The
  Firecrawl API is internal-only; nothing publishes it to the host.
- `BLOCK_MEDIA=true` on the Playwright service saves bandwidth; flip it off if
  a target site needs images to render content.
- `stealth-playwright` is a drop-in replacement for the stock renderer: real
  headed Google Chrome (not Playwright's Chromium) patched by `patchright`,
  running under Xvfb. Point `PLAYWRIGHT_MICROSERVICE_URL` at it when a site
  bot-walls the stock service (verified on royalcaribbean.com). It is slower
  and heavier per page — it is a mitigation, not a guarantee; WAFs can still
  block on IP reputation, TLS fingerprint, or behavioral signals, which need
  residential proxies or site-specific handling.
