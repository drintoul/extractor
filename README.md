# extractor

Deterministic website → RAG corpus pipeline. LangGraph-orchestrated,
self-hosted Firecrawl for scraping, Ollama (GPU) for classification and
embeddings.

## Stack

| Service | Image | Role |
|---|---|---|
| `ollama` | `ollama/ollama` | qwen2.5:14b (Q4_K_M) classify + bge-large-en-v1.5 embed, GPU |
| `ollama-pull` | `ollama/ollama` | one-shot model puller (~9 GB LLM + 671 MB embed) |
| `firecrawl-api` | `ghcr.io/firecrawl/firecrawl` | map / scrape / crawl / extract API (internal only) |
| `firecrawl-playwright` | `ghcr.io/firecrawl/playwright-service` | JS rendering |
| `redis` / `rabbitmq` / `nuq-postgres` | official | Firecrawl queue backend |
| `mongo` | `mongo:7` | corpus store (internal only) |
| `app` | `./app` | LangGraph pipeline + web UI on :8080 |

## Prerequisites

- Docker + `nvidia-container-toolkit` (for the `deploy` GPU reservation on `ollama`)
- ~24 GB VRAM free; ~12 GB disk for models

## Run

```bash
cp .env.example .env          # optional; defaults work
docker compose up -d          # starts infra, pulls models (slow first time)
docker compose logs -f ollama-pull   # watch model pull progress
```

Then open the UI at **http://localhost:8080** — paste a seed URL, hit Run,
and watch job status + per-page keep/drop results.

CLI equivalent (same graph, same checkpoints):

```bash
docker compose run --rm --service-ports=false app run https://docs.example.com
```

Output lands in `./data/` and MongoDB:

- `corpus-<run_id>.jsonl` — one chunk per line: `chunk_id`, `doc_id`,
  `source_url`, `title`, `section`, `text`, `embedding` (1024-dim, bge-large)
- `manifest-<run_id>.json` — counts, model pins, seed, status breakdown
- `extractor.docs` — `{_id: doc_id, url, title, markdown, run_ids[]}`
- `extractor.chunks` — `{_id: chunk_id, doc_id, source_url, title, section,
  text, embedding, run_id}`
- `extractor.pages` — per-URL audit: `{url, status, reason, word_count}`
  (query this to see why a page was dropped)
- `extractor.runs` — `{_id: run_id, ...manifest}`

All writes are upserts on content-hash IDs, so re-runs are idempotent.
Mongo is internal-only — query it via the container:
`docker compose exec mongo mongosh 'mongodb://$MONGO_INITDB_ROOT_USERNAME:$MONGO_INITDB_ROOT_PASSWORD@localhost:27017/extractor?authSource=admin'`

## Networking

All services share a single private bridge network (`extractor-internal`) with
outbound internet access. The **only** host-published port is the app UI on
`${APP_PORT:-8080}` → container `:8000`. Firecrawl, Ollama, Mongo, Redis,
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
flowchart LR
    S([START]) --> map["map<br/>Firecrawl /v1/map"]
    map --> triage["triage<br/>regex URL filter + normalize"]
    triage -- "Send × N (parallel, checkpointed per page)" --> pp
    triage -- "no URLs" --> dedupe

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
  extensions, tracking params — before any paid scrape
- `process_page`: `onlyMainContent` scrape → rule-based denoise → thin-page
  floor (`MIN_WORDS`) → Qwen keep/drop verdict (judge, never rewriter)
- `dedupe`: MinHash+LSH at 0.85 similarity
- `chunk_embed`: heading-aware ~600-word chunks with overlap; embeddings via
  Ollama `/api/embed`
- `persist`: idempotent upserts into `docs`, `chunks`, `pages`, `runs`

## Using /extract

Firecrawl's env points at Ollama (`OLLAMA_BASE_URL`, `MODEL_NAME`), so
`POST /v1/extract` on the internal `firecrawl-api:3002` runs on your local GPU
for structured
entities — keep it out of the bulk corpus path; use it for schema'd records
(FAQs, directories, product specs).

## Notes

- No API key needed: `USE_DB_AUTHENTICATION=false` (self-hosted mode). The
  Firecrawl API is internal-only; nothing publishes it to the host.
- `BLOCK_MEDIA=true` on the Playwright service saves bandwidth; flip it off if
  a target site needs images to render content.
