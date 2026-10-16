"""Minimal UI/API server for the pipeline. Serves a no-build single page and
JSON endpoints over Mongo-backed run data. Pipeline runs happen in background
threads with LangGraph checkpointing, so resubmitting a run-id resumes it."""
import json
import os
import pickle
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit

import httpx

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from . import cancel, store
from .config import SETTINGS
from .graph import build_checkpointer, build_graph, delete_checkpoints
from .store import get_db

app = FastAPI(title="extractor")
_graph = None
_jobs: dict[str, dict] = {}


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph(checkpointer=build_checkpointer())
    return _graph


def _default_run_id(url: str) -> str:
    import hashlib
    host = urlsplit(url).netloc.replace(".", "-")
    host = re.sub(r"^www-", "", host)  # www.sidekick.com -> sidekick-com-…
    return f"{host}-{hashlib.sha256(url.encode()).hexdigest()[:8]}"


def _canonical_url(url: str) -> str:
    """Resolve redirects — farmhousebrewing.com -> https://www.farmhousebrewing.com/
    so the stored seed_url and derived run_id reflect the site's canonical host."""
    try:
        r = httpx.get(url, timeout=15, follow_redirects=True)
        final = str(r.url)
        if final.startswith(("http://", "https://")):
            return final
    except Exception:
        pass
    return url


class RunRequest(BaseModel):
    url: str
    run_id: str | None = None


def _start(url: str, run_id: str) -> None:
    started = datetime.now(timezone.utc).isoformat()
    entry = {"started_at": started, "url": url}

    def work():
        _jobs[run_id] = {**entry, "status": "running"}
        try:
            _get_graph().invoke(
                {"seed_url": url, "run_id": run_id,
                 "pages": {}, "rejected_urls": {}},
                config={"configurable": {"thread_id": run_id}},
            )
            _jobs.pop(run_id, None)  # completed runs live in Runs, not Jobs
        except Exception as e:
            if cancel.is_cancelled(run_id) or isinstance(e, cancel.Cancelled):
                _jobs[run_id] = {**entry, "status": "cancelled"}
            else:
                _jobs[run_id] = {**entry, "status": "error", "error": str(e)}

    cancel.clear(run_id)
    threading.Thread(target=work, daemon=True).start()


@app.post("/api/runs")
def start_run(req: RunRequest):
    url = req.url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    url = _canonical_url(url)
    run_id = req.run_id or _default_run_id(url)
    if _jobs.get(run_id, {}).get("status") == "running":
        return {"run_id": run_id, "status": "already_running", "url": url}
    _start(url, run_id)
    return {"run_id": run_id, "status": "started", "url": url}


@app.post("/api/runs/{run_id}/rerun")
def rerun(run_id: str):
    """Fresh execution of a completed run — new run_id with a distinct hash."""
    import hashlib
    doc = get_db().runs.find_one({"_id": run_id})
    if not doc:
        raise HTTPException(404, "run not found")
    host = urlsplit(doc["seed_url"]).netloc.replace(".", "-")
    host = re.sub(r"^www-", "", host)
    salt = doc["seed_url"] + datetime.now(timezone.utc).isoformat()
    new_id = f"{host}-{hashlib.sha256(salt.encode()).hexdigest()[:8]}"
    _start(doc["seed_url"], new_id)
    return {"run_id": new_id, "status": "started"}


@app.post("/api/jobs/{run_id}/kill")
def kill(run_id: str):
    job = _jobs.get(run_id)
    if not job or job["status"] != "running":
        raise HTTPException(400, "job is not running")
    cancel.request(run_id)
    job["status"] = "cancelling"  # fan-out may take a while to unwind
    return {"run_id": run_id, "status": "cancelling"}


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    """Remove a run's full footprint: mongo rows, checkpoint, artifacts."""
    if _jobs.get(run_id, {}).get("status") in ("running", "cancelling"):
        raise HTTPException(409, "job is running — kill it first")
    db = get_db()
    removed = {
        "runs": db.runs.delete_one({"_id": run_id}).deleted_count,
        "pages": db.pages.delete_many({"run_id": run_id}).deleted_count,
        "chunks": db.chunks.delete_many({"run_id": run_id}).deleted_count,
    }
    # docs are shared across runs via content-hash IDs — pull the run_id and
    # drop docs left with no referencing run
    db.docs.update_many({}, {"$pull": {"run_ids": run_id}})
    removed["docs_orphaned"] = db.docs.delete_many(
        {"run_ids": {"$size": 0}}).deleted_count
    removed["checkpoints"] = delete_checkpoints(run_id)
    for name in (f"corpus-{run_id}.jsonl", f"manifest-{run_id}.json"):
        path = os.path.join(SETTINGS.data_dir, name)
        if os.path.exists(path):
            os.remove(path)
            removed["files"] = removed.get("files", 0) + 1
    _jobs.pop(run_id, None)
    return {"run_id": run_id, "deleted": removed}


def _progress(values: dict) -> dict:
    """Infer the current pipeline stage from checkpointed graph state."""
    total = len(values.get("filtered_urls") or [])
    processed = len(values.get("pages") or {})
    if values.get("manifest"):
        stage = "persisted"
    elif "docs" in values:
        stage = "deduping / chunking / embedding"
    elif total:
        stage = f"scraping & classifying {processed}/{total}"
    elif values.get("url_inventory"):
        stage = "filtering discovered URLs"
    else:
        stage = "discovering URLs (map)"
    return {"stage": stage, "processed": processed, "total": total}


def _writes_count(run_id: str) -> int | None:
    """Completed task writes in the LangGraph checkpoint — grows in real time
    during the fan-out, unlike `pages` which only appears on commit."""
    try:
        conn = sqlite3.connect(SETTINGS.checkpoint_db)
        for table in ("writes", "checkpoint_writes"):
            try:
                n = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE thread_id=?",
                    (run_id,)).fetchone()[0]
                conn.close()
                return n
            except sqlite3.OperationalError:
                continue
        conn.close()
    except Exception:
        pass
    return None


@app.get("/api/jobs")
def jobs():
    now = time.time()
    out = {}
    for run_id, job in _jobs.items():
        if job["status"] == "done":
            continue  # completed runs live in Runs, not Jobs
        entry = dict(job)
        if job["status"] in ("running", "cancelling"):
            try:
                values = _get_graph().get_state(
                    {"configurable": {"thread_id": run_id}}).values
                entry["progress"] = _progress(values)
            except Exception:
                pass
            w = _writes_count(run_id)
            if w is not None:
                if w != job.get("_w"):
                    job["_w"], job["_wt"] = w, now
                entry["writes"] = w
            entry["idle_s"] = int(now - job.get("_wt", now))
        for k in ("_w", "_wt"):
            entry.pop(k, None)
        out[run_id] = entry
    return out


@app.post("/api/runs/{run_id}/keep")
def keep_page(run_id: str, url: str = Query(...)):
    """Manually mark a dropped page as kept: pulls its markdown from the
    LangGraph checkpoint, chunks + embeds + persists it into the corpus."""
    if _jobs.get(run_id, {}).get("status") in ("running", "cancelling"):
        raise HTTPException(409, "run in progress — wait or kill first")
    try:
        values = _get_graph().get_state(
            {"configurable": {"thread_id": run_id}}).values
    except Exception:
        raise HTTPException(404, "no checkpoint for run")
    page = (values.get("pages") or {}).get(url)
    if not page:
        if url in (values.get("rejected_urls") or {}):
            raise HTTPException(
                400, "URL was rejected pre-scrape — no content to keep")
        raise HTTPException(404, "page not found in run state")
    md = page.get("markdown") or ""
    if len(md.split()) < 3:
        raise HTTPException(400, "no scraped content to keep")

    corpus_path = os.path.join(SETTINGS.data_dir, f"corpus-{run_id}.jsonl")
    inc = store.apply_keep(run_id, url, page.get("title"), md,
                           page["status"], corpus_path)
    get_db().runs.update_one({"_id": run_id}, {"$inc": inc})
    seed = values.get("seed_url")
    if seed:
        store.record_override(seed, run_id, url, "keep")
    return {"status": "kept", **inc}


@app.post("/api/runs/{run_id}/remove")
def remove_page(run_id: str, url: str = Query(...)):
    """Manually remove a kept page: drops its doc/chunks from the corpus,
    keeping shared docs that other runs still reference."""
    if _jobs.get(run_id, {}).get("status") in ("running", "cancelling"):
        raise HTTPException(409, "run in progress — wait or kill first")
    import hashlib
    db = get_db()
    pid = f"{run_id}:{hashlib.sha256(url.encode()).hexdigest()[:16]}"
    page = db.pages.find_one({"_id": pid})
    if not page or page["status"] != "kept" or not page.get("doc_id"):
        raise HTTPException(400, "only kept pages can be removed")

    corpus_path = os.path.join(SETTINGS.data_dir, f"corpus-{run_id}.jsonl")
    inc = store.apply_remove(run_id, url, page["doc_id"], "kept", corpus_path)
    db.runs.update_one({"_id": run_id}, {"$inc": inc})
    run = db.runs.find_one({"_id": run_id}, {"seed_url": 1})
    if run and run.get("seed_url"):
        store.record_override(run["seed_url"], run_id, url, "remove")
    return {"status": "removed", **inc}


def _corpus_payload(run_id: str) -> dict:
    """Self-describing corpus export: run metadata (incl. extraction settings),
    the user's manual keep/remove selections, and every chunk attributed to
    the run — joined via docs.run_ids (chunk.run_id is overwritten when a
    later run re-writes a shared doc's chunks)."""
    db = get_db()
    run = db.runs.find_one({"_id": run_id}, {"_id": 0}) or {}
    doc_ids = [d["_id"] for d in db.docs.find({"run_ids": run_id}, {"_id": 1})]
    chunks = [{"chunk_id": c["_id"], "doc_id": c["doc_id"],
               "source_url": c["source_url"], "title": c.get("title"),
               "section": c.get("section"), "text": c["text"],
               "embedding": c.get("embedding"),
               "archived": c.get("archived", False)}
              for c in db.chunks.find({"doc_id": {"$in": doc_ids}})]
    overrides = [{"url": o["url"], "action": o["action"], "at": o.get("at")}
                 for o in db.overrides.find({"run_id": run_id}, {"_id": 0})]
    return {"run_id": run_id, "seed_url": run.get("seed_url"),
            "models": {"llm": run.get("llm_model"),
                       "embed": run.get("embed_model")},
            "settings": run.get("settings", {}),
            "counts": run.get("counts", {}),
            "overrides": overrides, "chunks": chunks}


@app.get("/api/runs/{run_id}/export.pkl")
def export_pkl(run_id: str):
    """Corpus as a pickled dict — {run_id, seed_url, models, settings,
    counts, overrides, chunks}. Load with:
        corpus = pickle.load(open('corpus-<run>.pkl','rb'))
        chunks = corpus["chunks"]"""
    return Response(
        pickle.dumps(_corpus_payload(run_id), protocol=5),
        media_type="application/octet-stream",
        headers={"Content-Disposition":
                 f'attachment; filename="corpus-{run_id}.pkl"'})


@app.get("/api/runs/{run_id}/export.jsonl")
def export_jsonl(run_id: str):
    """Same corpus as JSONL — first line is metadata (_meta), rest are chunks."""
    p = _corpus_payload(run_id)
    meta = {k: p[k] for k in ("run_id", "seed_url", "models", "settings",
                              "counts", "overrides")}
    payload = (json.dumps({"_meta": meta}, ensure_ascii=False) + "\n" +
               "".join(json.dumps(r, ensure_ascii=False) + "\n"
                       for r in p["chunks"]))
    return Response(
        payload, media_type="application/jsonl",
        headers={"Content-Disposition":
                 f'attachment; filename="corpus-{run_id}.jsonl"'})


@app.get("/api/runs")
def runs():
    return sorted(
        ({"run_id": r["_id"], **{k: v for k, v in r.items() if k != "_id"}}
         for r in get_db().runs.find({})),
        key=lambda r: r.get("created_at", ""), reverse=True)


@app.get("/api/pages")
def pages(run_id: str = Query(...), status: str | None = None):
    q: dict = {"run_id": run_id}
    if status:
        q["status"] = status
    return [{"url": p["url"], "status": p["status"], "reason": p["reason"],
             "word_count": p["word_count"]}
            for p in get_db().pages.find(q, {"_id": 0}).sort("url", 1)]


@app.get("/api/docs")
def docs(run_id: str = Query(...)):
    return [{"doc_id": d["_id"], "url": d["url"], "title": d["title"],
             "words": len(d["markdown"].split())}
            for d in get_db().docs.find({"run_ids": run_id}).sort("url", 1)]


@app.get("/api/chunks")
def chunks(run_id: str = Query(...), doc_id: str | None = None):
    q: dict = {"run_id": run_id}
    if doc_id:
        q["doc_id"] = doc_id
    # embedding excluded — too bulky for the browser; fetch it from mongo directly
    return [{"chunk_id": c["_id"], "doc_id": c["doc_id"],
             "source_url": c["source_url"], "title": c["title"],
             "section": c["section"], "text": c["text"],
             "archived": c.get("archived", False)}
            for c in get_db().chunks.find(q, {"embedding": 0}).sort("_id", 1)]


_FAVICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<rect width="64" height="64" rx="12" fill="#111"/>
<text x="32" y="46" font-size="38" font-family="monospace" font-weight="bold"
      text-anchor="middle" fill="#2d5">E</text>
</svg>"""


@app.get("/favicon.ico")
def favicon():
    return Response(content=_FAVICON, media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse)
def index():
    return _PAGE


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>extractor</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" type="image/svg+xml" href="/favicon.ico">
<style>
  body{font-family:ui-monospace,Menlo,monospace;background:#111;color:#ddd;
       max-width:1100px;margin:2rem auto;padding:0 1rem}
  input,button{font:inherit;padding:.5rem;background:#1c1c1c;color:#ddd;
               border:1px solid #444}
  input{width:60%}
  button{cursor:pointer;background:#2d5}
  h2{margin-top:2rem;border-bottom:1px solid #333;padding-bottom:.3rem}
  table{border-collapse:collapse;width:100%;font-size:.85rem}
  td,th{border:1px solid #333;padding:.3rem .5rem;text-align:left;
        vertical-align:top}
  th{white-space:nowrap}
  td.nw{white-space:nowrap}
  td.u{word-break:break-all}
  td.u a{color:#7ab;text-decoration:none}
  td.u a:hover{color:#2d5;text-decoration:underline}
  .run{cursor:pointer;padding:.4rem;border:1px solid #333;margin:.3rem 0}
  .run:hover{background:#1c1c1c}
  .run.sel{border-color:#2d5;background:#17251c}
  .kept{color:#4d7}.dropped,.dropped_llm,.dropped_thin{color:#d66}
  .rejected,.removed{color:#888}
  .flag{color:#fc4}
  .error{color:#d66}.running,.cancelling{color:#fd4}.review,.scrape_error{color:#fa0}
  .cancelled{color:#c9a227}
  .empty{color:#fa0;font-style:italic}
  .pulse{color:#2d5;animation:pl 1.2s infinite}
  @keyframes pl{50%{opacity:.15}}
  .stalled{color:#fa0}
  button.kill{background:#c44}
  button.small{padding:.2rem .6rem;font-size:.8rem}
  a.btn{display:inline-block;padding:.2rem .6rem;font-size:.8rem;
      background:#2a2a2a;color:#ddd;border:none;border-radius:3px;
      text-decoration:none;cursor:pointer}
  a.btn:hover{background:#3a3a3a}
  pre{background:#1c1c1c;padding:1rem;overflow:auto;font-size:.8rem;
      white-space:pre-wrap;word-break:break-word}
  details{margin:.5rem 0}
  details summary{cursor:pointer;font-weight:700;font-size:1.1rem;
      padding:.3rem 0;user-select:none}
  details summary:hover{color:#2d5}
  .chunk{padding:.7rem .9rem;font-size:.8rem;white-space:pre-wrap;
      word-break:break-word;background:#1c1c1c;
      border:1px solid #2a2a2a;margin:.3rem 0;border-radius:3px}
  .chunk:nth-of-type(even){background:#242424}
  .chunk .meta{color:#7ab;display:block;margin-bottom:.4rem;font-size:.75rem}
</style></head><body>
<h1>extractor</h1>
<form id="f"><input id="url" placeholder="https://docs.example.com" required>
<button>Run</button></form>
<div id="msg"></div>

<h2>Jobs</h2><div id="jobs">none</div>
<h2>Runs</h2><div id="runs">none</div>
<div id="detail"></div>

<script>
const $ = id => document.getElementById(id);
const esc = s => (s||'').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

// Internal status codes -> plain English
const STATUS_LABELS = {
  kept: 'kept',
  review: 'needs review',
  dropped_thin: 'too short',
  dropped_llm: 'dropped by AI filter',
  scrape_error: 'scrape failed',
  rejected: 'skipped (URL rule)',
  removed: 'removed (manual)'
};
// Triage rule codes -> plain English
const RULE_LABELS = {
  'noise-path': 'URL pattern — tag/author/category/pagination/login/…',
  'binary-or-feed-extension': 'file or feed — pdf, xml, image, css, js…',
  'non-http-scheme': 'not an http(s) URL',
  'user-excluded': 'excluded by EXCLUDE_PATH_REGEX',
  'date-archive': 'date archive listing — duplicate of posts'
};
const statusLabel = s => STATUS_LABELS[s] || s;
const reasonLabel = r => RULE_LABELS[r] || r || '';
// Page audit row order: kept first, then descending severity of drop
const STATUS_RANK = {kept:0, review:1, dropped_llm:2, dropped_thin:3,
                     scrape_error:4, rejected:5, removed:6};

// "2026-10-15 02:35" — date + HH:MM only
const fmtTime = iso => {
  if (!iso) return '—';
  const d = new Date(iso), p = n => String(n).padStart(2,'0');
  return `${d.getFullYear()}-${p(d.getMonth()+1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
};
// seconds -> "1h 5m" / "12m 34s" / "45s"
const fmtDur = s => {
  if (s == null) return '—';
  const t = Math.round(s);          // round once — avoids "1m 60s"
  const h = Math.floor(t / 3600), m = Math.floor(t % 3600 / 60), r = t % 60;
  return h ? `${h}h ${m}m` : m ? `${m}m ${r}s` : `${r}s`;
};

async function post(url, body) {
  return (await fetch(url, {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: body ? JSON.stringify(body) : null})).json();
}
async function killJob(id) {
  await post(`/api/jobs/${id}/kill`); refresh();
}
async function resumeJob(id, url) {
  // Same run_id resumes the checkpoint — no re-scrape of finished pages
  await post('/api/runs', {url, run_id: id}); refresh();
}
async function rerunRun(id) {
  await post(`/api/runs/${id}/rerun`); refresh();
}
async function deleteRun(id) {
  if (!confirm(`Delete run "${id}"?\n\nRemoves its docs, chunks, page audit, ` +
               `checkpoints and output files. Shared docs kept by other runs ` +
               `are preserved.`)) return;
  await fetch(`/api/runs/${id}`, {method:'DELETE'});
  $('detail').innerHTML = '';
  refresh();
}
document.getElementById('f').onsubmit = async e => {
  e.preventDefault();
  const r = await post('/api/runs', {url: $('url').value});
  $('msg').textContent = r.detail ||
    (r.status === 'already_running'
      ? `already running — resubmits to ${r.run_id}`
      : `started: ${r.run_id}`);
  refresh();
};

async function refresh() {
  const jobs = await (await fetch('/api/jobs')).json();
  $('jobs').innerHTML = Object.entries(jobs).map(([k,v]) =>
    `<div class="run"><b>${k}</b> <span class="${v.status}">${v.status}</span>` +
    (v.url ? `<br><small>${esc(v.url)}</small>` : '') +
    (v.started_at ? ` — started ${fmtTime(v.started_at)}` : '') +
    (v.progress ? ` — ${esc(v.progress.stage)}` : '') +
    (v.status === 'cancelling'
      ? ' <span class="stalled">◌ cancelling…</span>'
      : v.status === 'running'
      ? (v.idle_s != null && v.idle_s >= 30
          ? ` <span class="stalled">◌ no progress for ${fmtDur(v.idle_s)}</span>`
          : ' <span class="pulse">●</span>') : '') +
    (v.writes != null ? ` <small>(${v.writes} tasks done)</small>` : '') +
    (v.error ? ` — ${esc(v.error)}` : '') +
    (v.status === 'running'
      ? ` <button class="kill small" onclick="killJob('${k}')">Kill</button>` : '') +
    (['cancelled','error'].includes(v.status)
      ? ` <button class="small" onclick="resumeJob('${k}','${esc(v.url||'')}')">Resume</button>` +
        ` <button class="kill small" onclick="deleteRun('${k}')">Delete</button>` : '') +
    '</div>').join('') || 'none';

  const runs = await (await fetch('/api/runs')).json();
  $('runs').innerHTML = runs.map(r => {
    const c = r.counts || {};
    const st = c.pages_by_status || {};
    const line1 = `started ${fmtTime(r.started_at)} · ` +
      `took ${fmtDur(r.duration_seconds)}<br>` +
      `discovered: ${c.urls_mapped||0} · ` +
      `skipped by URL rules: ${c.urls_rejected||0} · ` +
      `scraped: ${c.urls_filtered||0}`;
    const line2 = Object.entries(st)
      .map(([k,v]) => `${statusLabel(k)}: ${v}`).join(' · ') +
      ` · documents: ${c.docs||0} · chunks: ${c.chunks||0}`;
    const flag = c.catalog_heavy
      ? `<br><span class="flag">large product catalog — ` +
        `${c.catalog_urls_skipped} product URLs skipped. ` +
        `Company may be a poor fit for a RAG chatbot offering.</span>` : '';
    const empty = (c.docs||0) === 0
      ? `<br><span class="empty">Nothing kept — corpus is empty. ` +
        `Click to see why each URL was dropped.</span>` : '';
    return `<div class="run${r.run_id===_sel?' sel':''}" onclick="detail('${r.run_id}')">` +
      `<b>${r.run_id}</b> ` +
      `<button class="small" onclick="event.stopPropagation();rerunRun('${r.run_id}')">Re-run</button> ` +
      `<a class="small btn" href="/api/runs/${r.run_id}/export.pkl" onclick="event.stopPropagation()">pkl</a> ` +
      `<a class="small btn" href="/api/runs/${r.run_id}/export.jsonl" onclick="event.stopPropagation()">jsonl</a> ` +
      `<button class="kill small" onclick="event.stopPropagation();deleteRun('${r.run_id}')">Delete</button>` +
      `<br>${esc(r.seed_url||'')}<br>` +
      `${line1}<br><small>${line2}</small>${flag}${empty}</div>`;
  }).join('') || 'none';
}

async function keepPage(runId, btn) {
  const url = btn.closest('tr').querySelector('.u a').textContent;
  const r = await post(`/api/runs/${runId}/keep?url=` + encodeURIComponent(url));
  if (r.detail) { $('msg').textContent = r.detail; return; }
  detail(runId); refresh();
}

async function removePage(runId, btn) {
  const url = btn.closest('tr').querySelector('.u a').textContent;
  const r = await post(`/api/runs/${runId}/remove?url=` + encodeURIComponent(url));
  if (r.detail) { $('msg').textContent = r.detail; return; }
  detail(runId); refresh();
}

let _sel = null;
async function detail(runId) {
  _sel = runId; refresh();
  const ps = await (await fetch(`/api/pages?run_id=${runId}`)).json();
  const cs = await (await fetch(`/api/chunks?run_id=${runId}`)).json();
  ps.sort((a,b) => (STATUS_RANK[a.status]??9)-(STATUS_RANK[b.status]??9)
                   || a.url.localeCompare(b.url));
  $('detail').innerHTML = `<h2>${runId}</h2>
    <details open><summary>Page audit (${ps.length} URLs)</summary><table>
    <tr><th>status</th><th>words</th><th>url</th><th>why</th><th></th></tr>` +
    ps.map(p => `<tr><td class="${p.status} nw">${statusLabel(p.status)}</td>
      <td class="nw">${p.word_count||''}</td><td class="u"><a href="${esc(p.url).replace(/"/g,'%22')}" target="_blank" rel="noopener">${esc(p.url)}</a></td>
      <td>${esc(reasonLabel(p.reason))}</td>
      <td class="nw">${p.status==='kept'
        ? `<button class="small kill" onclick="removePage('${runId}',this)">remove</button>`
        : `<button class="small" onclick="keepPage('${runId}',this)">keep</button>`}</td>
      </tr>`).join('') + `</table></details>
    <details><summary>Sample chunks (${Math.min(5, cs.length)} of ${cs.length})</summary>` +
    (cs.slice(0,5).map(c =>
      `<div class="chunk"><span class="meta">${esc(c.source_url)} — ${esc(c.section||'(root)')}</span>${esc(c.text.slice(0,600))}…</div>`
    ).join('') || '(no chunks)') + '</details>';
}
refresh(); setInterval(refresh, 4000);
</script></body></html>"""
