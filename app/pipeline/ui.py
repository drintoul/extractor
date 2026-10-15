"""Minimal UI/API server for the pipeline. Serves a no-build single page and
JSON endpoints over Mongo-backed run data. Pipeline runs happen in background
threads with LangGraph checkpointing, so resubmitting a run-id resumes it."""
import re
import threading
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .graph import build_checkpointer, build_graph
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
    return f"{host}-{hashlib.sha256(url.encode()).hexdigest()[:8]}"


class RunRequest(BaseModel):
    url: str
    run_id: str | None = None


@app.post("/api/runs")
def start_run(req: RunRequest):
    if not re.match(r"^https?://", req.url):
        raise HTTPException(400, "url must start with http:// or https://")
    run_id = req.run_id or _default_run_id(req.url)
    if _jobs.get(run_id, {}).get("status") == "running":
        return {"run_id": run_id, "status": "already_running"}

    def work():
        _jobs[run_id] = {"status": "running"}
        try:
            _get_graph().invoke(
                {"seed_url": req.url, "run_id": run_id,
                 "pages": {}, "rejected_urls": {}},
                config={"configurable": {"thread_id": run_id}},
            )
            _jobs[run_id] = {"status": "done"}
        except Exception as e:
            _jobs[run_id] = {"status": "error", "error": str(e)}

    threading.Thread(target=work, daemon=True).start()
    return {"run_id": run_id, "status": "started"}


@app.get("/api/jobs")
def jobs():
    return _jobs


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
             "section": c["section"], "text": c["text"]}
            for c in get_db().chunks.find(q, {"embedding": 0}).sort("_id", 1)]


@app.get("/", response_class=HTMLResponse)
def index():
    return _PAGE


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>extractor</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
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
        vertical-align:top;word-break:break-all}
  .run{cursor:pointer;padding:.4rem;border:1px solid #333;margin:.3rem 0}
  .run:hover{background:#1c1c1c}
  .kept{color:#4d7}.dropped,.dropped_llm,.dropped_thin{color:#d66}
  .error{color:#d66}.running{color:#fd4}.review,.scrape_error{color:#fa0}
  pre{background:#1c1c1c;padding:1rem;overflow:auto;font-size:.8rem}
</style></head><body>
<h1>extractor</h1>
<form id="f"><input id="url" placeholder="https://docs.example.com" required>
<button>Run</button></form>

<h2>Jobs</h2><div id="jobs">none</div>
<h2>Runs</h2><div id="runs">none</div>
<div id="detail"></div>

<script>
const $ = id => document.getElementById(id);
document.getElementById('f').onsubmit = async e => {
  e.preventDefault();
  await fetch('/api/runs', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({url: $('url').value})});
  refresh();
};
async function refresh() {
  const jobs = await (await fetch('/api/jobs')).json();
  $('jobs').innerHTML = Object.entries(jobs).map(([k,v]) =>
    `<div class="run"><b>${k}</b> <span class="${v.status}">${v.status}</span>` +
    (v.error ? ` — ${v.error}` : '') + '</div>').join('') || 'none';
  const runs = await (await fetch('/api/runs')).json();
  $('runs').innerHTML = runs.map(r => {
    const c = r.counts || {};
    const st = Object.entries(c.pages_by_status || {})
      .map(([k,v]) => `${k}:${v}`).join(' ');
    return `<div class="run" onclick="detail('${r.run_id}')"><b>${r.run_id}</b>` +
      `<br>${r.seed_url}<br>mapped:${c.urls_mapped||0} ` +
      `filtered:${c.urls_filtered||0} docs:${c.docs||0} chunks:${c.chunks||0}` +
      `<br><small>${st}</small></div>`;
  }).join('') || 'none';
}
async function detail(runId) {
  const ps = await (await fetch(`/api/pages?run_id=${runId}`)).json();
  const cs = await (await fetch(`/api/chunks?run_id=${runId}`)).json();
  const esc = s => (s||'').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
  $('detail').innerHTML = `<h2>${runId}</h2>
    <h3>Pages (${ps.length})</h3><table><tr><th>status</th><th>words</th>
    <th>url</th><th>reason</th></tr>` +
    ps.map(p => `<tr><td class="${p.status}">${p.status}</td>
      <td>${p.word_count}</td><td>${esc(p.url)}</td>
      <td>${esc(p.reason||'')}</td></tr>`).join('') + `</table>
    <h3>Sample chunks</h3><pre>` +
    cs.slice(0,5).map(c =>
      `# ${esc(c.source_url)}\\n## ${esc(c.section||'(root)')}\\n\\n${esc(c.text.slice(0,600))}…`
    ).join('\\n\\n---\\n\\n') + '</pre>';
}
refresh(); setInterval(refresh, 4000);
</script></body></html>"""
