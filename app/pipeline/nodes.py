"""Graph nodes. All routing is pure functions of state; LLM calls are pinned
to temperature=0 + fixed seed and used only for keep/drop classification."""
import hashlib
import json
import os
from datetime import datetime, timezone

from langgraph.types import Send
from pydantic import BaseModel

from . import chunking, clients, dedupe as dd, denoise, store, triage
from .config import SETTINGS
from .state import CorpusState, PageResult, PageTask

_CLASSIFY_SYSTEM = (
    "You are a strict content filter for a RAG corpus. Decide whether the page "
    "contains substantive informational content worth keeping for question "
    "answering. Reject navigation pages, stubs, listings with no prose, "
    "login/error pages, and pure promotional filler. "
    'Respond ONLY as JSON: {"keep": true|false, "reason": "short reason"}'
)


class Verdict(BaseModel):
    keep: bool
    reason: str


def map_node(state: CorpusState) -> dict:
    links = clients.map_site(state["seed_url"])
    return {"url_inventory": sorted(set(links))}


def triage_node(state: CorpusState) -> dict:
    accepted, rejected = triage.triage(state["url_inventory"])
    return {"filtered_urls": accepted, "rejected_urls": rejected}


def fan_out(state: CorpusState):
    if not state["filtered_urls"]:
        return ["dedupe"]
    return [Send("process_page", {"url": u}) for u in state["filtered_urls"]]


def process_page_node(task: PageTask) -> dict:
    url = task["url"]
    page = PageResult(url=url, status="", title=None, markdown="",
                      word_count=0, doc_id=None, reason=None)
    try:
        data = clients.scrape_page(url)
    except Exception as e:
        page.update(status="scrape_error", reason=str(e)[:300])
        return {"pages": {url: page}}

    md = denoise.clean_markdown(data.get("markdown") or "")
    wc = denoise.word_count(md)
    page.update(markdown=md, word_count=wc,
                title=(data.get("metadata") or {}).get("title"))

    if wc < SETTINGS.min_words:
        page.update(status="dropped_thin", reason=f"{wc} words < {SETTINGS.min_words}")
    else:
        truncated = " ".join(md.split()[:3000])
        verdict = clients.chat_json(
            [{"role": "system", "content": _CLASSIFY_SYSTEM},
             {"role": "user", "content": f"URL: {url}\n\nCONTENT:\n{truncated}"}],
            Verdict,
        )
        if verdict is None:
            page.update(status="review", reason="llm_parse_error")
        elif verdict.keep:
            page.update(status="kept")
        else:
            page.update(status="dropped_llm", reason=verdict.reason[:200])

    if page["status"] in ("kept", "review"):
        page["doc_id"] = hashlib.sha256(md.encode()).hexdigest()
    return {"pages": {url: page}}


def dedupe_node(state: CorpusState) -> dict:
    kept = [p for _, p in sorted(state["pages"].items())
            if p["status"] in ("kept", "review")]
    dupes = dd.find_duplicates([p["doc_id"] for p in kept],
                               [p["markdown"] for p in kept])
    docs = [{"doc_id": p["doc_id"], "url": p["url"], "title": p["title"],
             "markdown": p["markdown"]}
            for p in kept if p["doc_id"] not in dupes]
    return {"docs": sorted(docs, key=lambda d: d["doc_id"])}


def chunk_embed_node(state: CorpusState) -> dict:
    chunks = []
    for d in state["docs"]:
        chunks.extend(chunking.chunk_document(
            d["doc_id"], d["url"], d["title"] or "", d["markdown"]))

    for i in range(0, len(chunks), 32):
        batch = chunks[i:i + 32]
        for c, vec in zip(batch, clients.embed([c["text"] for c in batch])):
            c["embedding"] = vec

    os.makedirs(SETTINGS.data_dir, exist_ok=True)
    corpus_path = os.path.join(SETTINGS.data_dir, f"corpus-{state['run_id']}.jsonl")
    with open(corpus_path, "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    status_counts: dict[str, int] = {}
    for p in state["pages"].values():
        status_counts[p["status"]] = status_counts.get(p["status"], 0) + 1

    manifest = {
        "run_id": state["run_id"],
        "seed_url": state["seed_url"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "llm_model": SETTINGS.llm_model,
        "embed_model": SETTINGS.embed_model,
        "llm_seed": SETTINGS.llm_seed,
        "counts": {
            "urls_mapped": len(state["url_inventory"]),
            "urls_rejected": len(state["rejected_urls"]),
            "urls_filtered": len(state["filtered_urls"]),
            "pages_by_status": status_counts,
            "docs": len(state["docs"]),
            "chunks": len(chunks),
        },
        "outputs": {"corpus": corpus_path},
    }
    manifest_path = os.path.join(SETTINGS.data_dir, f"manifest-{state['run_id']}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return {"chunks_written": len(chunks), "manifest": manifest}


def persist_node(state: CorpusState) -> dict:
    store.ensure_indexes()
    store.save_run(state, state["manifest"]["outputs"]["corpus"])
    return {}
