"""Graph nodes. All routing is pure functions of state; LLM calls are pinned
to temperature=0 + fixed seed and used only for keep/drop classification."""
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from urllib.parse import urlsplit

from langgraph.types import Send
from pydantic import BaseModel

from . import cancel, chunking, clients, dedupe as dd, denoise, sitemap, store, triage
from .config import SETTINGS
from .state import CorpusState, PageResult, PageTask

_CLASSIFY_SYSTEM = (
    "You are a content filter for a RAG corpus powering a chatbot that answers "
    "visitor/customer questions about this website. KEEP any page that could "
    "help answer a real question: menus, products, services, prices, events, "
    "schedules, entertainment, hours, locations, directions, contact info, "
    "reservations, about/history, team, FAQ, docs, guides, policies that affect "
    "customers. REJECT only pages with no usable information: navigation or "
    "link-list stubs, login/register/account pages, admin/edit screens, error "
    "pages, empty templates, cookie/legal boilerplate, image galleries without "
    "text. "
    'Respond ONLY as JSON: {"keep": true|false, "reason": "short reason"}'
)


class Verdict(BaseModel):
    keep: bool
    reason: str


# 'archive' anywhere in the URL path marks the doc as historical content —
# e.g. /beer-archive/<beer> describes discontinued items. Chunks get both a
# structured flag and an in-text note so the RAG layer hedges availability.
_ARCHIVED = re.compile(r"archive", re.IGNORECASE)
_ARCHIVE_NOTE = ("[archived content — may no longer be current or available]"
                 "\n\n")


def map_node(state: CorpusState) -> dict:
    links = set(clients.map_site(state["seed_url"]))
    out: dict = {}
    if len(links) < 10:
        # map found (almost) nothing — site likely has a nonstandard or
        # missing sitemap (e.g. BigCommerce xmlsitemap.php). Fall back to
        # parsing sitemaps directly.
        res = sitemap.discover(state["seed_url"], SETTINGS.map_limit)
        links |= set(res["urls"])
        out["catalog_skipped_urls"] = res["skipped_catalog_urls"]
    out["url_inventory"] = sorted(links)
    if not state.get("started_at"):
        out["started_at"] = datetime.now(timezone.utc).isoformat()
    return out


def triage_node(state: CorpusState) -> dict:
    accepted, rejected = triage.triage(state["url_inventory"],
                                       SETTINGS.exclude_path_regex)
    return {"filtered_urls": accepted, "rejected_urls": rejected}


def fan_out(state: CorpusState):
    if not state["filtered_urls"]:
        return ["dedupe"]
    return [Send("process_page", {"url": u, "run_id": state["run_id"]})
            for u in state["filtered_urls"]]


def process_page_node(task: PageTask) -> dict:
    if cancel.is_cancelled(task["run_id"]):
        raise cancel.Cancelled()
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
    if cancel.is_cancelled(state["run_id"]):
        raise cancel.Cancelled()
    kept = [p for _, p in sorted(state["pages"].items())
            if p["status"] in ("kept", "review")]
    dupes = dd.find_duplicates([p["doc_id"] for p in kept],
                               [p["markdown"] for p in kept])
    docs = [{"doc_id": p["doc_id"], "url": p["url"], "title": p["title"],
             "markdown": p["markdown"],
             "archived": bool(_ARCHIVED.search(urlsplit(p["url"]).path))}
            for p in kept if p["doc_id"] not in dupes]
    return {"docs": sorted(docs, key=lambda d: d["doc_id"])}


def chunk_embed_node(state: CorpusState) -> dict:
    if cancel.is_cancelled(state["run_id"]):
        raise cancel.Cancelled()
    chunks = []
    for d in state["docs"]:
        for c in chunking.chunk_document(
                d["doc_id"], d["url"], d["title"] or "", d["markdown"]):
            c["archived"] = d["archived"]
            if d["archived"]:
                c["text"] = _ARCHIVE_NOTE + c["text"]
            chunks.append(c)

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

    # Fit signal: product-catalog URLs excluded pre-scrape (skipped sitemaps
    # + user-excluded path patterns). Flag when they outnumber real content.
    catalog_urls = (state.get("catalog_skipped_urls") or 0) + sum(
        1 for r in (state.get("rejected_urls") or {}).values()
        if r == "user-excluded")

    finished_at = datetime.now(timezone.utc)
    started_at = state.get("started_at")
    duration = (finished_at - datetime.fromisoformat(started_at)).total_seconds() \
        if started_at else None

    manifest = {
        "run_id": state["run_id"],
        "seed_url": state["seed_url"],
        "started_at": started_at,
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round(duration, 1) if duration is not None else None,
        "created_at": finished_at.isoformat(),
        "llm_model": SETTINGS.llm_model,
        "embed_model": SETTINGS.embed_model,
        "llm_seed": SETTINGS.llm_seed,
        "settings": {
            "min_words": SETTINGS.min_words,
            "map_limit": SETTINGS.map_limit,
            "scrape_timeout_ms": SETTINGS.scrape_timeout_ms,
            "sitemap_skip_patterns": SETTINGS.sitemap_skip_patterns,
            "exclude_path_regex": SETTINGS.exclude_path_regex,
        },
        "counts": {
            "urls_mapped": len(state["url_inventory"]),
            "urls_rejected": len(state["rejected_urls"]),
            "urls_filtered": len(state["filtered_urls"]),
            "pages_by_status": status_counts,
            "docs": len(state["docs"]),
            "chunks": len(chunks),
            "catalog_urls_skipped": catalog_urls,
            "catalog_heavy": (catalog_urls >= 50
                              and catalog_urls > len(state["filtered_urls"])),
        },
        "outputs": {"corpus": corpus_path},
    }
    manifest_path = os.path.join(SETTINGS.data_dir, f"manifest-{state['run_id']}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return {"chunks_written": len(chunks), "manifest": manifest}


def persist_node(state: CorpusState) -> dict:
    if cancel.is_cancelled(state["run_id"]):
        raise cancel.Cancelled()
    store.ensure_indexes()
    store.save_run(state, state["manifest"]["outputs"]["corpus"])
    # re-apply the user's curated keep/remove selections for this site
    store.apply_overrides(state)
    return {}
