"""MongoDB persistence. All writes are upserts keyed by content hash IDs,
so re-running a pipeline (or resuming a checkpoint) is idempotent."""
import hashlib
import json
import os
from datetime import datetime, timezone
from urllib.parse import urlsplit

from pymongo import MongoClient, UpdateOne

from .config import SETTINGS

_ARCHIVE_NOTE = ("[archived content — may no longer be current or available]"
                 "\n\n")

_client: MongoClient | None = None


def get_db():
    global _client
    if _client is None:
        _client = MongoClient(SETTINGS.mongo_uri)
    return _client[SETTINGS.mongo_db]


def save_run(state: dict, corpus_path: str) -> dict:
    """Upsert docs, page-audit records, chunks (read back from the JSONL
    artifact to keep them out of checkpoint state), and the run manifest."""
    db = get_db()
    run_id = state["run_id"]

    db.docs.bulk_write([
        UpdateOne(
            {"_id": d["doc_id"]},
            {"$setOnInsert": {"url": d["url"], "title": d["title"],
                              "markdown": d["markdown"],
                              "archived": d["archived"]},
             "$addToSet": {"run_ids": run_id}},
            upsert=True,
        )
        for d in state["docs"]
    ]) if state["docs"] else None

    # Scraped pages + URLs rejected at triage (so the audit answers
    # "filtered — but why?" for every discovered URL)
    page_ops = [
        UpdateOne(
            {"_id": f"{run_id}:{hashlib.sha256(url.encode()).hexdigest()[:16]}"},
            {"$set": {"url": url, "run_id": run_id, "status": p["status"],
                      "reason": p["reason"], "word_count": p["word_count"],
                      "doc_id": p["doc_id"]}},
            upsert=True,
        )
        for url, p in state["pages"].items()
        if not p.get("dupe")  # identical-body pages -> count only, no row
    ] + [
        UpdateOne(
            {"_id": f"{run_id}:{hashlib.sha256(url.encode()).hexdigest()[:16]}"},
            {"$set": {"url": url, "run_id": run_id, "status": "rejected",
                      "reason": rule, "word_count": 0, "doc_id": None}},
            upsert=True,
        )
        for url, rule in state.get("rejected_urls", {}).items()
    ]
    if page_ops:
        db.pages.bulk_write(page_ops)

    chunks = []
    with open(corpus_path) as f:
        for line in f:
            c = json.loads(line)
            chunks.append(UpdateOne(
                {"_id": c["chunk_id"]},
                {"$set": {"doc_id": c["doc_id"], "source_url": c["source_url"],
                          "title": c["title"], "section": c["section"],
                          "text": c["text"], "embedding": c["embedding"],
                          "archived": c.get("archived", False),
                          "run_id": run_id}},
                upsert=True,
            ))
    if chunks:
        db.chunks.bulk_write(chunks)

    db.runs.update_one(
        {"_id": run_id},
        {"$set": state["manifest"]},
        upsert=True,
    )
    return {"run_id": run_id, "docs": len(state["docs"]), "chunks": len(chunks)}


def ensure_indexes() -> None:
    db = get_db()
    db.chunks.create_index("run_id")
    db.chunks.create_index("doc_id")
    db.pages.create_index([("run_id", 1), ("status", 1)])
    db.docs.create_index("url")
    db.overrides.create_index([("seed_url", 1), ("url", 1)], unique=True)


# --- Manual curation ---------------------------------------------------------
# keep/remove overrides are stored per seed_url so a re-run of the same site
# re-applies the user's selections at persist time.

def record_override(seed_url: str, run_id: str, url: str, action: str) -> None:
    get_db().overrides.update_one(
        {"seed_url": seed_url, "url": url},
        {"$set": {"action": action, "run_id": run_id,
                  "at": datetime.now(timezone.utc).isoformat()}},
        upsert=True)


def _page_pid(run_id: str, url: str) -> str:
    return f"{run_id}:{hashlib.sha256(url.encode()).hexdigest()[:16]}"


def apply_keep(run_id: str, url: str, title: str | None, markdown: str,
               old_status: str, corpus_path: str,
               reason: str | None = None) -> dict:
    """Chunk + embed + persist a page as kept. Returns the $inc dict for the
    run's manifest counts."""
    from . import chunking, clients
    db = get_db()
    doc_id = hashlib.sha256(markdown.encode()).hexdigest()
    archived = "archive" in urlsplit(url).path.lower()
    new_doc = db.docs.find_one({"_id": doc_id, "run_ids": run_id}) is None
    db.docs.update_one(
        {"_id": doc_id},
        {"$setOnInsert": {"url": url, "title": title, "markdown": markdown,
                          "archived": archived},
         "$addToSet": {"run_ids": run_id}},
        upsert=True)

    chunks = []
    for c in chunking.chunk_document(doc_id, url, title or "", markdown):
        c["archived"] = archived
        if archived:
            c["text"] = _ARCHIVE_NOTE + c["text"]
        chunks.append(c)
    for i in range(0, len(chunks), 32):
        batch = chunks[i:i + 32]
        for c, vec in zip(batch, clients.embed([c["text"] for c in batch])):
            c["embedding"] = vec
    res = db.chunks.bulk_write([
        UpdateOne({"_id": c["chunk_id"]},
                  {"$set": {**{k: c[k] for k in
                               ("doc_id", "source_url", "title", "section",
                                "text", "embedding", "archived")},
                            "run_id": run_id}},
                  upsert=True)
        for c in chunks])

    if os.path.exists(corpus_path):
        with open(corpus_path, "a") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    db.pages.update_one(
        {"_id": _page_pid(run_id, url)},
        {"$set": {"status": "kept", "doc_id": doc_id,
                  "reason": reason or f"manually kept — was {old_status}"}})

    inc = {f"counts.pages_by_status.{old_status}": -1,
           "counts.pages_by_status.kept": 1,
           "counts.chunks": res.upserted_count}
    if new_doc:
        inc["counts.docs"] = 1
    return inc


def apply_remove(run_id: str, url: str, doc_id: str, old_status: str,
                 corpus_path: str,
                 reason: str | None = None) -> dict:
    """Pull a page's doc/chunks out of the corpus. Returns the $inc dict."""
    db = get_db()
    doc = db.docs.find_one({"_id": doc_id})
    n_chunks = doc_removed = 0
    # only touch doc/chunks if this page is the doc's canonical source —
    # a dedupe loser shares doc_id with the surviving sibling's chunks
    if doc and doc.get("url") == url:
        shared = any(r != run_id for r in doc.get("run_ids", []))
        doomed = [c["_id"] for c in db.chunks.find(
            {"doc_id": doc_id} | ({"run_id": run_id} if shared else {}),
            {"_id": 1})]
        n_chunks = db.chunks.delete_many({"_id": {"$in": doomed}}).deleted_count
        if shared:
            db.docs.update_one({"_id": doc_id},
                               {"$pull": {"run_ids": run_id}})
        else:
            db.docs.delete_one({"_id": doc_id})
        doc_removed = 1
        if doomed and os.path.exists(corpus_path):
            with open(corpus_path) as f:
                lines = [l for l in f
                         if json.loads(l)["chunk_id"] not in doomed]
            with open(corpus_path, "w") as f:
                f.writelines(lines)

    db.pages.update_one(
        {"_id": _page_pid(run_id, url)},
        {"$set": {"status": "removed", "doc_id": None,
                  "reason": reason or f"manually removed — was {old_status}"}})
    return {f"counts.pages_by_status.{old_status}": -1,
            "counts.pages_by_status.removed": 1,
            "counts.chunks": -n_chunks,
            "counts.docs": -doc_removed}


def apply_overrides(state: dict) -> None:
    """Re-apply curated keep/remove selections for this seed site after a
    re-run — the classifier verdict doesn't override the user's judgment."""
    db = get_db()
    ovs = list(db.overrides.find({"seed_url": state["seed_url"]}))
    if not ovs:
        return
    run_id = state["run_id"]
    corpus_path = state["manifest"]["outputs"]["corpus"]
    inc: dict[str, int] = {}
    for ov in sorted(ovs, key=lambda o: o["url"]):
        page = (state.get("pages") or {}).get(ov["url"])
        st = (page or {}).get("status")
        if ov["action"] == "keep" and page and st != "kept":
            md = page.get("markdown") or ""
            if len(md.split()) < 3:
                continue
            d = apply_keep(run_id, ov["url"], page.get("title"), md, st,
                           corpus_path, reason="manual override")
        elif (ov["action"] == "remove" and page
                and st in ("kept", "review")):
            d = apply_remove(run_id, ov["url"], page["doc_id"], st,
                             corpus_path, reason="manual override")
        else:
            continue
        for k, v in d.items():
            inc[k] = inc.get(k, 0) + v
    if inc:
        db.runs.update_one({"_id": run_id}, {"$inc": inc})
        # keep the on-disk manifest in sync with adjusted counts
        mpath = os.path.join(os.path.dirname(corpus_path) or ".",
                             f"manifest-{run_id}.json")
        if os.path.exists(mpath):
            with open(mpath) as f:
                m = json.load(f)
            for k, v in inc.items():
                parts = k.split(".")
                node = m
                for p in parts[1:-1]:
                    node = node.setdefault(p, {})
                node[parts[-1]] = node.get(parts[-1], 0) + v
            with open(mpath, "w") as f:
                json.dump(m, f, indent=2)
