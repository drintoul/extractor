"""MongoDB persistence. All writes are upserts keyed by content hash IDs,
so re-running a pipeline (or resuming a checkpoint) is idempotent."""
import hashlib
import json

from pymongo import MongoClient, UpdateOne

from .config import SETTINGS

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
