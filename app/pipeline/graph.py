"""Deterministic LangGraph assembly.

START → map → triage → dispatch ──[Send batch]──→ process_page ─┐
                    ↑______________ (next batch) ______________┘
                    └── (urls exhausted) ─→ dedupe → chunk_embed → persist → END

The fan-out is batched (BATCH_SIZE per superstep): kills are honored at the
dispatch boundary, and each batch commits a checkpoint so progress is live
and resumes replay at batch granularity.

Conditional edges are pure functions of state — the LLM never decides routing.
Checkpointing makes every node output replayable and per-page retries free.
"""
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from . import nodes
from .config import SETTINGS
from .state import CorpusState


def build_graph(checkpointer=None):
    builder = StateGraph(CorpusState)
    builder.add_node("map", nodes.map_node)
    builder.add_node("triage", nodes.triage_node)
    builder.add_node("dispatch", nodes.dispatch_node)
    builder.add_node("process_page", nodes.process_page_node)
    builder.add_node("dedupe", nodes.dedupe_node)
    builder.add_node("chunk_embed", nodes.chunk_embed_node)
    builder.add_node("persist", nodes.persist_node)

    builder.add_edge(START, "map")
    builder.add_edge("map", "triage")
    builder.add_edge("triage", "dispatch")
    builder.add_conditional_edges("dispatch", nodes.fan_out,
                                  ["process_page", "dedupe"])
    builder.add_edge("process_page", "dispatch")  # next batch
    builder.add_edge("dedupe", "chunk_embed")
    builder.add_edge("chunk_embed", "persist")
    builder.add_edge("persist", END)
    return builder.compile(checkpointer=checkpointer)


def build_checkpointer() -> SqliteSaver:
    import os
    os.makedirs(os.path.dirname(SETTINGS.checkpoint_db) or ".", exist_ok=True)
    conn = sqlite3.connect(SETTINGS.checkpoint_db, check_same_thread=False)
    return SqliteSaver(conn)


def delete_checkpoints(thread_id: str) -> int:
    """Remove all checkpoint rows for a run — deletes resume capability."""
    conn = sqlite3.connect(SETTINGS.checkpoint_db, check_same_thread=False)
    deleted = 0
    for table in ("checkpoints", "writes",
                  "checkpoint_writes", "checkpoint_blobs"):
        try:
            deleted += conn.execute(
                f"DELETE FROM {table} WHERE thread_id=?", (thread_id,)
            ).rowcount
        except sqlite3.OperationalError:
            pass  # table name differs across langgraph-checkpoint versions
    conn.commit()
    conn.close()
    return deleted
