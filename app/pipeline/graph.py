"""Deterministic LangGraph assembly.

START → map → triage ──[Send per URL]──→ process_page → dedupe → chunk_embed → persist → END

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
    builder.add_node("process_page", nodes.process_page_node)
    builder.add_node("dedupe", nodes.dedupe_node)
    builder.add_node("chunk_embed", nodes.chunk_embed_node)
    builder.add_node("persist", nodes.persist_node)

    builder.add_edge(START, "map")
    builder.add_edge("map", "triage")
    builder.add_conditional_edges("triage", nodes.fan_out, ["process_page", "dedupe"])
    builder.add_edge("process_page", "dedupe")
    builder.add_edge("dedupe", "chunk_embed")
    builder.add_edge("chunk_embed", "persist")
    builder.add_edge("persist", END)
    return builder.compile(checkpointer=checkpointer)


def build_checkpointer() -> SqliteSaver:
    import os
    os.makedirs(os.path.dirname(SETTINGS.checkpoint_db) or ".", exist_ok=True)
    conn = sqlite3.connect(SETTINGS.checkpoint_db, check_same_thread=False)
    return SqliteSaver(conn)
