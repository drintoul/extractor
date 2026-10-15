import operator
from typing import Annotated, Optional, TypedDict


class PageResult(TypedDict):
    url: str
    status: str  # kept | dropped_rules | dropped_thin | dropped_llm | review | scrape_error
    title: Optional[str]
    markdown: str
    word_count: int
    doc_id: Optional[str]  # sha256 of cleaned markdown; None when dropped
    reason: Optional[str]


class PageTask(TypedDict):
    """Payload sent to each process_page fan-out via Send()."""
    url: str


class CorpusState(TypedDict):
    seed_url: str
    run_id: str
    url_inventory: list[str]
    rejected_urls: dict[str, str]  # url -> rejection rule
    filtered_urls: list[str]
    # Keyed by URL so parallel fan-out results merge order-independently.
    pages: Annotated[dict[str, PageResult], operator.or_]
    docs: list[dict]
    chunks_written: int
    manifest: dict
