"""HTTP clients for Firecrawl and Ollama. Raw REST to avoid SDK version drift."""
import json

import httpx
from pydantic import BaseModel, ValidationError

from .config import SETTINGS

_TIMEOUT = httpx.Timeout(120.0)


def map_site(seed_url: str) -> list[str]:
    r = httpx.post(
        f"{SETTINGS.firecrawl_api_url}/v1/map",
        json={"url": seed_url, "limit": SETTINGS.map_limit},
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    return r.json().get("links", [])


def scrape_page(url: str) -> dict:
    r = httpx.post(
        f"{SETTINGS.firecrawl_api_url}/v1/scrape",
        json={
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": True,
            "timeout": SETTINGS.scrape_timeout_ms,
        },
        timeout=SETTINGS.scrape_timeout_ms / 1000 + 90,
    )
    r.raise_for_status()
    body = r.json()
    if not body.get("success"):
        raise RuntimeError(body.get("error") or "scrape failed")
    return body["data"]


def chat_json(messages: list[dict], schema: type[BaseModel]) -> BaseModel | None:
    """Deterministic chat call. Returns parsed model, or None on parse failure."""
    r = httpx.post(
        f"{SETTINGS.ollama_base_url}/api/chat",
        json={
            "model": SETTINGS.llm_model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": SETTINGS.llm_options,
            "keep_alive": "30m",
        },
        timeout=300.0,
    )
    r.raise_for_status()
    try:
        return schema.model_validate(json.loads(r.json()["message"]["content"]))
    except (json.JSONDecodeError, ValidationError, KeyError):
        return None


def embed(texts: list[str]) -> list[list[float]]:
    r = httpx.post(
        f"{SETTINGS.ollama_base_url}/api/embed",
        json={"model": SETTINGS.embed_model, "input": texts},
        timeout=300.0,
    )
    r.raise_for_status()
    return r.json()["embeddings"]
