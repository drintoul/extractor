"""Sitemap discovery fallback — for sites Firecrawl /map under-discovers.

Used when /v1/map returns near-nothing: reads robots.txt Sitemap: directives
and probes common sitemap paths (covers BigCommerce xmlsitemap.php, WP
sitemap_index.xml, etc.), follows <sitemapindex> into child <urlset>s.
Deterministic, cheap, no JS needed.
"""
import xml.etree.ElementTree as ET
from urllib.parse import urlsplit, urlunsplit

import httpx

from .config import SETTINGS

_SITEMAP_PATHS = [
    "/sitemap.xml", "/sitemap_index.xml", "/xmlsitemap.php",
    "/sitemap.php", "/sitemap/", "/sitemaps.xml",
]
_MAX_SITEMAPS = 30          # cap recursion across sitemapindex children
_TIMEOUT = httpx.Timeout(20.0)


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_sitemap(text: str) -> tuple[str | None, list[str]]:
    """Returns ('sitemapindex'|'urlset'|None, [locs])."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None, []
    locs = [e.text.strip() for e in root.iter()
            if _localname(e.tag) == "loc" and e.text]
    return _localname(root.tag), locs


def _robots_sitemaps(base: str) -> list[str]:
    try:
        r = httpx.get(f"{base}/robots.txt", timeout=_TIMEOUT, follow_redirects=True)
        if r.status_code != 200:
            return []
        return [line.split(":", 1)[1].strip() for line in r.text.splitlines()
                if line.lower().startswith("sitemap:")]
    except Exception:
        return []


def discover(seed_url: str, cap: int) -> dict:
    """Collect page URLs from the site's sitemaps.

    Returns {"urls": [...], "skipped_catalog_urls": int}. Sitemaps matching
    SITEMAP_SKIP_PATTERNS are still fetched — count-only — so the run can
    report how large the excluded product catalog is (fit signal)."""
    parts = urlsplit(seed_url)
    base = urlunsplit((parts.scheme, parts.netloc, "", "", ""))
    queue: list[tuple[str, bool]] = [
        (u, False) for u in
        _robots_sitemaps(base) + [base + p for p in _SITEMAP_PATHS]]
    skip = [s.strip().lower() for s in SETTINGS.sitemap_skip_patterns.split(",")
            if s.strip()]
    seen: set[str] = set()
    urls: list[str] = []
    skipped = 0
    while queue and len(urls) < cap and len(seen) < _MAX_SITEMAPS:
        sm, count_only = queue.pop(0)
        if sm in seen:
            continue
        seen.add(sm)
        try:
            r = httpx.get(sm, timeout=_TIMEOUT, follow_redirects=True)
        except Exception:
            continue
        if r.status_code != 200:
            continue
        kind, locs = _parse_sitemap(r.text)
        if kind == "sitemapindex":
            queue.extend(
                (l, any(p in l.lower() for p in skip)) for l in locs)
        elif kind == "urlset":
            if count_only:
                skipped += len(locs)
            else:
                urls.extend(locs)
    return {"urls": urls[:cap], "skipped_catalog_urls": skipped}
