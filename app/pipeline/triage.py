"""URL triage: heuristic noise filtering + normalization. No LLM needed here."""
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

NOISE_PATH = re.compile(
    r"/(tag|tags|author|authors|category|categories|page/\d+|feed|feeds|"
    r"comments|wp-json|wp-admin|cart|checkout|search|login|logout|register|"
    r"account|share|print|amp|cdn-cgi|wp-includes)(/|$|\?)",
    re.IGNORECASE,
)
NOISE_EXT = re.compile(
    r"\.(jpe?g|png|gif|webp|svg|ico|css|js|xml|json|pdf|zip|tar|gz|mp[34]|"
    r"woff2?|ttf|eot|ics|csv|xlsx?|docx?|pptx?)$",
    re.IGNORECASE,
)
TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term",
                   "utm_content", "fbclid", "gclid", "ref", "replytocom", "share"}


def normalize_url(url: str) -> str | None:
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https"):
        return None
    query = urlencode(sorted(
        (k, v) for k, v in parse_qsl(parts.query) if k.lower() not in TRACKING_PARAMS
    ))
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def triage(urls: list[str]) -> tuple[list[str], dict[str, str]]:
    """Returns (accepted sorted list, {url: reason} rejected map)."""
    seen: set[str] = set()
    accepted: list[str] = []
    rejected: dict[str, str] = {}
    for raw in urls:
        url = normalize_url(raw)
        if url is None:
            rejected[raw] = "non-http-scheme"
            continue
        if url in seen:
            continue
        seen.add(url)
        path = urlsplit(url).path
        if NOISE_EXT.search(path):
            rejected[url] = "binary-or-feed-extension"
        elif NOISE_PATH.search(path):
            rejected[url] = "noise-path"
        else:
            accepted.append(url)
    return sorted(accepted), rejected
