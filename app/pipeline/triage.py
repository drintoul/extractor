"""URL triage: heuristic noise filtering + normalization. No LLM needed here."""
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

NOISE_PATH = re.compile(
    r"/(tag|tags|author|authors|category|categories|page/\d+|feed|feeds|"
    r"comments|wp-json|wp-admin|cart|checkout|search|login|logout|register|"
    r"account|share|print|amp|cdn-cgi|wp-includes|wishlist|orderstatus)"
    r"(/|$|\?|\.php$)",   # .php boundary catches BigCommerce-style cart.php etc.
    re.IGNORECASE,
)
NOISE_EXT = re.compile(
    r"\.(jpe?g|png|gif|webp|svg|ico|css|js|xml|json|pdf|zip|tar|gz|mp[34]|"
    r"woff2?|ttf|eot|ics|csv|xlsx?|docx?|pptx?)$",
    re.IGNORECASE,
)
# Pure date listings — /2024 or /2024/05 (path ends at the date). Deliberately
# does NOT match /2024/05/post-slug — real dated posts are content.
DATE_PATH = re.compile(r"^/(?:19|20)\d{2}(/\d{2})?$")

# Leading locale path segments — /en/faq, /en-us/guide, /gbr/deals. Sites that
# publish per-locale copies of every page otherwise produce N near-duplicate
# scrapes (different currency/phone numbers defeat content dedupe).
_LOCALE_SEGS = {
    "en", "en-us", "en-gb", "en-ca", "en-au", "en-nz", "en-ie",
    "us", "usa", "uk", "gb", "gbr", "au", "aus", "nz", "ca", "can", "ie",
    "es", "es-es", "fr", "fr-fr", "de", "de-de", "it", "it-it",
    "nl", "nl-nl", "pt", "pt-br", "zh", "zh-cn", "ja", "ko",
}
# Representative preference: English locales first (embeddings are bge-large-en),
# then English-speaking country codes, then everything else.
_LOCALE_RANK = {s: 0 for s in
                ("en", "en-us", "en-gb", "en-ca", "en-au", "en-nz", "en-ie")}
_LOCALE_RANK.update({s: 1 for s in ("us", "usa", "ca", "au", "uk", "gbr", "nz", "ie")})


def _fold_key(url: str) -> tuple[str, str | None]:
    """(fold key, locale segment). The key drops a leading locale segment so
    /usa/faq/x and /gbr/faq/x fold to the same key as /faq/x."""
    parts = urlsplit(url)
    seg = parts.path.lstrip("/").split("/", 1)
    first = seg[0].lower() if seg[0] else ""
    if first in _LOCALE_SEGS:
        rest = "/" + (seg[1] if len(seg) > 1 else "")
        return urlunsplit((parts.scheme, parts.netloc, rest, parts.query, "")), first
    return url, None

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


def triage(urls: list[str],
           extra_exclude: str = "") -> tuple[list[str], dict[str, str]]:
    """Returns (accepted sorted list, {url: reason} rejected map).

    extra_exclude: optional regex applied to the URL path — rejects e.g.
    /products/|/product/ on platforms where product URLs have no other
    discriminating signal.
    """
    user_ex = re.compile(extra_exclude, re.IGNORECASE) if extra_exclude else None
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
        elif DATE_PATH.match(path):
            rejected[url] = "date-archive"
        elif user_ex and user_ex.search(path):
            rejected[url] = "user-excluded"
        else:
            accepted.append(url)

    # Locale folding: URLs that differ only by a leading locale segment are the
    # same page — keep one representative per fold key.
    groups: dict[str, list[tuple[int, str]]] = {}
    for url in accepted:
        key, seg = _fold_key(url)
        rank = 0 if seg is None else 1 + _LOCALE_RANK.get(seg, 2)
        groups.setdefault(key, []).append((rank, url))
    kept: list[str] = []
    for key, cands in groups.items():
        cands.sort()
        kept.append(cands[0][1])
        for _, dup in cands[1:]:
            rejected[dup] = "locale-fold"
    return sorted(kept), rejected
