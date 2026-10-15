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
    return sorted(accepted), rejected
