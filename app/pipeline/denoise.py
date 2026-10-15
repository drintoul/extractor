"""Deterministic, rule-based markdown cleaning. Runs before any LLM call."""
import re

_IMG_ONLY = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$")
_LINK_ONLY = re.compile(r"^\s*\[[^\]]*\]\([^)]*\)\s*$")
_BOILERPLATE = re.compile(
    r"(subscribe|newsletter|sign up for|cookie|all rights reserved|"
    r"share (this|on)|follow us|terms of (service|use)|privacy policy)",
    re.IGNORECASE,
)


def clean_markdown(md: str) -> str:
    out: list[str] = []
    link_run = 0
    for line in md.splitlines():
        if _IMG_ONLY.match(line):
            continue
        if _BOILERPLATE.search(line):
            continue
        if _LINK_ONLY.match(line):
            link_run += 1
            # Runs of link-only lines are nav/sidebar/related-posts residue
            if link_run >= 3:
                continue
        else:
            link_run = 0
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def word_count(text: str) -> int:
    return len(text.split())
