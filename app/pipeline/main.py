"""CLI: python -m pipeline.main run <seed_url> [--run-id NAME]
       python -m pipeline.main serve            (UI + API on :8000)

Same run-id resumes from the last checkpoint (no re-scrapes); a new run-id
starts a fresh deterministic run.
"""
import argparse
import sys

from .graph import build_checkpointer, build_graph
from .ui import _default_run_id  # noqa: F401  (reused by CLI)


def main() -> int:
    ap = argparse.ArgumentParser(prog="pipeline")
    sub = ap.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="run the extraction pipeline once")
    run.add_argument("url", help="Seed URL to map and extract")
    run.add_argument("--run-id", default=None,
                     help="Checkpoint thread id (default: derived from URL)")

    sub.add_parser("serve", help="serve the UI + API on port 8000")
    args = ap.parse_args()

    if args.cmd == "run":
        import re
        if not re.match(r"^https?://", args.url):
            ap.error("url must start with http:// or https://")
        run_id = args.run_id or _default_run_id(args.url)
        graph = build_graph(checkpointer=build_checkpointer())
        result = graph.invoke(
            {"seed_url": args.url, "run_id": run_id,
             "pages": {}, "rejected_urls": {}},
            config={"configurable": {"thread_id": run_id}},
        )
        import json
        print(json.dumps(result["manifest"], indent=2))
        return 0

    if args.cmd in ("serve", None):
        import uvicorn
        uvicorn.run("pipeline.ui:app", host="0.0.0.0", port=8000)
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
