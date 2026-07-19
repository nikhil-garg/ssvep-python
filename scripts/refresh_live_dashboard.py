"""Regenerate the local dashboard periodically and add a browser auto-refresh."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from ssvep_toolkit.dashboard import render_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("outputs/registry/experiments.sqlite3"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dashboard/index.html"))
    parser.add_argument("--examples", type=Path, default=Path("outputs/examples/neuron_behavior"))
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()
    if args.interval < 5:
        raise ValueError("refresh interval must be at least five seconds")
    while True:
        target = render_dashboard(args.database, args.output, example_directory=args.examples)
        page = target.read_text(encoding="utf-8")
        marker = f'<meta http-equiv="refresh" content="{args.interval:g}">'
        if marker not in page:
            page = page.replace("<head>", f"<head>{marker}", 1)
            target.write_text(page, encoding="utf-8")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
