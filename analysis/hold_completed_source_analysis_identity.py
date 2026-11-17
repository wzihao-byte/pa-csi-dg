"""Expose a verified completed source-analysis identity during chain recovery."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--compatibility-marker", required=True)
    parser.add_argument("--seconds", type=int, default=120)
    args = parser.parse_args()
    if args.compatibility_marker != "analysis/continue_source_analyses.py":
        raise RuntimeError("Unexpected compatibility marker")
    completion_path = args.source_output.resolve() / "completion.json"
    with completion_path.open(encoding="utf-8-sig") as stream:
        completion = json.load(stream)
    if completion.get("completed_count") != 117 or completion.get("target_data_loaded"):
        raise RuntimeError("Source analysis completion is not the frozen 117-job source-only result")
    time.sleep(args.seconds)


if __name__ == "__main__":
    main()
