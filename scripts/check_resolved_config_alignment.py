from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ALIGNMENT_KEYS = [
    ("training", "lr"),
    ("training", "optimizer"),
    ("training", "epochs"),
    ("training", "patience"),
    ("training", "weight_decay"),
    ("training", "grad_clip_norm"),
    ("training", "batch_size"),
    ("training", "sampler"),
    ("training", "selection_metric"),
    ("training", "lr_scheduler"),
    ("data", "unwrap_phase"),
    ("split", "inner_validation"),
    ("losses", "temperature"),
    ("losses", "lambda_supcon"),
    ("losses", "lambda_supcon_warmup_epochs"),
    ("losses", "include_same_domain_same_class"),
    ("losses", "supcon_positive_mode"),
    ("losses", "contrastive_loss_type"),
    ("losses", "lambda_pair_margin"),
    ("model", "prototype_num_subcenters"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check resolved_config.json alignment across DG runs.")
    parser.add_argument("--output-root", required=True, help="Output root containing run directories.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def nested_get(config: Mapping[str, Any], path: Iterable[str]) -> Any:
    cursor: Any = config
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return "<MISSING>"
        cursor = cursor[key]
    return cursor


def value_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    paths = sorted(output_root.rglob("resolved_config.json"))
    if not paths:
        raise SystemExit(f"No resolved_config.json files found under {output_root}")

    records = [(path, read_json(path)) for path in paths]
    print(f"checked_configs={len(records)}")
    drift_found = False
    for key_path in ALIGNMENT_KEYS:
        values: dict[str, list[str]] = defaultdict(list)
        for path, config in records:
            values[value_key(nested_get(config, key_path))].append(str(path))

        label = ".".join(key_path)
        parsed_values = [json.loads(value) for value in values]
        print(f"{label}: {parsed_values}")
        if len(values) > 1:
            drift_found = True
            print(f"DRIFT {label}")
            for value, value_paths in values.items():
                print(f"  {json.loads(value)!r}: {len(value_paths)} run(s)")
                for value_path in value_paths:
                    print(f"    {value_path}")

    if drift_found:
        raise SystemExit("Resolved-config alignment failed.")
    print("Resolved-config alignment passed.")


if __name__ == "__main__":
    main()

