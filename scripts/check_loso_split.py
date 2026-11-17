from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dg_dataset import build_split_indices, build_split_manifest, load_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check DG LOSO inner-validation split composition.")
    parser.add_argument("--config", required=True, help="Path to a DG JSON config.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for deterministic split paths.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_loso_manifest(manifest: Mapping[str, Any]) -> None:
    target_env = str(manifest["target_env"])
    source_envs = set(str(env) for env in manifest["source_envs"])
    train_env_counts = manifest["train"]["env_counts"]
    val_env_counts = manifest["val"]["env_counts"]
    test_env_counts = manifest["test"]["env_counts"]

    if manifest.get("inner_validation") != "loso":
        raise AssertionError(f"Expected inner_validation='loso', found {manifest.get('inner_validation')!r}")
    if len(val_env_counts) != 1:
        raise AssertionError(f"Expected one validation source env, found {val_env_counts}")

    val_env = next(iter(val_env_counts))
    if val_env == target_env:
        raise AssertionError("Target environment leaked into validation.")
    if val_env not in source_envs:
        raise AssertionError(f"Validation environment {val_env!r} is not in source_envs={sorted(source_envs)}")
    if val_env in train_env_counts:
        raise AssertionError(f"Validation environment {val_env!r} leaked into training.")
    if target_env in train_env_counts:
        raise AssertionError("Target environment leaked into training.")
    if set(test_env_counts) != {target_env}:
        raise AssertionError(f"Test split must contain only target {target_env!r}, found {test_env_counts}")


def format_env_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{env}:{count}" for env, count in sorted(counts.items()))


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    split_config = config.get("split", {}) or {}
    dataset = load_dataset(config["data"], REPO_ROOT)

    target_envs = list(dataset.env_names) if config.get("target_env") in (None, "all") else [config["target_env"]]
    print(f"config={config_path}")
    print(f"seed={args.seed}")
    print(f"dataset_size={len(dataset)} envs={dataset.env_names}")
    print("target  source_envs  train_count  train_envs  val_count  val_envs  test_count  test_envs")
    for target_env in target_envs:
        split = build_split_indices(
            dataset,
            mode=config.get("mode", "dg_loeo"),
            seed=args.seed,
            val_ratio=float(split_config.get("val_ratio", 0.2)),
            test_ratio=float(split_config.get("test_ratio", 0.2)),
            target_env=str(target_env),
            inner_validation=str(split_config.get("inner_validation", "stratified_holdout")),
            loso_validation_env=split_config.get("loso_validation_env"),
        )
        manifest = build_split_manifest(dataset, split)
        assert_loso_manifest(manifest)
        print(
            f"{target_env}  "
            f"{','.join(manifest['source_envs'])}  "
            f"{manifest['train']['count']}  "
            f"{format_env_counts(manifest['train']['env_counts'])}  "
            f"{manifest['val']['count']}  "
            f"{format_env_counts(manifest['val']['env_counts'])}  "
            f"{manifest['test']['count']}  "
            f"{format_env_counts(manifest['test']['env_counts'])}"
        )
    print("LOSO split check passed")


if __name__ == "__main__":
    main()

