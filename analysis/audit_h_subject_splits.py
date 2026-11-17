"""Reconstruct the exact hashed H splits and attach newly verified identities."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.revision_evidence import H_BASE, H10, H11, OUT, read_json, sha256, write_json
from dg_dataset import _hash_indices, safe_stratified_split


def main():
    identity = read_json(OUT / "verified_subject_audit.json")
    assert identity["all_9000_verified"]
    sidecars = {env: read_json(OUT / f"subject_sidecars/{env}.json") for env in ("E1", "E2", "E3")}
    arrays = read_json(OUT / "array_audit.json")["environments"]
    labels = {env: np.load(info["files"]["label"]["path"], allow_pickle=False) for env, info in arrays.items()}
    runs = [r for r in read_json(H_BASE)["rows"] if r["arm"] in ("H00", "H01")]
    runs += read_json(H10)["rows"] + read_json(H11)["rows"]
    reports = []
    for run in runs:
        folder = Path(run["attempt_dir"])
        completion_path = folder / "attempt_completed.json"
        assert sha256(completion_path) == run["completion_manifest_sha256"]
        completion = read_json(completion_path)
        split_path = folder / "payload/run/split_manifest.json"
        config_path = folder / "payload/run/resolved_config.json"
        for path in (split_path, config_path):
            assert sha256(path) == completion["artifacts"][path.relative_to(folder).as_posix()]["sha256"]
        split, config = read_json(split_path), read_json(config_path)
        parts = {"train": [], "val": [], "test": []}
        for env in split["source_envs"]:
            for kind in ("amp", "phase", "label"):
                assert Path(config["data"]["env_files"][env][kind]).resolve() == Path(arrays[env]["files"][kind]["path"]).resolve()
            env_number = ("E1", "E2", "E3").index(env)
            train, val = safe_stratified_split(np.arange(3000), labels[env], 0.2, int(run["seed"]) + env_number + 1)
            for key, indices in (("train", train), ("val", val)):
                parts[key].extend((env, int(index), env_number * 3000 + int(index)) for index in indices)
        target = run["target_env"]
        target_number = ("E1", "E2", "E3").index(target)
        parts["test"] = [(target, i, target_number * 3000 + i) for i in range(3000)]
        for key, rows in parts.items():
            assert _hash_indices([row[2] for row in rows]) == split[key]["global_sample_id_sha256"]
            assert len(rows) == split[key]["count"]
        identities = {key: {
            "sample": {f"{env}:{index:04d}" for env, index, _ in rows},
            "subject": {sidecars[env]["subject_ids"][index] for env, index, _ in rows},
            "acquisition": {sidecars[env]["acquisition_ids"][index] for env, index, _ in rows},
        } for key, rows in parts.items()}
        overlaps = {field: {f"{a}_{b}": sorted(identities[a][field] & identities[b][field])
                            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
                    for field in ("sample", "subject", "acquisition")}
        assert not any(overlaps["sample"].values())
        assert not any(overlaps[field][pair] for field in ("subject", "acquisition") for pair in ("train_test", "val_test"))
        reports.append({"arm": run["arm"], "target": target, "seed": run["seed"],
                        "split_manifest_sha256": sha256(split_path), "all_three_sample_id_hashes_match": True,
                        "train_val_shared_subjects": len(overlaps["subject"]["train_val"]),
                        "train_val_shared_acquisitions": len(overlaps["acquisition"]["train_val"]),
                        "target_identity_overlap": False, "overlaps": overlaps})
    assert len(reports) == 36
    result = {"runs_checked": 36, "all_sample_membership_hashes_verified": True,
              "shared_source_subject_count_range": [min(r["train_val_shared_subjects"] for r in reports), max(r["train_val_shared_subjects"] for r in reports)],
              "shared_source_acquisition_count_range": [min(r["train_val_shared_acquisitions"] for r in reports), max(r["train_val_shared_acquisitions"] for r in reports)],
              "all_targets_identity_disjoint": True, "runs": reports,
              "interpretation": "Source validation shares subjects/acquisition families with source training. Target isolation remains intact. H-Y score differences still cannot identify a single causal factor."}
    write_json(OUT / "h_subject_split_audit.json", result)
    print({key: value for key, value in result.items() if key != "runs"})


if __name__ == "__main__":
    main()
