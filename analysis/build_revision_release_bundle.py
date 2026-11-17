"""Build a hash-bound, compact reproducibility package after every formal gate.

The package contains configurations, identity/split manifests, exact saved
predictions, histories, event records, source snapshots and evidence. Training
and resume checkpoints are not duplicated; their recorded SHA-256 values are
indexed so a separate checkpoint deposit can be verified without inflating the
compact release.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("PA_CSI_DG_ROOT", Path(__file__).resolve().parents[1])).resolve()
EXPECTED_STUDIES = {
    "r_core_v1": 36,
    "r_controls_v1": 81,
    "r_inputs_v1": 63,
    "r_widar_v2": 36,
}
RUN_FILES = (
    "config.json",
    "runtime.json",
    "events.jsonl",
    "history.json",
    "selected_source_predictions.json",
    "status.json",
)
EVALUATION_FILES = ("result.json", "predictions.json")
SOURCE_SUFFIXES = {".py", ".md", ".tex", ".bib", ".json", ".csv", ".sty", ".cls", ".bst"}
SOURCE_ROOTS = (
    "analysis",
    "configs",
    "experiments/revision_validation",
    "experiments/supcon_tpp_joint",
    "experiments/supcon_tpp_legacy_reproduction",
    "experiments/tpp_isolated",
    "paper/sensors_manuscript",
)
ROOT_SOURCE_FILES = ("dg_dataset.py", "dg_losses.py", "dg_models.py", "train_dg.py", "AGENTS.md")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path, predicate=lambda _path: True) -> int:
    count = 0
    for path in sorted(source.rglob("*")):
        if path.is_file() and predicate(path):
            copy_file(path, destination / path.relative_to(source))
            count += 1
    return count


def verify_manifest(directory: Path, manifest_name: str = "manifest.json") -> dict:
    manifest = read(directory / manifest_name)
    entries = manifest.get("files_sha256", manifest.get("artifacts_sha256"))
    if not isinstance(entries, dict):
        raise RuntimeError(f"No file hash table in {directory / manifest_name}")
    for name, expected in entries.items():
        path = directory / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Manifest mismatch: {path}")
    return manifest


def verify_formal_gate(evidence: Path, assets: Path, final_qa: Path) -> dict:
    report = read(evidence / "verified_formal_results.json")
    if (
        report.get("formal_run_count") != 216
        or report.get("formal_source_analysis_count") != 117
        or not report.get("all_formal_studies_completed")
        or not report.get("target_gates_verified")
        or report.get("population_inference_performed")
    ):
        raise RuntimeError("Formal evidence gate is incomplete")
    verify_manifest(evidence)
    asset_manifest = verify_manifest(assets)
    fragment = read(assets / "formal_fragment_manifest.json")
    if fragment.get("verified_formal_results_sha256") != sha256(evidence / "verified_formal_results.json"):
        raise RuntimeError("Formal fragment is not bound to the verified evidence")
    if fragment.get("formal_asset_manifest_sha256") != sha256(assets / "manifest.json"):
        raise RuntimeError("Formal fragment is not bound to the paper asset manifest")
    for name, expected in fragment["files_sha256"].items():
        if sha256(assets / name) != expected:
            raise RuntimeError(f"Formal fragment mismatch: {name}")
    qa = read(final_qa)
    manuscript = ROOT / "paper/sensors_manuscript/manuscript.pdf"
    if (
        qa.get("status") != "completed"
        or not qa.get("all_pages_visually_inspected")
        or qa.get("undefined_references_or_citations") != 0
        or qa.get("overfull_boxes") != 0
        or qa.get("pdf_sha256") != sha256(manuscript)
    ):
        raise RuntimeError("Final manuscript compilation/visual gate is incomplete")
    return {"evidence": report, "assets": asset_manifest, "final_qa": qa}


def epoch_schedule(rows: list[dict], seed: int, epoch: int, steps: int) -> np.ndarray:
    """Frozen R-family sampler, duplicated here and verified against every history."""
    rng = random.Random(int(seed) * 1_000_003 + int(epoch))
    environments = sorted({row["environment"] for row in rows})
    if len(environments) != 2:
        raise RuntimeError("R-family release schedules require two source environments")
    pools = {(env, label): [i for i, row in enumerate(rows) if row["environment"] == env and row["label"] == label]
             for env in environments for label in range(6)}
    if not all(pools.values()):
        raise RuntimeError("A source environment/class sampler pool is empty")
    for pool in pools.values():
        rng.shuffle(pool)
    offsets = {key: 0 for key in pools}
    schedule = []
    for _ in range(steps):
        batch = []
        for env in environments:
            classes = list(range(6))
            rng.shuffle(classes)
            for label in classes + classes[:2]:
                key, pool = (env, label), pools[(env, label)]
                while True:
                    if offsets[key] == len(pool):
                        rng.shuffle(pool)
                        offsets[key] = 0
                    index = pool[offsets[key]]
                    offsets[key] += 1
                    if index not in batch:
                        batch.append(index)
                        break
        rng.shuffle(batch)
        if len(batch) != len(set(batch)) or len(batch) != 16:
            raise RuntimeError("Invalid frozen sampler batch")
        schedule.append(batch)
    return np.asarray(schedule, dtype="<i8")


def schedule_hash(schedule: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(schedule, dtype="<i8").tobytes()).hexdigest()


def copy_formal_study(source: Path, destination: Path, expected_jobs: int, schedules: Path) -> tuple[list[dict], list[dict]]:
    protocol = read(source / "protocol.json")
    jobs = protocol.get("jobs", [])
    if len(jobs) != expected_jobs:
        raise RuntimeError(f"{source.name}: expected {expected_jobs} registered jobs, found {len(jobs)}")
    if not (source / "target_gate.json").is_file():
        raise RuntimeError(f"{source.name}: target gate is missing")
    summary = read(source / "evaluation/summary.json")
    if len(summary.get("runs", [])) != expected_jobs:
        raise RuntimeError(f"{source.name}: incomplete target evaluation")
    for relative, expected in summary.get("artifacts_sha256", {}).items():
        if sha256(source / relative) != expected:
            raise RuntimeError(f"{source.name}: evaluation artifact mismatch: {relative}")

    for name in ("protocol.json", "bundle_manifest.json", "target_gate.json"):
        copy_file(source / name, destination / name)
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix in {".json", ".log"} and path.name not in {
            "protocol.json", "bundle_manifest.json", "target_gate.json", "runner.lock.json", "queue_status.json"
        }:
            copy_file(path, destination / path.name)
    copy_tree(source / "code", destination / "code")
    copy_tree(source / "manifests", destination / "manifests")
    copy_file(source / "evaluation/summary.json", destination / "evaluation/summary.json")

    checkpoints, schedule_index = [], []
    grouped: dict[tuple, list[tuple[dict, Path, list[dict]]]] = defaultdict(list)
    for config in jobs:
        job_id = config["job_id"]
        run = source / "runs" / job_id
        status = read(run / "status.json")
        if status.get("status") != "completed" or status.get("optimizer_updates") != config["epochs"] * config["updates_per_epoch"]:
            raise RuntimeError(f"{source.name}/{job_id}: incomplete source training")
        for name in RUN_FILES:
            copy_file(run / name, destination / "runs" / job_id / name)
        evaluation = source / "evaluation" / job_id
        for name in EVALUATION_FILES:
            copy_file(evaluation / name, destination / "evaluation" / job_id / name)
        result = read(evaluation / "result.json")
        if not result.get("source_predictions_reproduced"):
            raise RuntimeError(f"{source.name}/{job_id}: source prediction reproduction not verified")
        checkpoint_hash = result.get("checkpoint_sha256")
        if checkpoint_hash != status.get("artifacts_sha256", {}).get("best_model.pt"):
            raise RuntimeError(f"{source.name}/{job_id}: checkpoint hashes disagree")
        checkpoint_path = run / "best_model.pt"
        if sha256(checkpoint_path) != checkpoint_hash:
            raise RuntimeError(f"{source.name}/{job_id}: checkpoint file differs from its recorded hash")
        checkpoints.append({"study": source.name, "job_id": job_id, "checkpoint_sha256": checkpoint_hash,
                            "checkpoint_bytes": checkpoint_path.stat().st_size,
                            "checkpoint_in_compact_release": False})

        manifest_path = Path(config["source_manifest_path"])
        if sha256(manifest_path) != config["source_manifest_sha256"]:
            raise RuntimeError(f"{source.name}/{job_id}: source manifest changed")
        manifest = read(manifest_path)
        train_hash = value_hash(manifest["train"])
        key = (train_hash, int(config["seed"]), int(config["epochs"]), int(config["updates_per_epoch"]))
        grouped[key].append((config, run, manifest["train"]))

    for number, (key, records) in enumerate(sorted(grouped.items()), start=1):
        train_hash, seed, epochs, steps = key
        arrays, epoch_hashes = [], []
        for epoch in range(epochs):
            array = epoch_schedule(records[0][2], seed, epoch, steps)
            arrays.append(array)
            epoch_hashes.append(schedule_hash(array))
        for config, run, _rows in records:
            history = read(run / "history.json")
            recorded = [row["schedule_sha256"] for row in history]
            if recorded != epoch_hashes:
                raise RuntimeError(f"{source.name}/{config['job_id']}: regenerated sampler schedule differs")
        schedule_array = np.stack(arrays)
        stem = f"schedule_{number:03d}_seed_{seed}"
        npz_path = schedules / source.name / f"{stem}.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, train_positions=schedule_array)
        metadata = {
            "study": source.name,
            "seed": seed,
            "epochs": epochs,
            "updates_per_epoch": steps,
            "batch_size": 16,
            "dtype": "<i8",
            "positions_reference": "zero-based positions in the included source manifest train list",
            "train_rows_value_sha256": train_hash,
            "epoch_schedule_sha256": epoch_hashes,
            "npz_sha256": sha256(npz_path),
            "jobs": [record[0]["job_id"] for record in records],
        }
        write(npz_path.with_suffix(".json"), metadata)
        schedule_index.append(metadata)
    return checkpoints, schedule_index


def historical_h_rows() -> list[tuple[str, dict]]:
    specifications = (
        ("factorial_s32", ROOT / "outputs_supcon_tpp_legacy_reproduction/factorial_s32/factorial_results.json", {"H00", "H01"}),
        ("l1248_h10_v1", ROOT / "outputs_tpp_level8_ablation/l1248_h10_v1/level8_results.json", {"H10"}),
        ("l1248_v1", ROOT / "outputs_tpp_level8_ablation/l1248_v1/level8_results.json", {"H11"}),
    )
    rows = []
    for label, path, arms in specifications:
        result = read(path)
        selected = [row for row in result["rows"] if row["arm"] in arms]
        copy_count = 18 if len(arms) == 2 else 9
        if len(selected) != copy_count:
            raise RuntimeError(f"Historical H result count differs in {path}")
        rows.extend((label, row) for row in selected)
    return rows


def verify_historical_h_checkpoints(rows: list[tuple[str, dict]]) -> list[dict]:
    verified = []
    for label, row in rows:
        job_id = row.get("job_id") or f"{row['target_env']}_{row['arm']}_{row['seed']}"
        run = Path(row["attempt_dir"]) / "payload/run"
        checkpoint = run / "best_model.pt"
        metrics = read(run / "metrics.json")
        expected = row.get("checkpoint_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeError(f"Historical H checkpoint hash is missing: {job_id}")
        if metrics.get("checkpoint_sha256") != expected or sha256(checkpoint) != expected:
            raise RuntimeError(f"Historical H checkpoint hashes disagree: {job_id}")
        verified.append({"protocol": "H_historical", "source_group": label, "job_id": job_id,
                         "checkpoint_sha256": expected, "checkpoint_bytes": checkpoint.stat().st_size})
    return verified


def copy_historical_h(destination: Path) -> dict:
    result_sources = (
        ROOT / "outputs_supcon_tpp_legacy_reproduction/factorial_s32/factorial_results.json",
        ROOT / "outputs_tpp_level8_ablation/l1248_h10_v1/level8_results.json",
        ROOT / "outputs_tpp_level8_ablation/l1248_v1/level8_results.json",
    )
    for source in result_sources:
        copy_file(source, destination / "summaries" / source.parent.name / source.name)
    rows = historical_h_rows()
    verified_checkpoints = {item["job_id"]: item for item in verify_historical_h_checkpoints(rows)}
    checkpoint_index = []
    for label, row in rows:
        job_id = row.get("job_id") or f"{row['target_env']}_{row['arm']}_{row['seed']}"
        attempt = Path(row["attempt_dir"])
        run = attempt / "payload/run"
        for name in ("attempt_completed.json", "attempt_seal.json", "attempt_started.json", "job_config.json"):
            if (attempt / name).is_file():
                copy_file(attempt / name, destination / "runs" / label / job_id / name)
        for name in ("resolved_config.json", "sampler_schedule.json", "split_manifest.json", "confusion_matrix.csv",
                     "metrics.json", "target_predictions.jsonl", "target_access_audit.json"):
            if (run / name).is_file():
                copy_file(run / name, destination / "runs" / label / job_id / name)
        checkpoint = verified_checkpoints[job_id]
        checkpoint_index.append({"protocol": "H_historical", "job_id": job_id,
                                 "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                                 "checkpoint_bytes": checkpoint["checkpoint_bytes"],
                                 "checkpoint_in_compact_release": False})
    return {"runs": len(rows), "checkpoint_index": checkpoint_index,
            "sampler_evidence": "Recorded global sample-sequence hash and frozen generator; historical full index lists were not stored."}


def historical_y_rows() -> list[dict]:
    root = ROOT / "outputs_supcon_tpp_joint/formal_interim_v12"
    with (root / "run_registry.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    if len(records) != 36 or any(row["status"] != "completed" for row in records):
        raise RuntimeError("Historical Y registry is not the completed 36-run record")
    return records


def verify_historical_y_checkpoints(records: list[dict]) -> list[dict]:
    verified = []
    for row in records:
        source = Path(row["output_dir"])
        checkpoint = source / "best_model.pt"
        expected = row.get("best_model_file_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeError(f"Historical Y checkpoint hash is missing: {row.get('job_id')}")
        if sha256(checkpoint) != expected:
            raise RuntimeError(f"Historical Y checkpoint hash differs: {row['job_id']}")
        verified.append({"protocol": "Y_grouped", "job_id": row["job_id"],
                         "checkpoint_sha256": expected, "checkpoint_bytes": checkpoint.stat().st_size})
    return verified


def copy_historical_y(destination: Path) -> dict:
    generated = ROOT / "experiments/supcon_tpp_joint/generated_formal_interim_v12"
    copy_tree(generated, destination / "generated_protocol")
    root = ROOT / "outputs_supcon_tpp_joint/formal_interim_v12"
    for name in ("formal_results.json", "run_registry.csv", "formal_launch_audit.json", "execution_lock.json"):
        copy_file(root / name, destination / name)
    records = historical_y_rows()
    verified_checkpoints = {item["job_id"]: item for item in verify_historical_y_checkpoints(records)}
    checkpoint_index = []
    for row in records:
        source = Path(row["output_dir"])
        job_id = row["job_id"]
        for path in sorted(source.iterdir()):
            if path.is_file() and path.suffix != ".pt":
                copy_file(path, destination / "runs" / job_id / path.name)
        checkpoint = verified_checkpoints[job_id]
        checkpoint_index.append({"protocol": "Y_grouped", "job_id": job_id,
                                 "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                                 "checkpoint_bytes": checkpoint["checkpoint_bytes"],
                                 "checkpoint_in_compact_release": False})
    return {"runs": len(records), "checkpoint_index": checkpoint_index,
            "sampler_evidence": "Frozen deterministic generator, split records, seed, first-batch IDs and per-epoch hashes; historical full index lists were not stored."}


def source_candidates() -> list[Path]:
    paths = []
    for name in ROOT_SOURCE_FILES:
        path = ROOT / name
        if path.is_file():
            paths.append(path)
    for relative in SOURCE_ROOTS:
        directory = ROOT / relative
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            parts = set(path.relative_to(ROOT).parts)
            if parts & {"__pycache__", ".pytest_cache"}:
                continue
            if path.name in {"manuscript.aux", "manuscript.log", "manuscript.out"}:
                continue
            paths.append(path)
    return sorted(set(paths))


def git_metadata() -> dict:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
    return {
        "base_git_head": head,
        "base_git_worktree_clean": not status,
        "base_git_status_entry_count": len(status),
        "version_rule": "The release file manifest is authoritative. The base Git HEAD alone does not identify this revision state.",
        "public_repository_or_archive_identifier": "AUTHOR_TO_SUPPLY_BEFORE_SUBMISSION",
    }


def verify_completed_schedule_records(runtime: Path) -> list[dict]:
    verified = []
    for study_name in EXPECTED_STUDIES:
        study = runtime / "studies" / study_name
        if not study.exists():
            continue
        for config in read(study / "protocol.json").get("jobs", []):
            run = study / "runs" / config["job_id"]
            events_path = run / "events.jsonl"
            if not events_path.is_file():
                continue
            events = [json.loads(line) for line in events_path.read_text(encoding="utf-8-sig").splitlines() if line]
            if not any(event.get("event") == "completed" for event in events):
                continue
            status = read(run / "status.json")
            if status.get("status") != "completed":
                raise RuntimeError(f"Completed event has no completed status: {study_name}/{config['job_id']}")
            checkpoint = run / "best_model.pt"
            expected_checkpoint = status.get("artifacts_sha256", {}).get("best_model.pt")
            if not isinstance(expected_checkpoint, str) or len(expected_checkpoint) != 64:
                raise RuntimeError(f"Completed checkpoint hash is missing: {study_name}/{config['job_id']}")
            if sha256(checkpoint) != expected_checkpoint:
                raise RuntimeError(f"Completed checkpoint file differs: {study_name}/{config['job_id']}")
            manifest_path = Path(config["source_manifest_path"])
            if sha256(manifest_path) != config["source_manifest_sha256"]:
                raise RuntimeError(f"Source manifest changed: {study_name}/{config['job_id']}")
            rows = read(manifest_path)["train"]
            history = read(run / "history.json")
            regenerated = [schedule_hash(epoch_schedule(rows, config["seed"], epoch, config["updates_per_epoch"]))
                           for epoch in range(config["epochs"])]
            if regenerated != [row["schedule_sha256"] for row in history]:
                raise RuntimeError(f"Completed sampler history differs: {study_name}/{config['job_id']}")
            verified.append({"study": study_name, "job_id": config["job_id"], "epochs": config["epochs"],
                             "updates_per_epoch": config["updates_per_epoch"],
                             "train_rows_value_sha256": value_hash(rows),
                             "checkpoint_sha256": expected_checkpoint,
                             "checkpoint_bytes": checkpoint.stat().st_size})
    return verified


def preflight(runtime: Path) -> dict:
    studies = {}
    for name, count in EXPECTED_STUDIES.items():
        path = runtime / "studies" / name
        if path.exists():
            protocol = read(path / "protocol.json")
            studies[name] = {"registered_jobs": len(protocol.get("jobs", [])), "expected_jobs": count,
                             "bundle_manifest_sha256": sha256(path / "bundle_manifest.json")}
        else:
            studies[name] = {"registered_jobs": 0, "expected_jobs": count, "not_yet_frozen": True}
    h_rows = historical_h_rows()
    h_checkpoints = verify_historical_h_checkpoints(h_rows)
    y_rows = historical_y_rows()
    y_checkpoints = verify_historical_y_checkpoints(y_rows)
    schedule_records = verify_completed_schedule_records(runtime)
    return {"checked_at": now(), "status": "preflight_only", "formal_gate_passed": False,
            "studies": studies, "historical_h_runs_available": len(h_rows),
            "historical_y_runs_available": len(y_rows), "source_snapshot_files": len(source_candidates()),
            "historical_checkpoint_files_verified": {
                "H": len(h_checkpoints), "Y": len(y_checkpoints),
                "all_hashes_nonempty_and_match_files": True,
            },
            "completed_formal_sampler_histories_verified": schedule_records,
            "git": git_metadata(),
            "limitations": ["Formal results are intentionally unavailable until the 216-run/117-analysis gate completes.",
                            "This preflight is not a release bundle and contains no scientific completion claim."]}


def build(runtime: Path, evidence: Path, assets: Path, final_qa: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Preserve existing release bundle: {output}")
    gate = verify_formal_gate(evidence, assets, final_qa)
    output.mkdir(parents=True)
    checkpoint_index, schedule_index, study_counts = [], [], {}
    for name, count in EXPECTED_STUDIES.items():
        source = runtime / "studies" / name
        checkpoints, schedules = copy_formal_study(source, output / "formal_studies" / name, count,
                                                   output / "sampler_schedules")
        checkpoint_index.extend(checkpoints)
        schedule_index.extend(schedules)
        study_counts[name] = count

    h = copy_historical_h(output / "historical/H")
    y = copy_historical_y(output / "historical/Y")
    checkpoint_index.extend(h["checkpoint_index"])
    checkpoint_index.extend(y["checkpoint_index"])

    copy_tree(ROOT / "docs/revision", output / "revision_record/docs/revision")
    for source in source_candidates():
        copy_file(source, output / "source_snapshot" / source.relative_to(ROOT))
    copy_tree(assets, output / "paper/formal_assets")
    copy_file(ROOT / "paper/sensors_manuscript/manuscript.pdf", output / "paper/manuscript.pdf")
    copy_file(final_qa, output / "paper/manuscript_final_qa.json")
    write(output / "checkpoint_index.json", checkpoint_index)
    write(output / "sampler_schedule_index.json", schedule_index)
    metadata = {
        "schema": "pa-csi-dg-revision-release/v1",
        "built_at": now(),
        "formal_studies": study_counts,
        "formal_run_count": gate["evidence"]["formal_run_count"],
        "formal_source_analysis_count": gate["evidence"]["formal_source_analysis_count"],
        "historical_H_runs": h["runs"],
        "historical_Y_runs": y["runs"],
        "exact_formal_sampler_schedule_files": len(schedule_index),
        "checkpoint_files_duplicated": False,
        "checkpoint_verification": "SHA-256 values are in checkpoint_index.json and formal run records.",
        "source_archives_repackaged": False,
        "source_data": {
            "MultiEnv": "https://doi.org/10.17632/v38wjmz6f6.1",
            "Widar3.0_project": "https://tns.thss.tsinghua.edu.cn/widar3.0/index.html",
            "Widar3.0_share": "https://cloud.tsinghua.edu.cn/d/2760bb9557ca4d09a74d/",
        },
        "git": git_metadata(),
        "scope": "H/Y retrospective records plus the complete frozen R/control/input/Widar validation chain.",
        "inference_scope": "Environment summaries and fixed-seed dispersion are descriptive; no population p-value or confidence interval is supplied.",
    }
    write(output / "release_metadata.json", metadata)
    readme = """# PA-CSI-DG revision reproducibility package

This directory is the hash-bound local staging package for the revision. It contains the exact frozen source snapshots used by each formal study, split and identity manifests, per-run configurations, histories, append-only event records, source and target predictions, target results, formal source-only analyses, manuscript evidence and paper assets.

`sampler_schedule_index.json` binds every new formal job to a compressed file containing every zero-based training-row position for every epoch and optimizer update. Each regenerated epoch was checked against the hash stored during training. Historical H/Y records retain the hashes and deterministic generators saved at the time; their unrecorded full batch-index lists are not reconstructed as if they were original artifacts.

The compact package does not duplicate model checkpoints or source dataset archives. `checkpoint_index.json` records the checkpoint SHA-256 values. Source datasets remain with their providers, and all processed inputs are identified by hashes and provenance records.

`release_manifest.json` is the authoritative identity of this staged revision. The Git HEAD in `release_metadata.json` is only the repository base because the revision working tree was not a clean committed state. Replace `AUTHOR_TO_SUPPLY_BEFORE_SUBMISSION` with the final public repository commit or archive DOI only after publishing this exact package.
"""
    (output / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    files = [path for path in output.rglob("*") if path.is_file() and path.name != "release_manifest.json"]
    manifest = {
        "schema": "pa-csi-dg-release-manifest/v1",
        "created_at": now(),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "files_sha256": {path.relative_to(output).as_posix(): sha256(path) for path in sorted(files)},
    }
    write(output / "release_manifest.json", manifest)
    # Verify after writing so a partial copy cannot be reported as complete.
    verify_manifest(output, "release_manifest.json")
    return {"status": "completed", "output": str(output), "file_count": manifest["file_count"],
            "total_bytes": manifest["total_bytes"], "release_manifest_sha256": sha256(output / "release_manifest.json"),
            "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=Path("D:/pa-csi-dg-revision-runtime"))
    parser.add_argument("--evidence", type=Path, default=ROOT / "docs/revision/evidence/formal_revision_v1")
    parser.add_argument("--assets", type=Path, default=ROOT / "paper/sensors_manuscript/revision_assets/formal_v1")
    parser.add_argument("--final-qa", type=Path, default=ROOT / "docs/revision/evidence/manuscript_final_qa.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    if args.preflight_only:
        report = preflight(runtime)
        if args.preflight_output:
            write(args.preflight_output.resolve(), report)
        print(json.dumps(report, indent=2))
        return
    if args.output is None:
        parser.error("--output is required unless --preflight-only is used")
    result = build(runtime, args.evidence.resolve(), args.assets.resolve(), args.final_qa.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
