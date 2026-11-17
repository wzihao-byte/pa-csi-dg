"""Independent CPU-only 63-run synthetic tests for the phase follow-up report."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import psutil

from experiments.revision_validation import phase_followup_study as phase
from experiments.revision_validation.study_io import read, sha256, value_hash, write
from experiments.revision_validation.time_pooling_report import _metrics


RUNTIME = Path("D:/pa-csi-dg-revision-runtime/qa")
ARMS = {"core": ("R10",), "inputs": ("NATIVE_LINEAR", "TWIN_AA", "TWIN_AP"),
        "new": ("HIST_TWIN_AA", "HIST_TWIN_AP", "NATIVE_HISTSTYLE_AP")}
CORRECT = {"R10": 1200, "HIST_TWIN_AA": 1800, "HIST_TWIN_AP": 1500,
           "NATIVE_LINEAR": 1704, "TWIN_AA": 2100, "TWIN_AP": 1800, "NATIVE_HISTSTYLE_AP": 1902}
PHASE_CODE = Path(phase.__file__)


def forbidden(*args, **kwargs):
    raise AssertionError("This report fixture must never construct/train a model or initialize CUDA")


phase.build_experiment_model = forbidden
phase.train_experiment = forbidden
phase.torch.cuda._lazy_init = forbidden
phase.torch.set_num_threads(1)
if os.name == "nt":
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


def fixture(root):
    roots = {role: root / role for role in ARMS}
    for role, bundle in roots.items():
        frozen, jobs, results = {}, [], []
        for arm in ARMS[role]:
            grid = "historical" if arm in {"R10", "HIST_TWIN_AA", "HIST_TWIN_AP"} else "native"
            for target, seeds in phase.SEEDS.items():
                target_rows = [{"sample_id": f"{grid}:{target}:target:{i}",
                                "subject_id": f"{grid}:{target}:target:subject:{i // 100}",
                                "acquisition_id": f"{grid}:{target}:target:acq:{i}", "label": i % 6}
                               for i in range(3000)]
                source_rows = [{"sample_id": f"{grid}:{target}:source:{i}",
                                "subject_id": f"{grid}:{target}:source:subject:{i}",
                                "acquisition_id": f"{grid}:{target}:source:acq:{i}", "label": i} for i in range(6)]
                source_path = bundle / "manifests" / arm / f"source_{target}.json"
                target_path = (bundle / "manifests" / f"deferred_target_{target}.json" if role == "core"
                               else bundle / "manifests" / arm / f"deferred_target_{target}.json")
                write(source_path, {"train": [], "val": source_rows})
                write(target_path, {"test": target_rows})
                frozen[source_path.relative_to(bundle).as_posix()] = sha256(source_path)
                frozen[target_path.relative_to(bundle).as_posix()] = sha256(target_path)
                source_predictions = [{"sample_id": row["sample_id"], "subject_id": row["subject_id"],
                                       "acquisition_id": row["acquisition_id"], "true": row["label"],
                                       "predicted": row["label"]} for row in source_rows]
                source_metrics = _metrics(source_predictions)
                for seed in seeds:
                    job_id = f"{target}_{arm}_{seed}"
                    config = {"job_id": job_id, "target_name_only": target, "seed": seed, "arm": "R10",
                              "epochs": 2, "updates_per_epoch": 1, "source_manifest_path": str(source_path),
                              "source_manifest_sha256": sha256(source_path)}
                    if role != "core":
                        config.update(control_id=arm, target_manifest_path=str(target_path),
                                      target_manifest_sha256=sha256(target_path))
                    run = bundle / "runs" / job_id
                    write(run / "config.json", config)
                    write(run / "history.json", [{"source_validation": source_metrics}] * 2)
                    write(run / "selected_source_predictions.json", source_predictions)
                    (run / "best_model.pt").write_bytes(b"CPU fixture only; never loaded")
                    artifacts = {name: sha256(run / name) for name in
                                 ("config.json", "history.json", "selected_source_predictions.json", "best_model.pt")}
                    write(run / "status.json", {"status": "completed", "epochs": 2,
                          "config_value_sha256": value_hash(config), "optimizer_updates": 2, "best_epoch": 1,
                          "best_source_macro_f1": 1.0, "target_data_loaded": False, "artifacts_sha256": artifacts})
                    count_correct = CORRECT[arm] + (seed % 10 - 1) * 6
                    predictions = [{"sample_id": row["sample_id"], "subject_id": row["subject_id"],
                                    "acquisition_id": row["acquisition_id"], "true": row["label"],
                                    "predicted": row["label"] if i < count_correct else (row["label"] + 1) % 6}
                                   for i, row in enumerate(target_rows)]
                    result = {"job_id": job_id, "arm": arm, "target": target, "seed": seed,
                              "config_value_sha256": value_hash(config), "checkpoint_sha256": sha256(run / "best_model.pt"),
                              "source_predictions_reproduced": True, "sample_count": 3000,
                              "source_metrics": source_metrics, "metrics": _metrics(predictions)}
                    write(bundle / "evaluation" / job_id / "result.json", result)
                    write(bundle / "evaluation" / job_id / "predictions.json", predictions)
                    jobs.append(config)
                    results.append(result)
        protocol = {"schema": "revision-core-study/v1" if role == "core" else "revision-extension-study/v1", "jobs": jobs}
        if role == "new":
            protocol.update(references=[{"role": ref, "bundle": str(roots[ref]),
                             "bundle_sha256": sha256(roots[ref] / "bundle_manifest.json"),
                             "evaluation_summary_sha256": sha256(roots[ref] / "evaluation/summary.json")}
                            for ref in ("core", "inputs")], inference_scope="Synthetic fixture; descriptive checks only",
                            limits=["Synthetic fixture: no physical CSI signal, no training, no target inference."])
        write(bundle / "protocol.json", protocol)
        frozen["protocol.json"] = sha256(bundle / "protocol.json")
        write(bundle / "bundle_manifest.json", {"files_sha256": frozen})
        write(bundle / "target_gate.json", {"bundle_sha256": sha256(bundle / "bundle_manifest.json"),
              "source_status_sha256": {c["job_id"]: sha256(bundle / "runs" / c["job_id"] / "status.json") for c in jobs}})
        write(bundle / "evaluation/summary.json", {"runs": results,
              "artifacts_sha256": {p.relative_to(bundle).as_posix(): sha256(p) for p in (bundle / "evaluation").glob("*/*.json")}})
    return roots["new"]


class RestoringEdits:
    def __init__(self):
        self.originals = {}

    def write(self, path, value):
        self.originals.setdefault(path, path.read_bytes() if path.exists() else None)
        write(path, value)

    def raw(self, path, value):
        self.originals.setdefault(path, path.read_bytes() if path.exists() else None)
        path.write_bytes(value)

    def restore(self):
        for path, content in self.originals.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)


def rebind(bundle, edit):
    """Make deliberate fixture mutations internally hash-consistent before testing semantics."""
    lock = read(bundle / "bundle_manifest.json")
    lock["files_sha256"] = {name: sha256(bundle / name) for name in lock["files_sha256"]}
    edit.write(bundle / "bundle_manifest.json", lock)
    gate = read(bundle / "target_gate.json")
    gate["bundle_sha256"] = sha256(bundle / "bundle_manifest.json")
    gate["source_status_sha256"] = {c["job_id"]: sha256(bundle / "runs" / c["job_id"] / "status.json")
                                   for c in read(bundle / "protocol.json")["jobs"]}
    edit.write(bundle / "target_gate.json", gate)
    summary = read(bundle / "evaluation/summary.json")
    summary["artifacts_sha256"] = {name: sha256(bundle / name) for name in summary["artifacts_sha256"]}
    edit.write(bundle / "evaluation/summary.json", summary)


def reject(bundle, action, message):
    edit = RestoringEdits()
    try:
        action(edit)
        try:
            phase.summarize(bundle)
        except RuntimeError as exc:
            assert message in str(exc), (message, str(exc))
        else:
            raise AssertionError("Invalid evidence was accepted")
    finally:
        edit.restore()


def run_suite():
    start_sha = sha256(PHASE_CODE)
    report_function_sha = hashlib.sha256(inspect.getsource(phase.summarize).encode("utf-8")).hexdigest()
    root = RUNTIME / "phase_followup_report_20260929_fixtures" / f"run_{time.time_ns()}"
    root.mkdir(parents=True)
    bundle = fixture(root)
    checks = []

    def check(name, fn):
        try:
            fn()
            checks.append({"name": name, "passed": True})
        except BaseException:
            checks.append({"name": name, "passed": False, "error": traceback.format_exc()})

    def complete():
        digest = phase.summarize(bundle)
        report = read(bundle / "comparison/report.json")
        assert report["run_count"] == len(report["runs"]) == 63 and report["new_run_count"] == 27
        assert len(report["cells"]) == 28 and len(report["paired_differences"]) == 72 and len(report["paired_summary"]) == 32
        # All six class supports and errors are balanced in this fixture, so
        # macro-F1 has an independently known value: fraction classified correctly.
        for row in report["runs"]:
            expected_f1 = (CORRECT[row["arm"]] + (row["seed"] % 10 - 1) * 6) / 3000
            assert abs(row["macro_f1"] - expected_f1) < 1e-12
        pairs = report["paired_differences"]
        assert len({(r["grid"], r["comparison"]) for r in pairs}) == 8
        assert all(sum(p["grid"] == r["grid"] and p["comparison"] == r["comparison"] for p in pairs) == 9 for r in pairs)
        lookup = {(r["grid"], r["cell"], r["target"], r["seed"]): r["macro_f1"] for r in report["runs"]}
        for pair in pairs:
            treatment, reference = pair["comparison"].split(" minus ")
            expected = 100 * (lookup[pair["grid"], treatment, pair["target"], pair["seed"]]
                              - lookup[pair["grid"], reference, pair["target"], pair["seed"]])
            assert pair["effect_pp"] == expected
        assert all(r["target"] in phase.SEEDS for r in pairs)
        assert len({r["seed"] for r in report["runs"]}) == 9
        return digest

    check("complete_63_runs_seven_cells_eight_paired_contrasts", complete)

    def negative():
        report = read(bundle / "comparison/report.json")
        row = next(r for r in report["paired_summary"] if r["grid"] == "historical"
                   and r["comparison"] == "AP_historical minus AA" and r["target"] == "ALL")
        assert row["negative_pairs"] == row["n"] == 9 and row["positive_pairs"] == 0
        assert abs(row["mean_pp"] + 10.0) < 1e-10

    check("retains_all_nine_negative_historical_phase_differences", negative)
    original_summary = read(bundle / "evaluation/summary.json")
    first_job = original_summary["runs"][0]["job_id"]
    predictions_path = bundle / "evaluation" / first_job / "predictions.json"

    def missing(edit):
        modified = copy.deepcopy(original_summary)
        modified["runs"].pop()
        edit.write(bundle / "evaluation/summary.json", modified)

    check("missing_evaluated_run_rejected", lambda: reject(bundle, missing, "Evaluation does not cover every frozen source job"))
    check("unrebound_prediction_hash_tamper_rejected", lambda: reject(bundle,
          lambda edit: edit.raw(predictions_path, predictions_path.read_bytes() + b"\n"), "Evaluation artifact changed"))

    def metric(edit):
        modified = copy.deepcopy(original_summary)
        modified["runs"][0]["metrics"]["macro_f1"] = .999
        result_path = bundle / "evaluation" / first_job / "result.json"
        edit.write(result_path, modified["runs"][0])
        modified["artifacts_sha256"][result_path.relative_to(bundle).as_posix()] = sha256(result_path)
        edit.write(bundle / "evaluation/summary.json", modified)

    check("hash_consistent_wrong_metrics_rejected", lambda: reject(bundle, metric, "Target metrics do not reconstruct"))

    def identities(edit):
        # Change a whole within-grid arm coherently: its prediction-to-manifest
        # verification passes, leaving the cross-cell identity guard to reject it.
        protocol = read(bundle / "protocol.json")
        summary = read(bundle / "evaluation/summary.json")
        target_path = bundle / "manifests/HIST_TWIN_AP/deferred_target_E1.json"
        target = read(target_path)
        target["test"][0]["subject_id"] = "wrong-but-consistent-with-this-arm"
        edit.write(target_path, target)
        for config in protocol["jobs"]:
            if config["control_id"] != "HIST_TWIN_AP" or config["target_name_only"] != "E1":
                continue
            config["target_manifest_sha256"] = sha256(target_path)
            job_id = config["job_id"]
            run = bundle / "runs" / job_id
            edit.write(run / "config.json", config)
            status = read(run / "status.json")
            status["config_value_sha256"] = value_hash(config)
            status["artifacts_sha256"]["config.json"] = sha256(run / "config.json")
            edit.write(run / "status.json", status)
            prediction_path = bundle / "evaluation" / job_id / "predictions.json"
            predictions = read(prediction_path)
            predictions[0]["subject_id"] = target["test"][0]["subject_id"]
            edit.write(prediction_path, predictions)
            result = next(r for r in summary["runs"] if r["job_id"] == job_id)
            result["config_value_sha256"] = value_hash(config)
            edit.write(bundle / "evaluation" / job_id / "result.json", result)
        edit.write(bundle / "protocol.json", protocol)
        edit.write(bundle / "evaluation/summary.json", summary)
        rebind(bundle, edit)

    check("within_grid_cross_cell_identity_mismatch_rejected", lambda: reject(bundle, identities, "Within-grid target identities differ"))

    def extra_job(edit):
        protocol = read(bundle / "protocol.json")
        old = protocol["jobs"][0]
        extra = {**old, "job_id": "UNPLANNED_EXTRA", "control_id": "UNPLANNED_EXTRA"}
        protocol["jobs"].append(extra)
        edit.write(bundle / "protocol.json", protocol)
        # _verify_study checks summary coverage and source gate, so construct valid
        # corresponding metadata rather than relying on an earlier rejection.
        edit.write(bundle / "runs/UNPLANNED_EXTRA/status.json", {"status": "completed"})
        summary = read(bundle / "evaluation/summary.json")
        result = {**summary["runs"][0], "job_id": extra["job_id"], "arm": extra["control_id"]}
        summary["runs"].append(result)
        edit.write(bundle / "evaluation/summary.json", summary)
        rebind(bundle, edit)

    check("extra_new_job_rejected", lambda: reject(bundle, extra_job, "27"))

    def duplicate_reference(edit):
        protocol = read(bundle / "protocol.json")
        protocol["references"].append(copy.deepcopy(protocol["references"][0]))
        edit.write(bundle / "protocol.json", protocol)
        rebind(bundle, edit)

    check("duplicate_reference_role_rejected", lambda: reject(bundle, duplicate_reference, "reference"))

    def stable():
        report_hash = sha256(bundle / "comparison/report.json")
        before = {path.name: sha256(path) for path in (bundle / "comparison").iterdir() if path.is_file()}
        assert phase.summarize(bundle) == report_hash
        assert {path.name: sha256(path) for path in (bundle / "comparison").iterdir() if path.is_file()} == before

    check("repeat_report_json_csv_md_stable", stable)
    module_unchanged = start_sha == sha256(PHASE_CODE)
    result = {"passed": all(c["passed"] for c in checks) and module_unchanged, "checks": checks,
              "synthetic_runs": 63, "synthetic_prediction_rows": 189000, "cells": 7,
              "contrasts": 8, "matched_pairs_per_contrast": 9,
              "phase_module_sha256_at_start": start_sha, "phase_module_sha256_at_end": sha256(PHASE_CODE),
              "phase_module_unchanged_during_tests": module_unchanged,
              "report_function_sha256": report_function_sha,
              "helper_sha256": sha256(Path(phase._verified_run.__code__.co_filename)),
              "test_script_sha256": sha256(__file__), "fixture_directory": str(root),
              "models_constructed": False, "gpu_initialized": False, "training_or_target_inference": False}
    write(root / "result.json", result)
    write(RUNTIME / "phase_followup_report_20260929.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    raise SystemExit(0 if run_suite()["passed"] else 1)
