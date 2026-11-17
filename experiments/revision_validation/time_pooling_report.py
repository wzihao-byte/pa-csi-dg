"""Verify and summarize all six time-preprocessing/pooling cells without filtering runs.

This module reads predictions and provenance only; it neither loads signal arrays
nor constructs a model. Run it after the new 18-run deferred evaluation finishes.
"""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import numpy as np

from experiments.revision_validation.study_io import (
    bound_path, read, sha256, value_hash, verify_completed, write,
)


SEEDS = {"E1": (401, 402, 403), "E2": (501, 502, 503), "E3": (601, 602, 603)}
CELLS = (
    ("A", "global_mlp", "core", "R00"),
    ("A", "tpp", "core", "R10"),
    ("B", "global_mlp", "new", "GLOBAL_NATIVE_MERGE"),
    ("B", "tpp", "inputs", "NATIVE_MERGE"),
    ("C", "global_mlp", "new", "GLOBAL_NATIVE_LINEAR"),
    ("C", "tpp", "inputs", "NATIVE_LINEAR"),
)
PIPELINES = {
    "A": "Historical MAT timestamp representation and historical merge.",
    "B": "Native DAT elapsed clock and historical merge.",
    "C": "Native DAT elapsed clock and linear interpolation onto 850 equally spaced points per recording.",
}
LIMITATIONS = [
    "This is a descriptive follow-up in previously inspected development environments; it is not an untouched confirmatory test.",
    "All three prespecified seeds in each of three environments are retained, including negative effects. No population confidence interval or population significance claim is produced.",
    "A to B changes timestamp representation/origin while retaining historical merge; it does not isolate a single physical clock defect.",
    "B to C compares resampling methods conditional on the native clock. The design is not a complete timestamp-by-resampling factorial experiment.",
    "C uses 850 points over each recording's elapsed duration: its effective sampling frequency varies with duration. Interpolation does not recover unobserved motion inside acquisition gaps.",
    "TPP-minus-global contrasts apply to these capacity-controlled modules under the frozen protocol; a contrast alone does not prove which temporal mechanism caused it.",
]


def _need(condition, message):
    if not condition:
        raise RuntimeError(message)


def _metrics(predictions):
    _need(bool(predictions), "Empty prediction artifact")
    confusion = np.zeros((6, 6), dtype=np.int64)
    for row in predictions:
        _need(type(row["true"]) is int and type(row["predicted"]) is int
              and 0 <= row["true"] < 6 and 0 <= row["predicted"] < 6,
              "Prediction has an invalid class index")
        confusion[row["true"], row["predicted"]] += 1
    diagonal = confusion.diagonal()
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros(6), where=denominator > 0)
    return {"macro_f1": float(f1.mean()), "class_f1": f1.tolist(),
            "accuracy": float(diagonal.sum() / confusion.sum()),
            "confusion_matrix": confusion.tolist()}


def _identities(predictions):
    identities = {row["sample_id"]: (row["subject_id"], row["acquisition_id"], row["true"])
                  for row in predictions}
    _need(len(identities) == len(predictions), "Duplicate prediction sample identity")
    return identities


def _verify_study(bundle, expected_bundle_hash=None, expected_summary_hash=None):
    bundle = Path(bundle).resolve()
    manifest = read(bundle / "bundle_manifest.json")
    bundle_hash = sha256(bundle / "bundle_manifest.json")
    _need(expected_bundle_hash is None or bundle_hash == expected_bundle_hash,
          f"Reference bundle manifest hash changed: {bundle}")
    _need("protocol.json" in manifest["files_sha256"], "Protocol is not covered by the freeze")
    for relative, expected in manifest["files_sha256"].items():
        _need(sha256(bound_path(bundle, relative)) == expected, f"Frozen file changed: {bundle}/{relative}")
    protocol = read(bundle / "protocol.json")
    _need(protocol["schema"] in ("revision-core-study/v1", "revision-extension-study/v1"),
          "Unsupported reference protocol schema")
    summary_path = bundle / "evaluation/summary.json"
    summary_hash = sha256(summary_path)
    _need(expected_summary_hash is None or summary_hash == expected_summary_hash,
          f"Frozen reference evaluation summary changed: {bundle}")
    summary = read(summary_path)
    for relative, expected in summary["artifacts_sha256"].items():
        _need(sha256(bound_path(bundle, relative)) == expected,
              f"Evaluation artifact changed: {bundle}/{relative}")
    jobs = {config["job_id"]: config for config in protocol["jobs"]}
    results = {result["job_id"]: result for result in summary["runs"]}
    _need(len(jobs) == len(protocol["jobs"]) and len(results) == len(summary["runs"]),
          "Duplicate job identifier")
    _need(set(jobs) == set(results), "Evaluation does not cover every frozen source job")
    gate = read(bundle / "target_gate.json")
    _need(gate["bundle_sha256"] == bundle_hash, "Target gate belongs to a different frozen bundle")
    _need(set(gate["source_status_sha256"]) == set(jobs), "Target gate does not cover all source jobs")
    for job_id, expected in gate["source_status_sha256"].items():
        _need(sha256(bound_path(bundle, f"runs/{job_id}/status.json")) == expected,
              f"Source status changed since target gate: {job_id}")
    return {"bundle": bundle, "bundle_sha256": bundle_hash, "protocol": protocol,
            "summary": summary, "evaluation_summary_sha256": summary_hash,
            "jobs": jobs, "results": results}


def _verified_run(study, result):
    bundle, job_id = study["bundle"], result["job_id"]
    config = study["jobs"][job_id]
    run = bound_path(bundle, f"runs/{job_id}")
    _need(read(run / "config.json") == config, f"Stored configuration differs: {job_id}")
    status = verify_completed(run, config)
    _need(not status.get("target_data_loaded", False), "Training status reports target data access")
    _need(result["config_value_sha256"] == value_hash(config), "Evaluation configuration hash mismatch")
    _need(result["checkpoint_sha256"] == sha256(run / "best_model.pt"), "Evaluation checkpoint hash mismatch")
    _need(result.get("source_predictions_reproduced") is True, "Source predictions were not reproduced")
    _need(result["target"] == config["target_name_only"] and result["seed"] == config["seed"],
          "Evaluation target or seed differs from its configuration")
    _need(result["arm"] == config.get("control_id", config["arm"]), "Evaluation arm differs from its configuration")
    artifacts = study["summary"]["artifacts_sha256"]
    for name in ("result.json", "predictions.json"):
        _need(f"evaluation/{job_id}/{name}" in artifacts, f"Unhashed evaluation artifact: {job_id}/{name}")
    _need(read(bundle / "evaluation" / job_id / "result.json") == result,
          f"Summary and stored result disagree: {job_id}")
    predictions = read(bundle / "evaluation" / job_id / "predictions.json")
    _need(len(predictions) == result["sample_count"] == 3000, "Target prediction count is not 3000")
    _need(_metrics(predictions) == result["metrics"], f"Target metrics do not reconstruct: {job_id}")
    identities = _identities(predictions)
    source_path = Path(config["source_manifest_path"])
    _need(sha256(source_path) == config["source_manifest_sha256"], "Source manifest hash changed")
    source = read(source_path)
    source_predictions = read(run / "selected_source_predictions.json")
    expected_source = {row["sample_id"]: (row["subject_id"], row["acquisition_id"], row["label"])
                       for row in source["val"]}
    _need(len(expected_source) == len(source["val"]), "Duplicate source validation identities")
    _need(_identities(source_predictions) == expected_source, "Source prediction identities differ")
    source_metrics = _metrics(source_predictions)
    _need(source_metrics == result["source_metrics"]
          == read(run / "history.json")[status["best_epoch"] - 1]["source_validation"],
          "Selected source metrics do not reconstruct")
    target_path = (Path(config["target_manifest_path"]) if "target_manifest_path" in config
                   else bundle / "manifests" / f"deferred_target_{result['target']}.json")
    if "target_manifest_sha256" in config:
        _need(sha256(target_path) == config["target_manifest_sha256"], "Target manifest hash changed")
    else:
        relative = target_path.relative_to(bundle).as_posix()
        _need(relative in read(bundle / "bundle_manifest.json")["files_sha256"], "Target manifest is not frozen")
    target_rows = read(target_path)["test"]
    expected_target = {row["sample_id"]: (row["subject_id"], row["acquisition_id"], row["label"])
                       for row in target_rows}
    _need(len(expected_target) == len(target_rows) and identities == expected_target,
          "Target predictions and manifest identities/labels differ")
    for key in ("sample_id", "subject_id", "acquisition_id"):
        source_ids = {row[key] for part in ("train", "val") for row in source[part]}
        _need(not source_ids.intersection(row[key] for row in target_rows), "Source/target identity overlap")
    return identities, status


def _stats(values):
    return {"n": len(values), "mean_pp": float(np.mean(values)),
            "sample_sd_pp": float(np.std(values, ddof=1)), "min_pp": min(values), "max_pp": max(values),
            "positive_pairs": sum(value > 0 for value in values),
            "zero_pairs": sum(value == 0 for value in values),
            "negative_pairs": sum(value < 0 for value in values)}


def _aggregate_pairs(rows, key):
    output = []
    for name in sorted({row[key] for row in rows}):
        for target in ("ALL", *SEEDS):
            selected = [row for row in rows if row[key] == name and (target == "ALL" or row["target"] == target)]
            output.append({key: name, "target": target, **_stats([row["effect_pp"] for row in selected])})
    return output


def _csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _write_text_once(path, text):
    if path.exists():
        _need(path.read_text(encoding="utf-8") == text, f"Existing completed comparison differs: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _markdown(report):
    lines = ["# Time preprocessing and pooling: complete paired comparison", "",
             "All 54 runs are included: 6 cells × 3 environments × 3 seeds. Macro-F1 is reconstructed from 3000 target predictions per run. Checkpoints use source validation only.", "",
             "| Pipeline | Input processing |", "|---|---|"]
    lines.extend(f"| {key} | {text} |" for key, text in PIPELINES.items())
    lines += ["", "## Six cells", "", "Values are percentages. ALL averages the three environment means equally.", "",
              "| Pipeline | Pooling | E1 | E2 | E3 | ALL |", "|---|---|---:|---:|---:|---:|"]
    for pipeline, pooling, _, _ in CELLS:
        values = {row["target"]: row["macro_f1_pct"] for row in report["cells"]
                  if row["pipeline"] == pipeline and row["pooling"] == pooling}
        lines.append(f"| {pipeline} | {pooling} | " + " | ".join(f"{values[t]:.3f}" for t in (*SEEDS, "ALL")) + " |")
    lines += ["", "## TPP minus global MLP", "", "Positive values favor TPP; differences use unrounded per-seed scores.", "",
              "| Pipeline | Environment | Mean difference (pp) | Positive / zero / negative |", "|---|---|---:|---|"]
    for row in report["paired_gain_summary"]:
        lines.append(f"| {row['pipeline']} | {row['target']} | {row['mean_pp']:+.3f} | {row['positive_pairs']} / {row['zero_pairs']} / {row['negative_pairs']} |")
    lines += ["", "## Change in the TPP advantage", "",
              "For example, B→C = (C TPP − C global) − (B TPP − B global), pairing the same environment and seed. Positive values mean that TPP's advantage is larger in the later pipeline.", "",
              "| Transition | Environment | Change in advantage (pp) | Positive / zero / negative |", "|---|---|---:|---|"]
    for row in report["gain_change_summary"]:
        lines.append(f"| {row['transition']} | {row['target']} | {row['mean_pp']:+.3f} | {row['positive_pairs']} / {row['zero_pairs']} / {row['negative_pairs']} |")
    lines += ["", "## Interpretation limits", ""] + [f"- {text}" for text in LIMITATIONS]
    lines += ["", "Within-environment standard deviations describe variation over three training seeds. Across-all-pairs standard deviations mix environment and seed variation; neither is a population confidence interval.", "",
              "Machine-readable evidence: `report.json`, `runs.csv`, `cells.csv`, `paired_gains.csv`, `gain_changes.csv`, `paired_gain_summary.csv`, and `gain_change_summary.csv`. Each run records its configuration, checkpoint, result, and prediction hashes.", ""]
    return "\n".join(lines)


def summarize(bundle):
    """Verify all evidence, write deterministic complete reports, and return report SHA-256."""
    bundle = Path(bundle).resolve()
    new = _verify_study(bundle)
    _need(len(new["jobs"]) == 18, "New study must contain exactly 18 prespecified global controls")
    references = new["protocol"]["references"]
    _need(len(references) == 2 and {row["role"] for row in references} == {"core", "inputs"},
          "Exactly one core and one inputs reference are required")
    studies = {"new": new}
    for reference in references:
        studies[reference["role"]] = _verify_study(reference["bundle"], reference["bundle_sha256"],
                                                   reference["evaluation_summary_sha256"])
    expected_pairs = {(target, seed) for target, seeds in SEEDS.items() for seed in seeds}
    rows, lookup, target_identities = [], {}, {}
    for pipeline, pooling, role, arm in CELLS:
        study = studies[role]
        selected = [result for result in study["results"].values() if result["arm"] == arm]
        _need(len(selected) == 9 and {(r["target"], r["seed"]) for r in selected} == expected_pairs,
              f"Cell {pipeline}/{pooling} does not contain exactly its nine prespecified target/seed pairs")
        for result in sorted(selected, key=lambda row: (row["target"], row["seed"])):
            identities, status = _verified_run(study, result)
            target = result["target"]
            if target in target_identities:
                _need(identities == target_identities[target], f"Target identities/labels differ across cells or seeds: {target}")
            else:
                target_identities[target] = identities
            output = study["bundle"] / "evaluation" / result["job_id"]
            row = {"pipeline": pipeline, "pooling": pooling, "study_role": role, "arm": arm,
                   "target": target, "seed": result["seed"], "job_id": result["job_id"],
                   "macro_f1": result["metrics"]["macro_f1"], "macro_f1_pct": 100 * result["metrics"]["macro_f1"],
                   "accuracy_pct": 100 * result["metrics"]["accuracy"], "sample_count": result["sample_count"],
                   "source_best_epoch": status["best_epoch"], "source_macro_f1_pct": 100 * result["source_metrics"]["macro_f1"],
                   "config_value_sha256": result["config_value_sha256"], "checkpoint_sha256": result["checkpoint_sha256"],
                   "result_sha256": sha256(output / "result.json"), "predictions_sha256": sha256(output / "predictions.json")}
            rows.append(row)
            lookup[(pipeline, pooling, target, result["seed"])] = row
    _need(len(rows) == len(lookup) == 54, "Incomplete or duplicated six-cell run matrix")
    cells = []
    for pipeline, pooling, _, arm in CELLS:
        for target in ("ALL", *SEEDS):
            selected = [row for row in rows if row["pipeline"] == pipeline and row["pooling"] == pooling
                        and (target == "ALL" or row["target"] == target)]
            values = [row["macro_f1_pct"] for row in selected]
            environment_means = [np.mean([row["macro_f1_pct"] for row in selected if row["target"] == t])
                                 for t in sorted({row["target"] for row in selected})]
            cells.append({"pipeline": pipeline, "pooling": pooling, "arm": arm, "target": target,
                          "n_runs": len(values), "macro_f1_pct": float(np.mean(environment_means)),
                          "run_sample_sd_pct": float(np.std(values, ddof=1)),
                          "min_run_macro_f1_pct": min(values), "max_run_macro_f1_pct": max(values)})
    paired_gains, gain_lookup = [], {}
    for pipeline in PIPELINES:
        for target, seed in sorted(expected_pairs):
            global_row, tpp_row = (lookup[(pipeline, pooling, target, seed)] for pooling in ("global_mlp", "tpp"))
            effect = 100 * (tpp_row["macro_f1"] - global_row["macro_f1"])
            paired_gains.append({"pipeline": pipeline, "target": target, "seed": seed,
                                 "global_macro_f1_pct": global_row["macro_f1_pct"],
                                 "tpp_macro_f1_pct": tpp_row["macro_f1_pct"], "effect_pp": effect})
            gain_lookup[(pipeline, target, seed)] = effect
    gain_changes = []
    for before, after in (("A", "B"), ("B", "C"), ("A", "C")):
        for target, seed in sorted(expected_pairs):
            before_gain, after_gain = (gain_lookup[(p, target, seed)] for p in (before, after))
            gain_changes.append({"transition": f"{before}_to_{after}", "target": target, "seed": seed,
                                 "before_gain_pp": before_gain, "after_gain_pp": after_gain,
                                 "effect_pp": after_gain - before_gain})
    report = {"schema": "revision-time-pooling-comparison/v1", "complete": True,
              "pipelines": PIPELINES, "prespecified_seeds": SEEDS, "runs": rows, "cells": cells,
              "paired_gains": paired_gains, "gain_changes": gain_changes,
              "paired_gain_summary": _aggregate_pairs(paired_gains, "pipeline"),
              "gain_change_summary": _aggregate_pairs(gain_changes, "transition"),
              "inference_scope": LIMITATIONS,
              "verification": {"run_count": 54, "new_run_count": 18, "metrics_reconstructed": True,
                               "same_target_identities_and_labels_across_all_cells": True,
                               "target_samples_per_environment": {target: len(ids) for target, ids in target_identities.items()},
                               "all_prespecified_seeds_retained": True, "population_confidence_intervals": False},
              "provenance": [{"role": role, "bundle": str(study["bundle"]),
                              "bundle_sha256": study["bundle_sha256"],
                              "evaluation_summary_sha256": study["evaluation_summary_sha256"]}
                             for role, study in sorted(studies.items())]}
    # Serialize tuples consistently before comparing an existing deterministic report.
    report["prespecified_seeds"] = {target: list(seeds) for target, seeds in SEEDS.items()}
    output = bundle / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    if report_path.exists():
        _need(read(report_path) == report, "Existing completed comparison report differs; investigate without overwriting")
    else:
        write(report_path, report)
    for name in ("runs", "cells", "paired_gains", "gain_changes", "paired_gain_summary", "gain_change_summary"):
        _write_text_once(output / f"{name}.csv", _csv_text(report[name]))
    _write_text_once(output / "report.md", _markdown(report))
    return sha256(report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    print(summarize(parser.parse_args().bundle), flush=True)
