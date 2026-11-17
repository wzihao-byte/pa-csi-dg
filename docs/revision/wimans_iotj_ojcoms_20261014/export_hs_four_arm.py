"""Export the complete H-S capacity control, retaining paired and negative results."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_existing_queue import prediction_metrics

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(name, rows):
    with (ROOT / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    snapshot_path = ROOT / "existing_queue_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    reference_path = Path(snapshot["historical_reference_source"]["path"])
    assert sha(reference_path) == snapshot["historical_reference_source"]["sha256"]
    references = json.loads(reference_path.read_text(encoding="utf-8"))["references"]
    rows = []
    for run in snapshot["verified_runs"]:
        if run["arm"] not in ("G10", "G11"):
            continue
        target, seed = run["job_id"].split("_")[1], int(run["job_id"].split("_")[-1])
        payload = Path(run["attempt"]) / "payload/run"
        assert sha(payload / "metrics.json") == run["metrics_sha256"]
        assert sha(payload / "target_predictions.jsonl") == run["predictions_sha256"]
        metrics = run["recomputed"]
        rows.append({"arm": run["arm"], "target": target, "seed": seed,
                     "macro_f1_pct": metrics["macro_f1_pct"], "accuracy_pct": metrics["accuracy_pct"],
                     "prediction_count": metrics["prediction_count"], "attempt": run["attempt"],
                     "config_sha256": run["config_sha256"], "metrics_sha256": run["metrics_sha256"],
                     "predictions_sha256": run["predictions_sha256"]})
    for run in references:
        payload = Path(run["attempt"]) / "payload/run"
        assert sha(payload / "metrics.json") == run["metrics_sha256"]
        assert sha(payload / "target_predictions.jsonl") == run["predictions_sha256"]
        metrics = prediction_metrics(payload / "target_predictions.jsonl")
        rows.append({"arm": "HS" + run["arm"][1:], "target": run["target"], "seed": run["seed"],
                     "macro_f1_pct": metrics["macro_f1_pct"], "accuracy_pct": metrics["accuracy_pct"],
                     "prediction_count": metrics["prediction_count"], "attempt": run["attempt"],
                     "config_sha256": run["config_sha256"], "metrics_sha256": run["metrics_sha256"],
                     "predictions_sha256": run["predictions_sha256"]})
    assert len(rows) == 36
    lookup = {(r["arm"], r["target"], r["seed"]): r for r in rows}
    assert len(lookup) == 36 and all(r["prediction_count"] == 3000 for r in rows)
    pairs = []
    for room in ("E1", "E2", "E3"):
        for seed in (42, 52, 62):
            g0, g1, t0, t1 = [lookup[(arm, room, seed)]["macro_f1_pct"] for arm in ("G10", "G11", "HS10", "HS11")]
            pairs.append({"target": room, "seed": seed, "global_ce_f1_pct": g0, "global_sc_f1_pct": g1,
                          "tpp_ce_f1_pct": t0, "tpp_sc_f1_pct": t1, "global_sc_increment_pp": g1 - g0,
                          "tpp_sc_increment_pp": t1 - t0, "tpp_minus_global_ce_pp": t0 - g0,
                          "tpp_minus_global_sc_pp": t1 - g1, "interaction_pp": (t1 - t0) - (g1 - g0)})
    summaries = []
    for arm in ("G10", "G11", "HS10", "HS11"):
        for target in ("E1", "E2", "E3"):
            values = [lookup[(arm, target, seed)]["macro_f1_pct"] for seed in (42, 52, 62)]
            summaries.append({"arm": arm, "target": target, "runs": 3, "macro_f1_mean_pct": float(np.mean(values)),
                              "seed_sample_sd_pp": float(np.std(values, ddof=1))})
    write_csv("hs_four_arm_per_run.csv", sorted(rows, key=lambda r: (r["arm"], r["target"], r["seed"])))
    write_csv("hs_four_arm_paired.csv", pairs)
    write_csv("hs_four_arm_by_room.csv", summaries)
    provenance = {"status": "complete_four_arm_capacity_control", "runs_in_table": 36,
                  "existing_new_queue_runs_in_table": 18, "historical_runs_in_table": 18,
                  "prediction_count_in_table": 108000, "snapshot_sha256": sha(snapshot_path),
                  "historical_reference_sha256": sha(reference_path), "scope": "MultiEnv H-S only",
                  "equal_environment_mean_interaction_pp": float(np.mean([r["interaction_pp"] for r in pairs])),
                  "independent_environment_count": 3, "target_scores_used_for_WiMANS_design": False,
                  "release_unique_runs_or_prediction_count_updated": False,
                  "outputs_sha256": {name: sha(ROOT / name) for name in ("hs_four_arm_per_run.csv", "hs_four_arm_paired.csv", "hs_four_arm_by_room.csv")}}
    (ROOT / "hs_four_arm_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "paired_blocks": len(pairs), "interaction_pp": provenance["equal_environment_mean_interaction_pp"]}))


if __name__ == "__main__":
    main()
