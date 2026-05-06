from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("outputs_diag_prototype_supcon_layer1/dg_loeo")
RUNS = [
    "pa_csi_dg_subcenter_proto_layer1_40e_k2_nopair_e2",
    "pa_csi_dg_subcenter_proto_layer1_40e_k3_nopair_e2",
    "pa_csi_dg_subcenter_proto_layer1_40e_k3_margin_e2",
]


def read_confusion(path: Path) -> list[list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [[int(cell) for cell in row] for row in csv.reader(handle)]


def main() -> None:
    for experiment in RUNS:
        run_dir = ROOT / experiment / "target_E2" / "seed_42"
        metrics_path = run_dir / "metrics.json"
        confusion_path = run_dir / "confusion_matrix.csv"
        if not metrics_path.exists():
            print(f"{experiment}: missing metrics.json")
            continue
        if not confusion_path.exists():
            print(f"{experiment}: missing confusion_matrix.csv")
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        confusion = read_confusion(confusion_path)
        losses = metrics.get("losses_enabled", {})
        shape_debug = metrics.get("shape_debug", {})
        accuracy = float(metrics["test_metrics"]["accuracy"])
        f1_macro = float(metrics["test_metrics"]["f1_macro"])

        print(experiment)
        print(f"  accuracy: {accuracy:.6f}")
        print(f"  f1_macro: {f1_macro:.6f}")
        print(f"  contrastive_loss_type: {losses.get('contrastive_loss_type')}")
        print(f"  prototype_num_subcenters: {losses.get('prototype_num_subcenters')}")
        print(f"  lambda_pair_margin: {losses.get('lambda_pair_margin')}")
        print(f"  multi_antenna_views: {shape_debug.get('multi_antenna_views')}")
        print(f"  key_confusions: 1->4={confusion[1][4]}, 4->1={confusion[4][1]}, 0->4={confusion[0][4]}")


if __name__ == "__main__":
    main()
