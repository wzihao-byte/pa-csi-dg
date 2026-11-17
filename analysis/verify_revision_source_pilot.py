"""Verify the new source trainer's saved checkpoint against all validation rows."""
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.revision_validation.models import build_model
from experiments.revision_validation.prepare_verified_protocol import read, sha256, write
from experiments.revision_validation.source_data import VerifiedSources
from experiments.revision_validation.train_source import score_source


def main():
    folder = ROOT / "outputs_revision_validation/pilot_E1_R11_401/attempt_001"
    status, config = read(folder / "status.json"), read(folder / "config.json")
    assert status["status"] == "completed" and status["optimizer_updates"] == 300
    for name, key in (("config.json", "config_sha256"), ("best_model.pt", "best_model_sha256"),
                      ("history.json", "history_sha256"), ("selected_source_predictions.json", "predictions_sha256")):
        assert sha256(folder / name) == status[key]
    for path, expected in config["code_sha256"].items():
        assert sha256(ROOT / path) == expected
    assert sha256(config["source_manifest_path"]) == config["source_manifest_sha256"]
    dataset = VerifiedSources(config["source_manifest_path"])
    torch.set_num_threads(4)
    model = build_model(config["arm"], config["seed"])
    model.load_state_dict(torch.load(folder / "best_model.pt", map_location="cpu", weights_only=True), strict=True)
    model.to(config["device"])
    metrics, predictions = score_source(model, dataset, config["device"])
    assert len(predictions) == 1200
    assert predictions == read(folder / "selected_source_predictions.json")
    assert metrics == read(folder / "history.json")[status["best_epoch"] - 1]["source_validation"]
    result = {"scope": "One-epoch source trainer integration pilot, not a full baseline result",
              "checkpoint_strictly_loaded": True, "source_prediction_rows_verified": len(predictions),
              "all_predictions_and_metrics_equal": True, "target_data_loaded": False,
              "status_sha256": sha256(folder / "status.json"),
              "source_manifest_sha256": config["source_manifest_sha256"],
              "checkpoint_sha256": status["best_model_sha256"],
              "optimizer_updates_executed": status["optimizer_updates"],
              "selected_epoch": status["best_epoch"], "source_metrics": metrics}
    write(ROOT / "docs/revision/evidence/source_pilot_verification.json", result)
    print({k: v for k, v in result.items() if k != "source_metrics"})


if __name__ == "__main__":
    main()
