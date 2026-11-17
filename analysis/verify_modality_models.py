"""Verify historical phase ordering and the paired two-stream capacity control."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dg_dataset import unwrap_phase_array
from experiments.revision_validation.experiment_data import VerifiedInputs
from experiments.revision_validation.models import build_model, count_matmul_conv_macs
from experiments.revision_validation.modality_models import build_twin_model, historical_feature_phase
from experiments.revision_validation.source_data import epoch_schedule
from experiments.revision_validation.study_engine import equal_state, runtime_settings
from experiments.revision_validation.study_io import now, sha256, write


class TwinCounter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, value, *, project=False):
        return self.model(value, secondary=value, project=project)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous modality QA")
    runtime_settings()
    source = ROOT / "experiments/revision_validation/verified_protocol/source_E1.json"
    dataset = VerifiedInputs(source, ("amp", "phase"))
    batch = epoch_schedule(dataset.manifest["train"], 401, 0, steps=1)[0]
    inputs, labels, rows = dataset.batch("train", batch, "cpu")
    original = np.stack([dataset.arrays[r["environment"]]["phase"][r["index"]] for r in rows])
    expected = unwrap_phase_array(original, axis=-1).astype(np.float32)
    phase = historical_feature_phase(original)
    assert np.array_equal(expected, phase)
    cast_first = np.unwrap(original.astype(np.float32), axis=-1)
    historical = {"legacy_loader_values_exact": True, "stored_dtype": str(original.dtype), "output_dtype": str(phase.dtype),
                  "cast_before_unwrap_max_error": float(np.max(np.abs(cast_first-phase))), "legacy_loader_sha256": sha256(ROOT / "dg_dataset.py")}
    inputs["phase"] = torch.from_numpy(phase)
    seed = 9401
    single, twin_a, twin_p = build_model("R10", seed), build_twin_model(seed), build_twin_model(seed)
    assert equal_state(twin_a.state_dict(), twin_p.state_dict())
    common = [key for key in single.state_dict() if not key.startswith("representation.0.")]
    assert all(torch.equal(single.state_dict()[key], twin_a.state_dict()[key]) for key in common)
    results = {}
    for name, model, kwargs in (("historical_phase_only", single, {"primary": inputs["phase"]}),
                               ("amplitude_amplitude", twin_a, {"primary": inputs["amp"], "secondary": inputs["amp"]}),
                               ("amplitude_phase", twin_p, {"primary": inputs["amp"], "secondary": inputs["phase"]})):
        torch.manual_seed(9402)
        if "secondary" in kwargs:
            output = model(kwargs["primary"], secondary=kwargs["secondary"], project=False)
        else:
            output = model(kwargs["primary"], project=False)
        loss = F.cross_entropy(output["logits"], labels)
        loss.backward()
        assert output["logits"].shape == (16, 6) and output["feature"].shape == (16, 256)
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        results[name] = {"parameters": sum(p.numel() for p in model.parameters()),
                         "active_CE_parameters": sum(p.numel() for p in model.parameters() if p.grad is not None),
                         "objective": float(loss.detach()), "optimizer_updates": 0}
    counts = {"single": count_matmul_conv_macs(single, torch.zeros(1,850,90)),
              "twin": count_matmul_conv_macs(TwinCounter(twin_a), torch.zeros(1,850,90))}
    report = {"passed": True, "checked_at": now(), "historical_phase": historical,
              "amp_amp_and_amp_phase_all_initial_tensors_exact": True, "single_twin_common_tensors_exact": len(common),
              "secondary_seed_offset": 100000, "fusion_seed_offset": 200000,
              "checks": results, "counted_macs": counts, "target_data_loaded": False,
              "training_completed": False, "opened_source_paths": dataset.opened_paths, "source_sample_ids": [r["sample_id"] for r in rows],
              "code_sha256": {name: sha256(ROOT / name) for name in ("analysis/verify_modality_models.py",
                    "experiments/revision_validation/modality_models.py", "experiments/revision_validation/models.py",
                    "experiments/revision_validation/experiment_data.py", "dg_dataset.py")}}
    write(args.output, report)
    print({"passed": True, "checks": results, "historical_phase": historical, "twin_counted_macs": counts["twin"]["total_counted_macs"]})


if __name__ == "__main__":
    main()
