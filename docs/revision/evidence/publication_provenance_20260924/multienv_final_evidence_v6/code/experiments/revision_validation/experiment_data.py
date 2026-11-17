"""Hash-bound source-only inputs for controls and separately prepared datasets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from experiments.revision_validation.study_io import read, sha256


class VerifiedInputs:
    def __init__(self, manifest_path, modalities=("amp",)):
        self.path = Path(manifest_path)
        self.manifest = read(self.path)
        self.target = self.manifest["outer_target"]
        self.sources = self.manifest["source_environments"]
        self.modalities = tuple(modalities)
        if len(self.sources) != 2 or self.target in self.sources or set(self.manifest["files"]) != set(self.sources):
            raise ValueError("Exactly two source environments, excluding the target, are required")
        if self.manifest["inner_validation"] != "subject_grouped_holdout":
            raise ValueError("Verified subject-grouped source validation is required")
        if not self.modalities or not set(self.modalities) <= {"amp", "phase"}:
            raise ValueError("Unsupported input modality")
        self.arrays, self.opened_paths, labels = {}, [], {}
        for environment, files in self.manifest["files"].items():
            self.arrays[environment] = {}
            for kind in self.modalities + ("label",):
                specification = files["data_files"][kind]
                path = Path(specification["path"])
                if sha256(path) != files["actual_file_sha256"][kind]:
                    raise ValueError(f"Source input hash mismatch: {environment}/{kind}")
                self.opened_paths.append(str(path))
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if list(array.shape) != specification["shape"] or str(array.dtype) != specification["dtype"]:
                    raise ValueError("Source shape/dtype differs from manifest")
                if kind == "label":
                    labels[environment] = np.asarray(array).reshape(-1)
                else:
                    if array.shape[1:] != (850, 90):
                        raise ValueError("This experiment expects 850 frames and 90 input columns")
                    self.arrays[environment][kind] = array
        for part in ("train", "val"):
            rows = self.manifest[part]
            if not rows or len({r["sample_id"] for r in rows}) != len(rows):
                raise ValueError("Empty partition or duplicate sample identity")
            if {r["label"] for r in rows} != set(range(6)):
                raise ValueError("Both source partitions must contain all six declared classes")
            for row in rows:
                if row["environment"] not in self.sources or int(labels[row["environment"]][row["index"]]) != row["label"]:
                    raise ValueError("Source row environment or label mismatch")
        for key in ("sample_id", "subject_id", "acquisition_id"):
            if {r[key] for r in self.manifest["train"]} & {r[key] for r in self.manifest["val"]}:
                raise ValueError(f"Source train/validation overlap: {key}")

    def batch(self, part, positions, device):
        rows = [self.manifest[part][int(position)] for position in positions]
        inputs = {}
        for kind in self.modalities:
            values = np.stack([self.arrays[r["environment"]][kind][r["index"]] for r in rows]).astype(np.float32)
            inputs[kind] = torch.from_numpy(values).to(device)
        labels = torch.tensor([r["label"] for r in rows], dtype=torch.long, device=device)
        return inputs, labels, rows
