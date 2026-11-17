"""CPU-only independent gates for the original-H grouped-source supplement.

The real-data smoke compares two updates of the unmodified historical backend
against the adapter, at the full batch size and model size. It is a source-only
implementation equivalence check; it does not tune or shorten a formal run.
"""
from __future__ import annotations

import argparse
import copy
import functools
import gc
import hashlib
import itertools
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from common import (ARMS, BUNDLE, PARAMETERS, TARGETS, grouped_split, legacy,
                    metadata_dataset, read, replay_sampler, require, sha,
                    state_hash, verify_bundle, verify_config, write_new)
from train_factorial import install


# Calculated independently using the original continuous-global-RNG sampler,
# then anchored by replaying a sealed historical schedule (see below).
EXPECTED_SCHEDULES = {
    ("E1", 42): "65679e2c062f7af78cef4621952552ff931c14b5362b9fe29fb02e165f41c964",
    ("E1", 52): "a9789d7913a6d1459de414904fc3bbddbcd268d06c7f4ce54c970fcd27d1a093",
    ("E1", 62): "45c96778dfba9e0fbfcc7906c41f440e93821f3581149e01c5478bed15084fd8",
    ("E2", 42): "3f47568ec3742aacf07da23ad9bf1a301cd7588787e4458069f33113ef3ec2d6",
    ("E2", 52): "8325757890d50373e317221a5e95b30a2df97bda6c46b835479fdb4f26ecba06",
    ("E2", 62): "be8642be9af5e9e8643f37002cfbe3114127f849015f28526b99092ac99029c9",
    ("E3", 42): "17ebb62d75e9f8a9e27f4609e069c6dd562420a26973ffaa15ab03434003f1a8",
    ("E3", 52): "4a59cd909fc14547ff85a8040264198abeeb86b1a72e87264a0d3efdfd4b1c10",
    ("E3", 62): "1188e5808e09576a78529bba4a118f2815098e28405e9924a0ec433bc6ca9e86",
}
HISTORICAL_SCHEDULE = "d29837557c9b4847e3c875a3be1c074db7e80c4e40303c1c39a9e5dc24e7b7b0"


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def expect_failure(name, callback):
    try:
        callback()
    except (RuntimeError, ValueError, KeyError, FileNotFoundError) as error:
        return {"case": name, "rejected": True, "reason": str(error)}
    raise RuntimeError(f"Negative QA case was accepted: {name}")


def nested_hash(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            a = item.detach().cpu().contiguous().numpy()
            digest.update(b"tensor" + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update((type(item).__name__ + ":" + repr(item)).encode())

    visit(value)
    return digest.hexdigest()


class TwoBatches:
    """Keep the actual historical DataLoader and sampler iterator intact."""

    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return 2

    def __iter__(self):
        yield from itertools.islice(iter(self.loader), 2)


class SourceSampleCache:
    """Full source index space backed only by the two batches accessed in QA."""

    def __init__(self, metadata, selected, amplitudes, phases, dataset_module):
        self.env_names = list(metadata.env_names)
        self.labels = metadata.labels.copy()
        self.env_ids = metadata.env_ids.copy()
        self.sample_ids = metadata.sample_ids.copy()
        self.input_dim, self.num_classes, self.time_steps = 90, 6, 850
        self._positions = {int(index): offset for offset, index in enumerate(selected)}
        self._items = dataset_module.PaCsiArrayDataset(
            amplitudes, phases, self.labels[selected], self.env_ids[selected],
            self.env_names, self.sample_ids[selected])
        self._original_indices_by_env = dataset_module.PaCsiArrayDataset.indices_by_env
        self.accessed = []

    def __len__(self):
        return 6000

    def indices_by_env(self, indices):
        return self._original_indices_by_env(self, indices)

    def __getitem__(self, index):
        require(int(index) in self._positions, "Smoke sampled an unexpected source item")
        self.accessed.append(int(self.sample_ids[index]))
        return self._items[self._positions[int(index)]]


def real_source_batches(config, modules, first_two_global_batches):
    """Read source signals only; keep historical float64 unwrap -> float32."""
    source = read(config["h_grouped"]["source_manifest"])
    metadata = metadata_dataset(source)
    lookup = {int(sample): index for index, sample in enumerate(metadata.sample_ids)}
    selected = sorted({lookup[int(sample)] for sample in first_two_global_batches.reshape(-1)})
    dd = modules["dg_dataset"]
    amplitudes = np.empty((len(selected), 850, 90), dtype=np.float32)
    phases = np.empty_like(amplitudes)
    opened = []
    for domain, env in enumerate(metadata.env_names):
        require(env != config["target_env"], "Target input requested during CPU QA")
        positions = [offset for offset, index in enumerate(selected) if metadata.env_ids[index] == domain]
        local = [selected[offset] - domain * 3000 for offset in positions]
        files = config["data"]["env_files"][env]
        amp = np.load(files["amp"], mmap_mode="r", allow_pickle=False)
        phase = np.load(files["phase"], mmap_mode="r", allow_pickle=False)
        labels = np.load(files["label"], mmap_mode="r", allow_pickle=False)
        require(amp.shape == phase.shape == (3000, 850, 90), "Historical input dimensions changed")
        require(str(amp.dtype) == str(phase.dtype) == "float64", "Historical input dtype changed")
        require(np.array_equal(labels[local], metadata.labels[np.asarray(selected)[positions]]), "Source labels disagree with manifest")
        raw_phase = np.asarray(phase[local])
        # Unwrapping is along features, so selecting samples first is identical
        # to the historical full-array operation; no statistic crosses samples.
        unwrapped = dd.unwrap_phase_array(raw_phase, axis=config["data"]["phase_unwrap_axis"])
        transformed_phase = dd.ensure_time_feature_layout(unwrapped)
        require(np.array_equal(transformed_phase, np.unwrap(raw_phase, axis=-1).astype(np.float32)), "Historical phase processing differs")
        amplitudes[positions] = dd.ensure_time_feature_layout(np.asarray(amp[local]))
        phases[positions] = transformed_phase
        opened.extend(str(Path(files[k]).resolve()) for k in ("amp", "phase", "label"))
        del amp, phase, labels, raw_phase, unwrapped, transformed_phase
    require(np.isfinite(amplitudes).all() and np.isfinite(phases).all(), "Nonfinite source inputs")
    return metadata, selected, amplitudes, phases, opened


def trajectory(config, data, run_dir, adapted):
    cfg = copy.deepcopy(config)
    cfg["output_dir"] = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    base = legacy()
    if adapted:
        base = install(base, cfg)
    modules = base.configure_imports(cfg)
    trainer, dd = modules["train_dg"], modules["dg_dataset"]
    metadata, selected, amplitudes, phases, _opened = data
    dataset = SourceSampleCache(metadata, selected, amplitudes, phases, dd)
    source = read(cfg["h_grouped"]["source_manifest"])
    reference_split, _identity = grouped_split(metadata, source, cfg["target_env"])
    split = base.source_split(dataset, cfg["seed_list"][0], cfg["target_env"], dd) if adapted else reference_split
    require(split == reference_split, "Adapter changed grouped membership")
    trainer.set_seed(cfg["seed_list"][0])
    trainer.PaCsiDGLite = base.model_class(cfg, modules)
    # The actual legacy loader and original Python-global balanced sampler run
    # here, including DataLoader's own Torch RNG draw when iteration starts.
    loader = trainer.build_data_loader(dataset, split["train_indices"], 32, 0,
                                       shuffle=False, sampler_mode="domain_class_balanced")
    model, optimizer, scheduler, scheduler_mode = trainer.build_model_and_optimizer(dataset, cfg, torch.device("cpu"))
    require(sum(p.numel() for p in model.parameters()) == PARAMETERS, "Smoke model size changed")
    initial = state_hash(model)
    require(initial == cfg["h_grouped"]["initial_state_sha256"], "Smoke initialization differs from frozen reference")
    require(all(p.device.type == "cpu" for p in model.parameters()), "CPU QA allocated a CUDA model")
    loss = trainer.DomainAwareSupConLoss(
        temperature=cfg["losses"]["temperature"],
        include_same_domain_same_class=cfg["losses"]["include_same_domain_same_class"],
        positive_mode=cfg["losses"]["supcon_positive_mode"])
    trace = []
    original_step = optimizer.step

    @functools.wraps(original_step)
    def step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        trace.append({"step": len(trace) + 1, "model_sha256": state_hash(model),
                      "optimizer_sha256": nested_hash(optimizer.state_dict()),
                      "torch_rng_sha256": nested_hash(torch.random.get_rng_state()),
                      "python_rng_sha256": nested_hash(random.getstate()),
                      "learning_rate_before_scheduler": optimizer.param_groups[0]["lr"]})
        emit("cpu_update", arm=cfg["factorial"]["arm"], adapted=adapted, step=len(trace))
        return result

    optimizer.step = step
    loss_config = trainer.loss_config_for_lambda(cfg["losses"], trainer.scheduled_lambda_supcon(cfg["losses"], 1))
    loss_config["grad_clip_norm"] = cfg["training"]["grad_clip_norm"]
    metrics = trainer.run_epoch(model, TwoBatches(loader), optimizer, scheduler,
                               scheduler_mode, torch.device("cpu"), loss_config,
                               cfg["losses"]["contrastive_loss_type"], loss, None, None, train=True)
    require(len(trace) == 2 and len(dataset.accessed) == 64, "CPU smoke did not perform exactly two full batches")
    require(np.isfinite(metrics["loss_total"]), "CPU smoke produced nonfinite loss")
    if cfg["factorial"]["arm"] == "H11":
        require(metrics["supcon_frac_anchors_with_positive"] == 1.0, "Historical hybrid lost positive coverage")
    result = {"initial_state_sha256": initial, "updates": trace,
              "final_scheduler_sha256": nested_hash(scheduler.state_dict()),
              "final_learning_rate": optimizer.param_groups[0]["lr"],
              "accessed_global_sample_ids": dataset.accessed, "metrics": metrics}
    if adapted:
        audit = read(run_dir / "initialization_audit.json")
        require(audit["initial_state_sha256"] == initial and audit["parameter_count"] == PARAMETERS, "Adapter initialization audit disagrees")
    del model, optimizer, scheduler, loader, loss, dataset
    gc.collect()
    return result


def negative_gates(config, work):
    records = []
    for name, mutate in (
        ("changed_supcon_weight", lambda c: c["losses"].update(lambda_supcon=0.123)),
        ("changed_tpp_levels", lambda c: c["model"].update(tpp_levels=[1, 2, 4])),
        ("changed_checkpoint_metric", lambda c: c["training"].update(selection_metric="f1_macro")),
    ):
        changed = copy.deepcopy(config)
        mutate(changed)
        records.append(expect_failure(name, lambda: verify_config(changed)))
    source = read(config["h_grouped"]["source_manifest"])
    metadata = metadata_dataset(source)
    for key in ("sample_id", "subject_id", "acquisition_id"):
        changed = copy.deepcopy(source)
        if key == "sample_id":
            changed["val"][0] = copy.deepcopy(changed["train"][0])
        else:
            changed["val"][0][key] = changed["train"][0][key]
        records.append(expect_failure(f"overlapping_{key}", lambda: grouped_split(metadata, changed, config["target_env"])))
    cfg = copy.deepcopy(config)
    cfg["output_dir"] = str(work / "target_gate")
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=False)
    base = install(legacy(), cfg)
    modules = base.configure_imports(cfg)
    opened = []
    original_np_load = np.load

    def forbidden_load(*args, **kwargs):
        opened.append(str(args[0]))
        raise AssertionError("Target gate reached an array read before checkpoint freeze")

    np.load = forbidden_load
    try:
        target = cfg["target_data"]["data"]
        records.append(expect_failure("target_before_freeze", lambda: modules["dg_dataset"].load_dataset(target, modules["clean_root"])))
        mixed = copy.deepcopy(cfg["data"])
        mixed["env_files"].update(copy.deepcopy(target["env_files"]))
        records.append(expect_failure("mixed_source_target_loader", lambda: modules["dg_dataset"].load_dataset(mixed, modules["clean_root"])))
    finally:
        np.load = original_np_load
    require(not opened, "Target array was opened during negative QA")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--threads", type=int, default=2, choices=(2, 3, 4))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    bundle = args.bundle.resolve()
    manifest, registry = verify_bundle(bundle, require_qa=False)
    bundle_hash = sha(bundle / "bundle_manifest.json")
    qa_source = Path(__file__).resolve()
    frozen_self = [bundle / relative for relative in manifest["files_sha256"] if relative.replace("\\", "/").endswith("/qa.py")]
    require(len(frozen_self) == 1 and sha(frozen_self[0]) == sha(qa_source), "QA code not bound to frozen bundle")
    qa_dir = bundle.parent / "qa"
    require(not (qa_dir / "approval.json").exists(), "An approval already exists; never overwrite it")
    work = qa_dir / ("work_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    work.mkdir(parents=True, exist_ok=False)
    configs = {(j["target"], j["seed"], j["arm"]): read(bundle / j["config"]) for j in registry["jobs"]}
    first = configs[("E1", 42, "H11")]
    backend = legacy()
    modules = backend.configure_imports(first)
    source = read(first["h_grouped"]["source_manifest"])
    metadata = metadata_dataset(source)
    original_split = backend.source_split(metadata, 42, "E1", modules["dg_dataset"])
    historical_hash, _ = replay_sampler(metadata, original_split, 42, modules["dg_dataset"].DomainClassBalancedBatchSampler)
    require(historical_hash == HISTORICAL_SCHEDULE, "Sampler audit failed to reproduce sealed historical reference")
    emit("historical_sampler_passed", sha256=historical_hash)
    schedules = []
    first_two = None
    for (target, seed), expected in EXPECTED_SCHEDULES.items():
        h10, h11 = configs[(target, seed, "H10")], configs[(target, seed, "H11")]
        require(h10["model"] == h11["model"] and h10["training"] == h11["training"], "Paired model/training mismatch")
        require(h10["h_grouped"]["initial_state_sha256"] == h11["h_grouped"]["initial_state_sha256"], "Paired initialization mismatch")
        require(h10["h_grouped"]["source_manifest_sha256"] == h11["h_grouped"]["source_manifest_sha256"], "Paired source split mismatch")
        src = read(h10["h_grouped"]["source_manifest"])
        ds = metadata_dataset(src)
        split, identity = grouped_split(ds, src, target)
        python_state = random.getstate()
        schedule_hash, schedule = replay_sampler(ds, split, seed, modules["dg_dataset"].DomainClassBalancedBatchSampler)
        require(random.getstate() == python_state, "Sampler replay perturbed caller Python RNG")
        require(schedule_hash == expected, "Grouped sampler differs from independent replay")
        for cfg in (h10, h11):
            require(cfg["h_grouped"]["expected_sampler_sha256"] == expected, "Config sampler differs from independent replay")
        if (target, seed) == ("E1", 42):
            first_two = schedule[0, :2].copy()
        schedules.append({"target": target, "seed": seed, "sha256": schedule_hash,
                          "train_subjects": len(identity["train"]["subject_id"]),
                          "val_subjects": len(identity["val"]["subject_id"]),
                          "optimizer_steps": int(schedule.shape[0] * schedule.shape[1])})
        emit("grouped_sampler_passed", target=target, seed=seed)
        del schedule
    negatives = negative_gates(first, work)
    emit("negative_gates_passed", count=len(negatives))
    # Actual source files are hash checked once; no target arrays are opened.
    actual_source_hashes = backend.verify_source_files(first)
    data = real_source_batches(first, modules, first_two)
    trajectories = {}
    for arm in ARMS:
        cfg = configs[("E1", 42, arm)]
        emit("cpu_equivalence_started", arm=arm, batch_size=32, batches=2)
        baseline = trajectory(cfg, data, work / f"{arm}_historical", adapted=False)
        adapted = trajectory(cfg, data, work / f"{arm}_adapter", adapted=True)
        require(baseline == adapted, f"Historical/adapter training trajectory mismatch for {arm}")
        require(baseline["accessed_global_sample_ids"] == first_two.reshape(-1).tolist(), "Real-data smoke used a different batch order")
        trajectories[arm] = baseline
        emit("cpu_equivalence_passed", arm=arm)
    verify_bundle(bundle, require_qa=False)
    require(sha(bundle / "bundle_manifest.json") == bundle_hash, "Bundle changed during QA")
    result = {
        "schema_version": "h-l1248-grouped-independent-qa/v1", "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_manifest_sha256": bundle_hash,
        "qa_source_sha256": sha(qa_source), "qa_source": str(qa_source),
        "external_code_sha256": manifest["external_code_sha256"],
        "cpu_threads": args.threads, "torch_version": torch.__version__,
        "numpy_version": np.__version__, "python_version": sys.version,
        "historical_sampler_sha256": historical_hash,
        "grouped_schedules": schedules, "negative_gates": negatives,
        "actual_source_file_sha256": actual_source_hashes,
        "array_paths_opened_for_smoke": data[-1], "target_arrays_opened": False,
        "formal_configuration_modified": False, "cpu_training_equivalence": trajectories,
        "scope": "Two real full source batches per arm verify adapter equivalence; final run completion still requires 200 epochs and target audits.",
    }
    write_new(qa_dir / "approval.json", result)
    emit("qa_passed", approval=str(qa_dir / "approval.json"), bundle_manifest_sha256=bundle_hash)


if __name__ == "__main__":
    main()
