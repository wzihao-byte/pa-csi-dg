from __future__ import annotations

import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.wimans_20261014.adapter import (
    Manifest, ProcessedInput, ROOMS, file_sha256, require_formal_input, sampling_schedule, schedule_hash,
)
from experiments.wimans_20261014.frozen_models.control_components import control_pair_masks
from experiments.wimans_20261014.metrics import confusion_metrics
from experiments.wimans_20261014.models import make_model, objective, paired_initialization_audit, source_coral
from experiments.wimans_20261014.prepare_configs import base_config
from experiments.wimans_20261014.training import (
    atomic_torch, choose_last_best, recorded_call, restore_rng, rng_state, seed_all, train_source, validate_config,
)

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs/revision/wimans_iotj_ojcoms_20261014"


class MetadataDataset:
    def __init__(self, manifest, target):
        ids = manifest.folds[target]["sample_ids"]["train"]
        self.labels = np.asarray([int(manifest.by_id[x]["class_id"]) for x in ids])
        self.domains = np.asarray([ROOMS.index(manifest.by_id[x]["environment"]) for x in ids])
        self.numeric_ids = np.asarray([manifest.ids.index(x) for x in ids])
    def __len__(self):
        return len(self.labels)


class ImplementationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.manifest = Manifest(DOCS / "subset_manifest.csv", DOCS / "fold_manifest.json")

    def test_manifest_fixed_membership_and_reject_duplicate(self):
        for target in ROOMS:
            fold = self.manifest.folds[target]
            self.assertEqual([len(fold["sample_ids"][part]) for part in ("train", "validation", "test")], [792, 396, 594])
        payload = json.loads((DOCS / "fold_manifest.json").read_text())
        payload["folds"][0]["sample_ids"]["train"][1] = payload["folds"][0]["sample_ids"]["train"][0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad_folds.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "Duplicate"):
                Manifest(DOCS / "subset_manifest.csv", path)

    def test_sampler_fixed_budget_balance_membership_and_seed(self):
        dataset = MetadataDataset(self.manifest, "classroom")
        schedule = sampling_schedule(dataset, 42)
        self.assertEqual(schedule.shape, (30000, 32))
        np.testing.assert_array_equal(schedule, sampling_schedule(dataset, 42))
        self.assertNotEqual(schedule_hash(dataset, schedule), schedule_hash(dataset, sampling_schedule(dataset, 52)))
        for batch in schedule[:600]:
            for domain in np.unique(dataset.domains):
                selected = batch[dataset.domains[batch] == domain]
                self.assertEqual(len(selected), 16)
                counts = np.bincount(dataset.labels[selected], minlength=9)
                self.assertEqual(sorted(counts.tolist()), [1, 1, 2, 2, 2, 2, 2, 2, 2])
        self.assertTrue(set(dataset.numeric_ids[schedule].flatten()) <= set(dataset.numeric_ids))

    def test_factory_capacity_shared_initialization_rng_and_850_shape(self):
        for seed in (42, 52, 62):
            audit = paired_initialization_audit(seed)
            self.assertEqual(audit["model_parameter_count"], 4787065)
            self.assertTrue(audit["shared_state_tensors_exactly_equal"])
            self.assertEqual(audit["shared_state_tensor_count"], 358)
            make_model("tpp", seed)
            reference_rng = torch.get_rng_state()
            model = make_model("global_mlp", seed)
            self.assertTrue(torch.equal(reference_rng, torch.get_rng_state()))
        model.eval()
        with torch.no_grad():
            output = model(torch.ones(1, 850, 90), torch.ones(1, 850, 90))
        self.assertEqual(tuple(output["logits"].shape), (1, 9))
        self.assertTrue(torch.isfinite(output["logits"]).all())

    def test_pseudo_tx_slices_synchronous_and_hybrid_identity_rule(self):
        model = make_model("tpp", 42)
        amp = torch.arange(90).float().reshape(1, 1, 90).expand(2, 3, 90)
        phase = amp + 900
        a, p = model.view_builder(amp), model.view_builder(phase)
        self.assertEqual(tuple(a.shape), (2, 3, 3, 90))
        for view in range(3):
            mask = torch.zeros(90, dtype=torch.bool)
            mask[view * 30:(view + 1) * 30] = True
            self.assertTrue(torch.equal(a[:, view, :, mask], amp[:, :, mask]))
            self.assertTrue(torch.equal(p[:, view, :, mask], phase[:, :, mask]))
            self.assertEqual(a[:, view, :, ~mask].count_nonzero(), 0)
            self.assertEqual(p[:, view, :, ~mask].count_nonzero(), 0)
        features = torch.randn(3, 4, 8)
        _flat, positive, denominator = control_pair_masks(features, torch.tensor([1, 1, 1]),
                        torch.tensor([0, 0, 1]), torch.tensor([100, 101, 100]))
        self.assertTrue(positive[0, 4])  # same-class full/full, same source permitted
        self.assertFalse(positive[1, 5])  # distinct record masked/masked same class
        self.assertTrue(positive[1, 9])  # repeated record identity, same view index
        self.assertTrue(positive[1, 11])  # same record across arbitrary views
        self.assertTrue(denominator[1, 5])  # historical denominator retains unselected same class
        self.assertFalse(positive.diagonal().any())

    def test_exposure_forward_bn_rng_and_hybrid_gradient(self):
        inputs = torch.randn(4, 80, 90)
        batch = (inputs, inputs * 0.1, torch.tensor([0, 1, 0, 1]), torch.tensor([0, 0, 1, 1]), torch.arange(4))
        exposed = make_model("tpp", 42)
        hybrid = make_model("tpp", 42)
        ce = make_model("tpp", 42)
        seed_all(99)
        loss_exposure, parts_exposure = objective(exposed, batch, "W10", masked_forward_exposure=True)
        rng_exposure = torch.get_rng_state()
        loss_exposure.backward()
        seed_all(99)
        loss_hybrid, parts_hybrid = objective(hybrid, batch, "W11")
        rng_hybrid = torch.get_rng_state()
        loss_hybrid.backward()
        seed_all(99)
        loss_ce, _parts_ce = objective(ce, batch, "W10")
        self.assertTrue(torch.equal(rng_exposure, rng_hybrid))
        self.assertAlmostEqual(parts_exposure["ce"], parts_hybrid["ce"], places=6)
        self.assertAlmostEqual(float(loss_ce), float(loss_exposure), places=6)
        self.assertEqual(parts_exposure["supcon"], 0)
        self.assertGreater(parts_hybrid["supcon"], 0)
        bn_name = next(name for name in exposed.state_dict() if name.endswith("num_batches_tracked"))
        self.assertEqual(int(exposed.state_dict()[bn_name]), 2)
        self.assertEqual(int(hybrid.state_dict()[bn_name]), 2)
        self.assertEqual(int(ce.state_dict()[bn_name]), 1)
        for name in exposed.state_dict():
            if "running_mean" in name or "running_var" in name:
                self.assertTrue(torch.equal(exposed.state_dict()[name], hybrid.state_dict()[name]))
        self.assertTrue(all(p.grad is None for p in exposed.projection_head.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in hybrid.projection_head.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in hybrid.parameters()))

    def test_coral_formula_unbiased_and_gradient(self):
        features = torch.tensor([[0., 0.], [2., 4.], [1., 2.], [5., 8.]], requires_grad=True)
        domains = torch.tensor([0, 0, 1, 1])
        loss = source_coral(features, domains)
        x, y = features[:2], features[2:]
        expected = (x.mean(0) - y.mean(0)).square().mean() + (torch.cov(x.T) - torch.cov(y.T)).square().mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(float(features.grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            source_coral(features[:2], torch.tensor([0, 1]))

    def test_fixed_nine_class_metrics_and_last_tie(self):
        perfect = confusion_metrics(np.arange(9), np.arange(9))
        self.assertEqual(perfect["f1_macro"], 1)
        self.assertEqual(perfect["accuracy"], 1)
        collapse = confusion_metrics(np.arange(9), np.zeros(9, dtype=np.int64))
        self.assertAlmostEqual(collapse["f1_macro"], 0.2 / 9)
        self.assertAlmostEqual(collapse["accuracy"], 1 / 9)
        self.assertEqual(len(collapse["confusion_matrix"]), 9)
        history = [{"epoch": i + 1, "validation": {"accuracy": x}} for i, x in enumerate((0.5, 0.8, 0.6, 0.8))]
        self.assertEqual(choose_last_best(history), (4, 0.8))

    def test_formal_gate_unresolved_preview_and_target_gate(self):
        with self.assertRaises(RuntimeError):
            require_formal_input({"status": "blocked_axis_export", "full_input_qa_passed": False})
        with self.assertRaises(RuntimeError):
            require_formal_input({"status": "passed", "full_input_qa_passed": True,
                                  "axis_export_verification": {"status": "unresolved"}})
        with self.assertRaises(RuntimeError):
            require_formal_input({"status": "passed_storage_order_v2", "full_input_qa_passed": True})
        inputs = ProcessedInput.__new__(ProcessedInput)
        inputs.manifest, inputs.candidate_source_qa = self.manifest, False
        with self.assertRaisesRegex(RuntimeError, "strict reloaded"):
            inputs.dataset("classroom", "test")
        inputs.candidate_source_qa = True
        with self.assertRaisesRegex(RuntimeError, "cannot open target"):
            inputs.dataset("classroom", "test", {"strict_reloaded": True})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "input_audit.json").write_text("{}")
            config = base_config(self.manifest, path, "W11", "classroom", 42)
            with self.assertRaisesRegex(RuntimeError, "not frozen"):
                validate_config(config, formal=True)
            config_path = path / "candidate.json"
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(RuntimeError, "not frozen"):
                train_source(config_path, path / "run", device="cpu")
            self.assertFalse((path / "run").exists())

    def test_storage_fallback_exact_slicing_and_audit_binding(self):
        registered = DOCS / "axis_storage_protocol.json"
        audit = {"status": "passed_storage_order_v2", "full_input_qa_passed": True, "numerical_input_qa_passed": True,
                 "expected_rows": 1782, "candidate_rows": 1782, "completed_rows": 1782, "preview": False, "failures": [],
                 "axis_storage_protocol": {"path": str(registered), "sha256": file_sha256(registered)},
                 "representation_version": "storage_order_v2", "phase_representation": "native_historical_feature_unwrap_v1",
                 "loader_unwrap_phase": False, "axis_export_verification": {"status": "storage_order_protocol_registered"}}
        require_formal_input(audit)  # Registered storage protocol is independently acceptable, not physical Tx0 verification.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            protocol = json.loads(registered.read_text())
            protocol["selected_first_stored_axis_index"] = 1
            changed = path / "changed_protocol.json"
            changed.write_text(json.dumps(protocol))
            altered = copy.deepcopy(audit)
            altered["axis_storage_protocol"] = {"path": str(changed), "sha256": file_sha256(changed)}
            with self.assertRaisesRegex(RuntimeError, "slicing drift"):
                require_formal_input(altered)
            altered = copy.deepcopy(audit)
            altered["manifest_sha256"] = "wrong"
            altered["fold_manifest_sha256"] = file_sha256(DOCS / "fold_manifest.json")
            (path / "input_audit.json").write_text(json.dumps(altered))
            with self.assertRaisesRegex(RuntimeError, "manifest binding"):
                ProcessedInput(path, self.manifest)

    def test_rng_optimizer_checkpoint_restore_replays_next_update_exactly(self):
        # Nontrivial Adam moments + dropout + BN verify replay, not just weight reload.
        seed_all(123)
        model = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.BatchNorm1d(7),
                                    torch.nn.Dropout(0.2), torch.nn.Linear(7, 9))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        data, labels = torch.randn(4, 5), torch.tensor([0, 1, 2, 3])
        def update():
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(data), labels)
            loss.backward()
            optimizer.step()
            return float(loss.detach())
        update()
        checkpoint = {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": rng_state()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            atomic_torch(path, checkpoint)
            first_loss = update()
            expected = copy.deepcopy(model.state_dict())
            expected_python, expected_numpy = __import__("random").random(), np.random.random()
            restored = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(restored["model"], strict=True)
            optimizer.load_state_dict(restored["optimizer"])
            restore_rng(restored["rng"])
            second_loss = update()
            self.assertEqual(first_loss, second_loss)
            self.assertEqual(__import__("random").random(), expected_python)
            self.assertEqual(np.random.random(), expected_numpy)
            self.assertTrue(all(torch.equal(model.state_dict()[name], value) for name, value in expected.items()))

    def test_runner_first_epoch_interruption_has_zero_boundary_and_replays_rng(self):
        # Exercise the actual runner's initialization/resume branches with tiny
        # synthetic source data; no formal gate is bypassable outside this test.
        class TinyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4
            def __getitem__(self, index):
                return torch.ones(2, 3), torch.ones(2, 3), index % 2, index // 2, index
        class TinyInput:
            signal_access = []
            def dataset(self, _target, _partition):
                return TinyDataset()
        samples = []
        def interrupted_objective(*args, **kwargs):
            samples.append(torch.rand(1).item())
            raise RuntimeError("synthetic interruption before first update")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = {"target_environment": "classroom", "seed": 42, "losses": {"pool": "tpp"},
                      "sampling_schedule_sha256": "synthetic", "arm": "W11", "gamma": 0.0}
            config_path = path / "config.json"
            config_path.write_text(json.dumps(config))
            with patch("experiments.wimans_20261014.training.validate_config"), \
                 patch("experiments.wimans_20261014.training.inputs_for", return_value=TinyInput()), \
                 patch("experiments.wimans_20261014.training.sampling_schedule", return_value=np.tile(np.arange(4), (30000, 8))), \
                 patch("experiments.wimans_20261014.training.schedule_hash", return_value="synthetic"), \
                 patch("experiments.wimans_20261014.training.make_model", side_effect=lambda *args: torch.nn.Linear(3, 9)), \
                 patch("experiments.wimans_20261014.training.objective", side_effect=interrupted_objective):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    train_source(config_path, path / "run", device="cpu")
                latest = torch.load(path / "run/latest.pt", map_location="cpu", weights_only=False)
                self.assertEqual(latest["completed_epochs"], 0)
                self.assertEqual(latest["optimizer_updates"], 0)
                self.assertEqual(latest["history"], [])
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    train_source(config_path, path / "run", device="cpu", resume=True)
                self.assertEqual(samples[0], samples[1])

    def test_attempt_failure_and_retry_receipts_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            def interrupted():
                raise RuntimeError("synthetic failure")
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                recorded_call(path, "synthetic", interrupted)
            recorded_call(path, "synthetic", lambda: None)
            receipts = [json.loads(p.read_text()) for p in (path.parent / "attempts").glob("*.json")]
            self.assertEqual(len(receipts), 2)
            self.assertEqual({row["status"] for row in receipts}, {"failed_or_interrupted", "completed"})
            self.assertTrue(all(row["elapsed_seconds"] >= 0 for row in receipts))


if __name__ == "__main__":
    unittest.main()
