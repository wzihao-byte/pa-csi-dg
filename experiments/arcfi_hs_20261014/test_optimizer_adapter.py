"""Meaningful synthetic CPU checks of the actual pinned author class repair."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from .optimizer_adapter import (DEFAULT_AUTHOR_SOURCE, PINNED_FILES_SHA256, correct_fresh_optimizer_coverage,
                                load_pinned_author_class, make_cpu_prototype, parameter_coverage,
                                require_exact_parameter_coverage)


def author_config():
    return SimpleNamespace(csidataset="Widar3.0", backbone="ResNet", device="cpu", last_dim=512,
                           classify="linear", num_classes=6, optimizer="Adam", lr=0.003,
                           weight_decay=0.0003, temperature=0.07, beta=0.5, scale_cap=6, ratio=None)


class OptimizerAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.author_class = load_pinned_author_class()

    def test_repair_preserves_state_functions_predictions_and_hyperparameters(self):
        torch.manual_seed(42)
        model = self.author_class(author_config(), None, 30000)
        model.eval()
        state = copy.deepcopy(model.state_dict())
        methods = {name: getattr(type(model), name) for name in ("update", "forward", "forward_f", "randomize", "predict")}
        classifier_optimizer, classifier_scheduler = model.optimizer_c, model.scheduler_c
        feature_defaults = dict(model.optimizer_f.defaults)
        inputs, labels = torch.randn(4, 2, 2, 64, 30), torch.tensor([0, 0, 1, 1])
        with torch.no_grad():
            before_prediction, before_loss = model.predict(inputs, labels)
        omitted_before = [name for name, count in parameter_coverage(model).items() if count == 0]
        self.assertEqual(omitted_before, ["neck.0.weight", "neck.0.bias", "neck.2.weight", "neck.2.bias"])
        receipt = correct_fresh_optimizer_coverage(model)
        self.assertEqual(receipt["corrected_neck_parameter_count"], 328320)
        self.assertEqual(receipt["parameter_count"], 11502470)
        self.assertTrue(all(count == 1 for count in require_exact_parameter_coverage(model).values()))
        self.assertIs(model.optimizer_c, classifier_optimizer)
        self.assertIs(model.scheduler_c, classifier_scheduler)
        self.assertIs(model.scheduler_f.optimizer, model.optimizer_f)
        self.assertEqual(model.optimizer_f.defaults, feature_defaults)
        self.assertEqual(model.scheduler_f.T_max, 30000)
        self.assertEqual(model.scheduler_f.eta_min, 0)
        self.assertTrue(all(torch.equal(model.state_dict()[name], value) for name, value in state.items()))
        self.assertTrue(all(getattr(type(model), name) is function for name, function in methods.items()))
        with torch.no_grad():
            after_prediction, after_loss = model.predict(inputs, labels)
        self.assertTrue(torch.equal(before_prediction, after_prediction))
        self.assertTrue(torch.equal(before_loss, after_loss))

    def test_actual_author_update_changes_neck_and_optimizer_zero_grad_clears_it(self):
        model, receipt = make_cpu_prototype(seed=42)
        model.train()
        before = {name: p.detach().clone() for name, p in model.neck.named_parameters()}
        inputs, labels = torch.randn(4, 2, 2, 64, 30), torch.tensor([0, 0, 1, 1])
        predictions, losses, lr_f, lr_c = model.update(inputs, labels, torch.tensor([0, 1, 0, 1]), torch.arange(4), 0, None)
        self.assertEqual(tuple(predictions.shape), (4, 6))
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in losses.values()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.neck.parameters()))
        self.assertTrue(all(not torch.equal(before[name], p.detach()) for name, p in model.neck.named_parameters()))
        self.assertEqual(model.scheduler_f.last_epoch, 1)
        self.assertEqual(model.scheduler_c.last_epoch, 1)
        self.assertEqual(lr_f, lr_c)
        model.optimizer_f.zero_grad(set_to_none=True)
        model.optimizer_c.zero_grad(set_to_none=True)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertFalse(receipt["formal_training_qualified"])

    def test_duplicate_ownership_and_after_update_repair_fail_closed(self):
        model, _receipt = make_cpu_prototype(seed=42)
        model.optimizer_f.add_param_group({"params": [next(model.classifier.parameters())]})
        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            require_exact_parameter_coverage(model)
        torch.manual_seed(42)
        fresh = self.author_class(author_config(), None, 30000)
        fresh.optimizer_f.step()
        fresh.scheduler_f.step()
        with self.assertRaisesRegex(RuntimeError, "precede all"):
            correct_fresh_optimizer_coverage(fresh)

    def test_modified_author_source_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in PINNED_FILES_SHA256:
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((DEFAULT_AUTHOR_SOURCE / relative).read_bytes())
            modified = root / "models/contrast_methods.py"
            modified.write_bytes(modified.read_bytes() + b"\n# source drift\n")
            with self.assertRaisesRegex(RuntimeError, "Pinned author source changed"):
                load_pinned_author_class(root)


if __name__ == "__main__":
    unittest.main()
