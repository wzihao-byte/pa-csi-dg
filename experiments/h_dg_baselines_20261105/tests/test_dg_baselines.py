"""CPU checks of the DG baseline components; no data, no training."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dg_baselines as dg  # noqa: E402

torch.set_num_threads(1)


class CoralTests(unittest.TestCase):
    def test_matches_domainbed_formula(self):
        g = torch.Generator().manual_seed(1)
        x = torch.randn(16, 8, generator=g, dtype=torch.float64)
        y = torch.randn(16, 8, generator=g, dtype=torch.float64) * 2 + 1
        features = torch.cat([x, y])
        environments = torch.tensor([0] * 16 + [1] * 16)
        penalty = dg.source_coral(features, environments)
        # Independent reference: torch.cov (unbiased) instead of the explicit centered product.
        mean_diff = ((x.mean(0) - y.mean(0)) ** 2).mean()
        cov_diff = ((torch.cov(x.t()) - torch.cov(y.t())) ** 2).mean()
        self.assertAlmostEqual(float(penalty), float(mean_diff + cov_diff), places=10)

    def test_identical_sources_give_zero(self):
        x = torch.randn(16, 8, dtype=torch.float64)
        features = torch.cat([x, x[torch.randperm(16)]])
        environments = torch.tensor([0] * 16 + [1] * 16)
        self.assertAlmostEqual(float(dg.source_coral(features, environments)), 0.0, places=12)

    def test_requires_two_sources_with_two_examples(self):
        with self.assertRaises(ValueError):
            dg.source_coral(torch.randn(4, 3), torch.tensor([0, 0, 0, 0]))
        with self.assertRaises(ValueError):
            dg.source_coral(torch.randn(3, 3), torch.tensor([0, 0, 1]))

    def test_gradient_flows_to_features(self):
        features = torch.randn(32, 6, requires_grad=True)
        environments = torch.tensor([0] * 16 + [1] * 16)
        dg.source_coral(features, environments).backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(float(features.grad.abs().sum()), 0.0)


class MixupTests(unittest.TestCase):
    def setUp(self):
        self.environments = torch.tensor([0] * 16 + [1] * 16)

    def test_plan_is_reproducible_and_step_keyed(self):
        a = dg.mixup_plan(self.environments, 42, 7)
        b = dg.mixup_plan(self.environments, 42, 7)
        c = dg.mixup_plan(self.environments, 42, 8)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
        self.assertFalse(all(torch.equal(x, y) for x, y in zip(a, c)))

    def test_two_directed_mixtures_cover_both_domains(self):
        first, second, coefficients = dg.mixup_plan(self.environments, 42, 0)
        self.assertEqual(len(first), 32)
        self.assertTrue(torch.equal(self.environments[first][:16], torch.zeros(16, dtype=torch.long)))
        self.assertTrue(torch.equal(self.environments[second][:16], torch.ones(16, dtype=torch.long)))
        self.assertTrue(torch.equal(self.environments[first][16:], torch.ones(16, dtype=torch.long)))
        self.assertTrue(torch.equal(self.environments[second][16:], torch.zeros(16, dtype=torch.long)))
        self.assertTrue(bool((coefficients >= 0).all()) and bool((coefficients <= 1).all()))
        self.assertEqual(len(torch.unique(coefficients)), 2)
        # every right-domain sample is used exactly once in each direction
        self.assertEqual(sorted(second[:16].tolist()), list(range(16, 32)))
        self.assertEqual(sorted(first[16:].tolist()), list(range(16, 32)))

    def test_same_coefficients_for_amplitude_and_phase(self):
        first, second, coefficients = dg.mixup_plan(self.environments, 1, 3)
        amplitude = torch.randn(32, 5, 4)
        phase = torch.randn(32, 5, 4)
        mixed_a = dg.mix_inputs(amplitude, first, second, coefficients)
        mixed_p = dg.mix_inputs(phase, first, second, coefficients)
        w = coefficients.reshape(-1, 1, 1)
        self.assertTrue(torch.allclose(mixed_a, w * amplitude[first] + (1 - w) * amplitude[second]))
        self.assertTrue(torch.allclose(mixed_p, w * phase[first] + (1 - w) * phase[second]))

    def test_mixed_cross_entropy_reduces_to_cross_entropy(self):
        logits = torch.randn(8, 6)
        labels = torch.randint(0, 6, (8,))
        other = (labels + 1) % 6
        ones = torch.ones(8)
        self.assertTrue(torch.allclose(dg.mixed_cross_entropy(logits, labels, other, ones), F.cross_entropy(logits, labels)))
        self.assertTrue(torch.allclose(dg.mixed_cross_entropy(logits, labels, other, 0 * ones), F.cross_entropy(logits, other)))

    def test_requires_balanced_sources(self):
        with self.assertRaises(ValueError):
            dg.mixup_plan(torch.tensor([0, 0, 0, 1]), 1, 1)


class SpecificationTests(unittest.TestCase):
    def test_arm_specifications(self):
        self.assertEqual(dg.arm_specification("C10")["method"], "coral")
        self.assertEqual(dg.arm_specification("C10")["coral_gamma"], 1.0)
        self.assertEqual(dg.arm_specification("X10")["method"], "mixup")
        self.assertEqual(dg.arm_specification("X10")["mixup_alpha"], 0.2)
        dg.validate_specification(dg.arm_specification("X10"), "X10")
        with self.assertRaises(ValueError):
            dg.validate_specification(dg.arm_specification("X10"), "C10")


class EpochLoopTests(unittest.TestCase):
    """The replayed loop equals a hand-written CE loop on a toy model."""

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(6, 4)
            self.classifier = torch.nn.Linear(4, 6)
            self.projection_head = torch.nn.Linear(4, 3)

        def forward(self, amplitude, phase):
            fused = torch.tanh(self.encoder(amplitude.mean(1) + phase.mean(1)))
            return {"fused_feature": fused, "logits": self.classifier(fused),
                    "projection": F.normalize(self.projection_head(fused), dim=-1)}

    class FakeTrainer:
        @staticmethod
        def metric_dict(labels, predictions):
            return {"accuracy": float((labels == predictions).mean())}

        @staticmethod
        def zero_supcon_diagnostics():
            return {"supcon_num_anchors": 0.0}

    def loader(self):
        g = torch.Generator().manual_seed(3)
        amplitude = torch.randn(8, 5, 6, generator=g)
        phase = torch.randn(8, 5, 6, generator=g)
        labels = torch.randint(0, 6, (8,), generator=g)
        domains = torch.tensor([0, 0, 1, 1] * 2)
        ids = torch.arange(8)
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(amplitude, phase, labels, domains, ids),
                                           batch_size=4, shuffle=False)

    def run_loop(self, method):
        torch.manual_seed(0)
        model = self.ToyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        spec = {"method": method, "coral_gamma": 1.0, "feature": "fused_feature", "mixup_alpha": 0.2, "rng_namespace": 1}
        steps = {"steps": 0}
        original = optimizer.step

        def counted(*a, **k):
            value = original(*a, **k)
            steps["steps"] += 1
            return value
        optimizer.step = counted
        result = dg.baseline_epoch(self.FakeTrainer, model, self.loader(), optimizer, None, "step", torch.device("cpu"),
                                   {"lambda_supcon": 0.0, "grad_clip_norm": 0.0}, spec=spec, seed=42, step_counter=steps)
        return model, result, steps["steps"]

    def test_ce_mode_matches_manual_loop(self):
        model, result, steps = self.run_loop("ce")
        torch.manual_seed(0)
        manual = self.ToyModel()
        optimizer = torch.optim.Adam(manual.parameters(), lr=1e-2)
        for amplitude, phase, labels, _d, _i in self.loader():
            loss = F.cross_entropy(manual(amplitude, phase)["logits"], labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        for (name, a), (_, b) in zip(model.state_dict().items(), manual.state_dict().items()):
            self.assertTrue(torch.equal(a, b), name)
        self.assertEqual(steps, 2)
        self.assertNotIn("loss_dg_regularizer", result)
        self.assertEqual(result["loss_supcon"], 0.0)

    def test_regularized_modes_run_and_report(self):
        for method in ("coral", "mixup"):
            model, result, steps = self.run_loop(method)
            self.assertEqual(steps, 2)
            self.assertTrue(np.isfinite(result["loss_total"]))
            self.assertIn("loss_dg_regularizer", result)
            self.assertEqual(result["dg_method_is_coral"], 1.0 if method == "coral" else 0.0)
            if method == "coral":
                self.assertGreater(result["loss_dg_regularizer"], 0.0)
                self.assertEqual(result["dg_regularizer_weight"], 1.0)
            else:
                self.assertGreater(result["mixup_mean_coefficient"], 0.0)

    def test_rejects_contrastive_coefficient(self):
        with self.assertRaises(ValueError):
            dg.baseline_epoch(self.FakeTrainer, self.ToyModel(), self.loader(), torch.optim.SGD(self.ToyModel().parameters(), lr=0.1),
                              None, "step", torch.device("cpu"), {"lambda_supcon": 0.05},
                              spec={"method": "ce"}, seed=1, step_counter={"steps": 0})


if __name__ == "__main__":
    unittest.main()
