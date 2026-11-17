"""CPU checks against byte-verified frozen H modules, with no training/data IO.

Run from the repository:
C:/ProgramData/anaconda3/python.exe -m unittest discover -s experiments/h_main_controls_20261008/tests -v
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT
import control_components as controls

SNAPSHOT = ROOT / "paper/mbl_ieee_revised_20261008/supplement/code/H"
BOUND_HASHES = {
    "dg_models.py": "7f99caf93a5103e00866d24263faf0444429ae8a128d954838bef82a6cd02d07",
    "dg_losses.py": "fd689168b48601e82f754590e7b881faec33f9de8544f949a6b2f63fcc902dbc",
    "experiments/tpp_isolated/tpp_models.py": "fc6aeb3858763c282b892f8c8e91ac31425d93df4343d83bb33cd1c6749ebccf",
}


def load_bound_module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def load_frozen_h():
    for relative, expected in BOUND_HASHES.items():
        actual = hashlib.sha256((SNAPSHOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"Frozen H source changed: {relative}: {actual}")
    old = sys.modules.get("dg_models")
    try:
        load_bound_module("dg_models", SNAPSHOT / "dg_models.py")
        tpp = load_bound_module(
            "_bound_h_controls_tpp", SNAPSHOT / "experiments/tpp_isolated/tpp_models.py"
        )
    finally:
        if old is None:
            sys.modules.pop("dg_models", None)
        else:
            sys.modules["dg_models"] = old
    losses = load_bound_module("_bound_h_controls_losses", SNAPSHOT / "dg_losses.py")
    config = json.loads((SNAPSHOT / "configs/published_h11_l1248_config.json").read_text(encoding="utf-8"))
    return tpp.IsolatedPaCsiDGLiteTPP, losses, config["model"]


BASE_CLASS, FROZEN_LOSSES, MODEL_CONFIG = load_frozen_h()
torch.set_num_threads(1)


def construct(mode, seed=42):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = controls.make_model_class(BASE_CLASS, mode)(90, 6, copy.deepcopy(MODEL_CONFIG))
        next_rng = torch.random.get_rng_state().clone()
    return model, next_rng


class CapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tpp, cls.tpp_rng = construct("tpp")
        cls.global_mlp, cls.global_rng = construct("global_mlp")

    def test_actual_complete_h_parameter_count_is_equal(self):
        expected_projections = [983296, 15392, 983296, 15392]
        for model in (self.tpp, self.global_mlp):
            self.assertEqual(sum(p.numel() for p in model.parameters()), 4_786_294)
            self.assertEqual(
                [sum(p.numel() for p in model.get_submodule(path).projection.parameters())
                 for path in controls.POOL_PATHS],
                expected_projections,
            )
        for path, width, hidden in zip(controls.POOL_PATHS, (256, 32, 256, 32), (1920, 240, 1920, 240)):
            pool = self.global_mlp.get_submodule(path)
            self.assertEqual(pool.projection[0].in_features, width)
            self.assertEqual(pool.projection[0].out_features, hidden)
            self.assertIsNone(pool.projection[0].bias)
            self.assertEqual(pool.projection[2].out_features, width)
            self.assertIsNotNone(pool.projection[2].bias)
            self.assertEqual(pool.dropout.p, 0.1)
            self.assertEqual(pool.pyramid_scale, 0.25)

    def test_every_shared_tensor_matches_and_rng_is_preserved(self):
        audit = controls.initialization_audit(self.global_mlp, self.tpp)
        self.assertTrue(audit["shared_state_tensors_exactly_equal"], audit)
        self.assertGreater(audit["shared_state_tensor_count"], 100)
        self.assertEqual(audit["model_common_state_sha256"], audit["reference_common_state_sha256"])
        self.assertNotEqual(audit["model_full_state_sha256"], audit["reference_full_state_sha256"])
        self.assertTrue(torch.equal(self.tpp_rng, self.global_rng))
        for name in ("classifier.weight", "classifier.bias",
                     "projection_head.network.0.weight", "fusion.gate.weight"):
            # Inspect actual module keys below if a bound implementation names them differently.
            if name in self.tpp.state_dict():
                self.assertTrue(torch.equal(self.tpp.state_dict()[name], self.global_mlp.state_dict()[name]))

    def test_tpp_factory_returns_exact_frozen_class_and_full_state(self):
        self.assertIs(controls.make_model_class(BASE_CLASS, "tpp"), BASE_CLASS)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            untouched = BASE_CLASS(90, 6, copy.deepcopy(MODEL_CONFIG))
        audit = controls.initialization_audit(self.tpp, untouched)
        self.assertEqual(audit["model_full_state_sha256"], audit["reference_full_state_sha256"])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(self.tpp.state_dict().values(), untouched.state_dict().values())))

    def test_replacement_retains_all_four_convolution_objects(self):
        model, _ = construct("tpp", seed=52)
        original = {path: model.get_submodule(path) for path in controls.POOL_PATHS}
        controls.install_capacity_matched_global(model)
        for path, old in original.items():
            new = model.get_submodule(path)
            self.assertIs(new.convs, old.convs)
            self.assertIs(new.activation, old.activation)
            self.assertIs(new.dropout, old.dropout)
        bad_model, _ = construct("tpp")
        before = {path: bad_model.get_submodule(path) for path in controls.POOL_PATHS}
        bad_model.phase_encoder.channel_pool.levels = [1, 2, 4]
        with self.assertRaises(ValueError):
            controls.install_capacity_matched_global(bad_model)
        self.assertTrue(all(bad_model.get_submodule(path) is pool for path, pool in before.items()))

    def test_zero_correction_exact_global_max_and_pool_backward(self):
        for path in controls.POOL_PATHS:
            pool = copy.deepcopy(self.global_mlp.get_submodule(path)).eval()
            for parameter in pool.projection.parameters():
                parameter.data.zero_()
            input_width = pool.convs[0].in_channels
            x = torch.randn(2, 45, input_width, requires_grad=True)
            expected = torch.cat([
                F.adaptive_max_pool1d(torch.relu(conv(x.transpose(1, 2))), 1).flatten(1)
                for conv in pool.convs
            ], dim=1)
            actual = pool(x)
            self.assertTrue(torch.equal(actual, expected))
            actual.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(float(x.grad.abs().sum()), 0.0)

    def test_full_model_and_pseudo_tx_view_interfaces_on_cpu(self):
        model = self.global_mlp.eval()
        amplitude = torch.randn(1, 82, 90)
        phase = torch.randn(1, 82, 90)
        with torch.no_grad():
            output = model(amplitude, phase)
            views = model.encode_antenna_views(amplitude, phase)
        self.assertEqual(tuple(output["logits"].shape), (1, 6))
        self.assertEqual(tuple(output["projection"].shape), (1, 128))
        self.assertEqual(tuple(views["projection_views"].shape), (1, 3, 128))
        self.assertTrue(torch.allclose(output["projection"].norm(dim=-1), torch.ones(1), atol=1e-6))
        self.assertTrue(torch.isfinite(output["logits"]).all())
        for view in range(3):
            expected = torch.zeros_like(amplitude)
            expected[:, :, 30 * view:30 * (view + 1)] = amplitude[:, :, 30 * view:30 * (view + 1)]
            self.assertTrue(torch.equal(views["amp_views"][:, view], expected))


# Four draws, four views each. Draws 0 and 2 are the same original record;
# draw 1 is a different record of the same class; draw 3 is a different class.
# All three same-class records cross/within domains as frozen H permits.
LABELS = torch.tensor([0, 0, 0, 1])
DOMAINS = torch.tensor([0, 1, 0, 1])
IDS = torch.tensor([7, 8, 7, 9])
HYBRID_POSITIVE_SETS = [
    {1, 2, 3, 4, 8, 9, 10, 11}, {0, 2, 3, 8, 9, 10, 11},
    {0, 1, 3, 8, 9, 10, 11}, {0, 1, 2, 8, 9, 10, 11},
    {0, 5, 6, 7, 8}, {4, 6, 7}, {4, 5, 7}, {4, 5, 6},
    {0, 1, 2, 3, 4, 9, 10, 11}, {0, 1, 2, 3, 8, 10, 11},
    {0, 1, 2, 3, 8, 9, 11}, {0, 1, 2, 3, 8, 9, 10},
    {13, 14, 15}, {12, 14, 15}, {12, 13, 15}, {12, 13, 14},
]


def row_sets(mask):
    return [{index for index, included in enumerate(row) if included} for row in mask.tolist()]


def manual_loss(feature_rows, positive_sets, denominator_sets, temperature=0.2):
    # Scalar arithmetic is independent of the tensor-mask implementation.
    units = [[x / math.sqrt(sum(v * v for v in row)) for x in row] for row in feature_rows]
    anchor_losses = []
    for i, positives in enumerate(positive_sets):
        if not positives:
            continue
        scaled = [sum(a * b for a, b in zip(units[i], other)) / temperature for other in units]
        shift = max(scaled)
        normalizer = math.log(sum(math.exp(scaled[j] - shift) for j in denominator_sets[i]) + 1e-12)
        anchor_losses.append(-sum(scaled[j] - shift - normalizer for j in positives) / len(positives))
    return sum(anchor_losses) / len(anchor_losses) if anchor_losses else 0.0


class PositiveRuleTests(unittest.TestCase):
    def setUp(self):
        angles = torch.arange(16, dtype=torch.float64) * 0.231
        self.features = torch.stack((angles.cos(), angles.sin(), angles * 0.03 + 0.1), dim=-1).reshape(4, 4, 3)

    def masks(self, mode):
        return controls.control_pair_masks(self.features, LABELS, DOMAINS, IDS, True, mode)

    def test_hybrid_hand_mask_including_repeated_identity_same_view(self):
        flat, positives, denominator = self.masks("global_class_instance_views")
        self.assertEqual(row_sets(positives), HYBRID_POSITIVE_SETS)
        self.assertEqual(row_sets(denominator), [set(range(16)) - {i} for i in range(16)])
        self.assertTrue(positives[1, 9])  # Same view, repeated original record.
        self.assertFalse(positives[1, 5])  # Same view/class, different original record.
        self.assertTrue(positives[0, 4])  # Full/full same class across source domains.
        self.assertTrue(positives[0, 8])  # Repeated draw full/full.
        self.assertTrue(torch.equal(flat, self.features.reshape(16, 3)))

    def test_all_view_hand_mask(self):
        _, positives, denominator = self.masks("all_views")
        expected = [
            (set(range(12)) if i < 12 else set(range(12, 16))) - {i}
            for i in range(16)
        ]
        self.assertEqual(row_sets(positives), expected)
        self.assertEqual(row_sets(denominator), [set(range(16)) - {i} for i in range(16)])

    def test_removed_denominator_hand_mask_keeps_every_positive(self):
        _, positives, denominator = self.masks("hybrid_remove_other_sameclass_denominator")
        self.assertEqual(row_sets(positives), HYBRID_POSITIVE_SETS)
        expected = [
            positives_i | (set(range(12, 16)) if i < 12 else set(range(12)))
            for i, positives_i in enumerate(HYBRID_POSITIVE_SETS)
        ]
        self.assertEqual(row_sets(denominator), expected)
        self.assertEqual(row_sets(denominator)[0], HYBRID_POSITIVE_SETS[0] | {12, 13, 14, 15})
        self.assertFalse((positives & ~denominator).any())

    def test_all_three_losses_against_independent_scalar_calculation(self):
        for mode in ("global_class_instance_views", "all_views", "hybrid_remove_other_sameclass_denominator"):
            with self.subTest(mode=mode):
                _, positives, denominator = self.masks(mode)
                expected = manual_loss(self.features.reshape(16, 3).tolist(),
                                       row_sets(positives), row_sets(denominator))
                actual = controls.ControlDomainAwareSupConLoss(positive_mode=mode)(
                    self.features, LABELS, DOMAINS, IDS
                )
                self.assertAlmostEqual(actual.item(), expected, places=13)
        full = controls.ControlDomainAwareSupConLoss(positive_mode="hybrid")(self.features, LABELS, DOMAINS, IDS)
        removed = controls.ControlDomainAwareSupConLoss(
            positive_mode="hybrid_remove_other_sameclass_denominator"
        )(self.features, LABELS, DOMAINS, IDS)
        self.assertLess(removed.item(), full.item())

    def test_frozen_hybrid_and_all_view_loss_and_gradient_bitwise_equal(self):
        for dtype in (torch.float32, torch.float64):
            for mode in ("global_class_instance_views", "all_views"):
                for include_same_domain in (True, False):
                    for ids in (IDS, None):
                        with self.subTest(dtype=dtype, mode=mode, domain=include_same_domain, implicit=ids is None):
                            original_input = self.features.to(dtype).clone().requires_grad_(True)
                            control_input = original_input.detach().clone().requires_grad_(True)
                            kwargs = dict(temperature=0.2, include_same_domain_same_class=include_same_domain, positive_mode=mode)
                            original = FROZEN_LOSSES.DomainAwareSupConLoss(**kwargs)(
                                original_input, LABELS, DOMAINS, ids
                            )
                            actual = controls.ControlDomainAwareSupConLoss(**kwargs)(
                                control_input, LABELS, DOMAINS, ids
                            )
                            original.backward()
                            actual.backward()
                            self.assertTrue(torch.equal(actual, original), (actual, original))
                            self.assertTrue(torch.equal(control_input.grad, original_input.grad))

    def test_no_positive_single_view_and_invalid_positive_denominator(self):
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        loss = controls.ControlDomainAwareSupConLoss()(
            features, torch.tensor([0, 1]), sample_ids=torch.tensor([11, 12])
        )
        self.assertEqual(loss.item(), 0.0)
        singleton = controls.ControlDomainAwareSupConLoss()(features[:1], torch.tensor([0]))
        self.assertEqual(singleton.item(), 0.0)
        positive = torch.tensor([[False, True], [True, False]])
        with self.assertRaises(ValueError):
            controls.masked_contrastive_loss(features, positive, 0.2, torch.zeros_like(positive))
        with self.assertRaises(ValueError):
            controls.ControlDomainAwareSupConLoss(positive_mode="invented")

    def test_removed_loss_finite_gradients_and_diagnostics_pair_counts(self):
        features = self.features.clone().requires_grad_(True)
        loss = controls.ControlDomainAwareSupConLoss(
            positive_mode="hybrid_remove_other_sameclass_denominator"
        )(features, LABELS, DOMAINS, IDS)
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        diagnostics = controls.supcon_positive_pair_diagnostics(
            self.features, LABELS, DOMAINS, IDS,
            positive_mode="hybrid_remove_other_sameclass_denominator",
        )
        self.assertEqual(diagnostics["supcon_num_valid_positive_pairs"], sum(map(len, HYBRID_POSITIVE_SETS)))
        self.assertEqual(diagnostics["supcon_num_denominator_pairs"],
                         sum(len(p) + (4 if i < 12 else 12) for i, p in enumerate(HYBRID_POSITIVE_SETS)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
