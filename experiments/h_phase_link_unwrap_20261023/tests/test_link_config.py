"""Frozen N_L11 configs may differ from paired N_H11 only in phase bindings."""
from __future__ import annotations
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BUNDLE, read, verify_config


class LinkConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = read(BUNDLE / 'configs/HS_E2_N_L11_52.json')

    def rejects(self, mutate):
        config = copy.deepcopy(self.config)
        mutate(config)
        with self.assertRaises(RuntimeError):
            verify_config(config)

    def test_frozen_config_passes(self):
        verify_config(copy.deepcopy(self.config))

    def test_attempt_output_dir_passes(self):
        config = copy.deepcopy(self.config)
        config['output_dir'] = str(BUNDLE.parent / 'runs/E2/seed_52/HS_N_L11/attempt_001/payload/run')
        verify_config(config)

    def test_paired_phase_array_rejected(self):
        paired = read(self.config['h_controls']['paired_control_config'])
        def mutate(c):
            c['source_file_records']['E1']['phase'] = paired['source_file_records']['E1']['phase']
            c['data']['env_files']['E1']['phase'] = paired['data']['env_files']['E1']['phase']
        self.rejects(mutate)

    def test_loader_unwrap_rejected(self):
        self.rejects(lambda c: c['data'].update(unwrap_phase=True))

    def test_training_change_rejected(self):
        self.rejects(lambda c: c['training'].update(lr=.002))

    def test_loss_change_rejected(self):
        self.rejects(lambda c: c['losses'].update(lambda_supcon=.1))

    def test_initial_state_change_rejected(self):
        self.rejects(lambda c: c['h_controls'].update(initial_state_sha256='0' * 64))

    def test_amplitude_change_rejected(self):
        def mutate(c):
            c['source_file_records']['E1']['amp']['sha256'] = '0' * 64
        self.rejects(mutate)

    def test_pseudo_tx_layout_change_rejected(self):
        self.rejects(lambda c: c['model']['antenna_layout'].update(num_tx=1, num_rx=3))


if __name__ == '__main__':
    unittest.main()
