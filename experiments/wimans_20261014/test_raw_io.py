import unittest

import numpy as np

from experiments.wimans_20261014.raw_io import reconstruct, relative_microseconds


class RawInputTests(unittest.TestCase):
    def test_unsigned_wrap_and_signed_bits(self):
        elapsed, stats = relative_microseconds([-2, -1, 0, 0, 4])
        np.testing.assert_array_equal(elapsed, [0, 1, 2, 2, 6])
        self.assertEqual(stats["wrap_count"], 1)
        self.assertEqual(stats["duplicate_timestamp_count"], 1)

    def test_backward_and_zero_duration_rejected(self):
        for times in ([10, 9], [10, 10], [0, 2**31]):
            with self.assertRaises(ValueError):
                relative_microseconds(times)

    def test_nonintegral_nonfinite_and_out_of_range_rejected(self):
        for times in ([0.25, 10.75], [0, np.inf], [0, np.nan], [0, 2**32], [-2**31 - 1, 0]):
            with self.assertRaises(ValueError):
                relative_microseconds(times)

    def test_wrapped_angle_arithmetic_and_shared_grid(self):
        csi = np.exp(1j * np.repeat(np.array([3.1, -3.1])[:, None], 90, axis=1))
        amp, phase = reconstruct(csi, [0, 100])
        np.testing.assert_allclose(amp, 1, atol=1e-6)
        np.testing.assert_allclose(phase[:, 0], np.linspace(3.1, -3.1, 850), atol=1e-6)
        dup = np.repeat(csi, 2, axis=0)
        _, opposing = reconstruct(dup, [0, 0, 100, 100])
        np.testing.assert_array_equal(opposing, phase)
        _, zeros = reconstruct(np.tile(csi, (2, 1)), [0, 0, 100, 100])
        np.testing.assert_allclose(zeros, 0, atol=1e-6)

    def test_feature_axis_boundary_unwrap_and_zero_amplitude(self):
        angles = np.tile(np.repeat([3.1, -3.1, 3.1], 30), (2, 1))
        _, phase = reconstruct(np.exp(1j * angles), [0, 100])
        self.assertAlmostEqual(float(phase[0, 30]), 2 * np.pi - 3.1, places=6)
        amp, phase = reconstruct(np.zeros((2, 90), dtype=np.complex128), [0, 100])
        self.assertFalse(amp.any())
        self.assertFalse(phase.any())


if __name__ == "__main__":
    unittest.main()
