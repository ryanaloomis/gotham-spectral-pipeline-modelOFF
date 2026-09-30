"""Small numerical checks requiring only NumPy and SciPy.

Load the isolated builder directly so importing the full pipeline (Astropy,
GitPython, etc.) is unnecessary for these tests.
"""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    "off_reference_test_module",
    Path(__file__).resolve().parents[1] / "gotham_spectral_pipeline/reference.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
SmoothOffReference = module.SmoothOffReference
ReferenceFitError = module.ReferenceFitError


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(-1, 1, 8192)
        self.band = 100 + 8 * self.x + 3 * self.x**2
        self.options = dict(edge_channels=64, fit_bin=16, knot_spacing=512)

    def test_off_fit_rejects_spikes_and_has_explicit_edges(self):
        rng = np.random.default_rng(5)
        off = self.band + rng.normal(0, 0.2, self.band.size)
        off[3000:3003] += 50
        result = SmoothOffReference(**self.options).build(off_counts=off, on_counts=off)
        mask = result.valid_mask
        self.assertLess(np.std((result.counts - self.band)[mask]), 0.04)
        self.assertFalse(mask[:64].any())
        self.assertFalse(mask[-64:].any())
        self.assertTrue(np.isfinite(result.counts).all())

    def test_on_normalization_preserves_narrow_injection(self):
        on = 1.04 * self.band
        on[4000:4003] += 2
        result = SmoothOffReference(normalization="on", **self.options).build(
            off_counts=self.band, on_counts=on,
        )
        recovered = (on - result.counts) / result.counts
        np.testing.assert_allclose(recovered[4000:4003],
                                   2 / (1.04 * self.band[4000:4003]), rtol=0.005)

    def test_independent_off_noise_is_removed_on_narrow_scales(self):
        rng = np.random.default_rng(17)
        off = self.band + rng.normal(0, 0.2, self.band.size)
        on = self.band + rng.normal(0, 0.2, self.band.size)
        result = SmoothOffReference(**self.options).build(off_counts=off, on_counts=on)
        measured = (on - off) / off
        modeled = (on - result.counts) / result.counts
        adjacent = result.valid_mask[:-1] & result.valid_mask[1:]
        ratio = (np.std(np.diff(modeled)[adjacent]) /
                 np.std(np.diff(measured)[adjacent]))
        self.assertGreater(ratio, 0.65)
        self.assertLess(ratio, 0.77)

    def test_wide_gap_is_rejected(self):
        mask = np.ones(self.band.size, dtype=bool)
        mask[2500:4500] = False
        with self.assertRaises(ReferenceFitError):
            SmoothOffReference(**self.options).build(
                off_counts=self.band, on_counts=self.band, fit_mask=mask,
            )

    def test_invalid_configuration_and_empty_support(self):
        with self.assertRaises(ValueError):
            SmoothOffReference(knot_spacing=3)
        with self.assertRaises(ReferenceFitError):
            SmoothOffReference(**self.options).build(
                off_counts=np.full(8192, np.nan), on_counts=self.band,
            )


if __name__ == "__main__":
    unittest.main()
