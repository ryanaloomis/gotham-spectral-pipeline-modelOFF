import unittest

import numpy as np

from gotham_spectral_pipeline.pipeline import Pipeline
from gotham_spectral_pipeline.spectrum import Exposure, Spectrum


class PersistentRfiTests(unittest.TestCase):
    def make_products(self, *, peak=0.0, protect_peak=False):
        frequency = np.arange(256, dtype=float)
        intensity = np.zeros(frequency.size)
        intensity[128] = peak
        noise = np.ones(frequency.size)
        flags = np.full(
            frequency.size,
            Spectrum.FlagReason.VALID_DATA.value,
            dtype=np.int32,
        )
        if protect_peak:
            flags[128] |= Spectrum.FlagReason.SIGNAL.value
        spectrum = Spectrum(
            intensity=intensity, frequency=frequency, noise=noise, flag=flags
        )
        exposure = Exposure(exposure=np.ones(frequency.size), frequency=frequency)
        return spectrum, exposure

    def test_final_stack_detector_flags_feature_and_exposure(self):
        spectrum, exposure = self.make_products(peak=6.5)

        Pipeline.flag_persistent_model_rfi(spectrum, exposure)

        center = 128
        self.assertTrue(
            spectrum.flagged(Spectrum.FlagReason.FREQUENCY_DOMAIN_RFI)[center]
        )
        self.assertFalse(spectrum.flagged(Spectrum.FlagReason.VALID_DATA)[center])
        self.assertEqual(exposure.exposure[center], 0)

    def test_final_stack_detector_protects_existing_signal_mask(self):
        spectrum, exposure = self.make_products(peak=12.0, protect_peak=True)

        Pipeline.flag_persistent_model_rfi(spectrum, exposure)

        center = 128
        self.assertFalse(
            spectrum.flagged(Spectrum.FlagReason.FREQUENCY_DOMAIN_RFI)[center]
        )
        self.assertTrue(spectrum.flagged(Spectrum.FlagReason.VALID_DATA)[center])
        self.assertEqual(exposure.exposure[center], 1)


if __name__ == "__main__":
    unittest.main()
