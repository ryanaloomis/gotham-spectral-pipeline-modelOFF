"""Integration-contract checks; run in the full pipeline environment."""

from pathlib import Path
import tempfile
import unittest

import astropy.io.fits
import numpy as np
import pandas as pd

from gotham_spectral_pipeline.array_input import ArrayInput
from gotham_spectral_pipeline.calibration import (
    PositionSwitchedCalibration as Calibration, CalOnOffPairedHDUList, SigRefPairedHDUList,
)
from gotham_spectral_pipeline.reference import ReferenceResult
from gotham_spectral_pipeline.sdfits import HDUList


class MeasuredBuilder:
    def build(self, *, off_counts, on_counts, fit_mask=None):
        return ReferenceResult(
            counts=off_counts, shape=off_counts / np.median(off_counts),
            normalization=float(np.median(off_counts)),
            fit_mask=np.ones(off_counts.size, dtype=bool),
            valid_mask=np.ones(off_counts.size, dtype=bool),
        )


class CalibrationTests(unittest.TestCase):
    def pair(self):
        def hdu(data):
            header = astropy.io.fits.Header(dict(
                TCAL=2.0, EXPOSURE=10.0, FREQRES=1000.0, OBSFREQ=29e9,
                CTYPE1="FREQ", CUNIT1="Hz", CRPIX1=1.0, CRVAL1=29e9,
                CDELT1=-1000.0, VFRAME=0.0,
            ))
            return HDUList([astropy.io.fits.PrimaryHDU(data=data, header=header)])
        ref = np.linspace(10000, 11000, 128)
        signal = ref.copy()
        signal[50] += 20
        return SigRefPairedHDUList(
            ref=CalOnOffPairedHDUList(caloff=hdu(ref), calon=hdu(ref + 1000)),
            sig=CalOnOffPairedHDUList(caloff=hdu(signal), calon=hdu(signal + 1000)),
        )

    def test_default_is_legacy_arithmetic(self):
        pair = self.pair()
        tsys = Calibration.get_system_temperature(pair["ref"], Tcal=2.0, trim_fraction=0.1)
        kwargs = dict(Tcal=2.0, Tsys=tsys, ref_calonoffpair=pair["ref"])
        expected = (Calibration.get_total_power_spectrum(pair["sig"], **kwargs) -
                    Calibration.get_total_power_spectrum(pair["ref"], **kwargs))
        actual = Calibration.get_calibrated_spectrum(pair)
        for field in ("intensity", "noise", "frequency", "flag"):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))

    def test_reference_substitution_and_descending_grid(self):
        pair = self.pair()
        measured, metadata = Calibration.get_calibrated_spectrum(pair, return_metadata=True)
        modeled, model_metadata = Calibration.get_calibrated_spectrum(
            pair, reference_builder=MeasuredBuilder(), return_metadata=True,
        )
        np.testing.assert_allclose(modeled.intensity, measured.intensity, atol=1e-13)
        np.testing.assert_allclose(modeled.noise, measured.noise / np.sqrt(2))
        self.assertTrue(np.all(np.diff(modeled.frequency) > 0))
        self.assertEqual(np.argmax(modeled.intensity), 127 - 50)
        for key in ("sig_calon", "sig_caloff", "ref_calon", "ref_caloff"):
            np.testing.assert_array_equal(model_metadata[key].intensity, metadata[key].intensity)
            np.testing.assert_array_equal(model_metadata[key].noise, metadata[key].noise)

    def test_export_uses_array_position_and_explicit_integration(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "raw"
            rows = []
            for scan, state in ((12, "OFF"), (13, "ON")):
                for integration in (0, 1):
                    for cal in ("F", "T"):
                        if (scan, integration, cal) == (13, 0, "T"):
                            continue
                        rows.append(dict(
                            INDEX=100 + 4 * len(rows), SCAN=scan, INT=integration,
                            CAL=cal, PROCSCAN=state, PROCEDURE="OffOn", SOURCE="TMC-1",
                            SAMPLER="A1_0", RESTFREQ=29e9, FREQRES=1000.0,
                            EXPOSURE=10.0, CENTFREQ=29e9, NUMCHN=64,
                        ))
            data = np.arange(len(rows) * 64, dtype=float).reshape(len(rows), 64)
            np.save(str(prefix) + ".npy", data)
            pd.DataFrame(rows).to_csv(str(prefix) + "_meta.csv", index=False)
            source = ArrayInput(prefix)
            pairs = source.pair_up_rows()
            self.assertEqual(len(pairs), 1)
            self.assertEqual(source.unpaired_rows, 3)
            pair = pairs[0]
            row = pair["sig"]["caloff"].iloc[0]
            self.assertEqual(row.INT, 1)
            loaded = pair.get_paired_hdu(source)
            np.testing.assert_array_equal(loaded["sig"]["caloff"][0].data,
                                          data[int(row.ARRAY_ROW)])


if __name__ == "__main__":
    unittest.main()
