"""Read exported raw counts without inventing missing SDFITS metadata."""

from pathlib import Path

import astropy.io.fits
import numpy as np
import pandas as pd

from .calibration import PositionSwitchedCalibration, CalOnOffPairedRows, SigRefPairedRows
from .sdfits import HDUList


class ArrayInput:
    def __init__(self, prefix):
        prefix = str(prefix)
        if prefix.endswith("_meta.csv"):
            prefix = prefix[:-9]
        elif prefix.endswith(".npy"):
            prefix = prefix[:-4]
        self.path = Path(prefix)
        self.data = np.load(prefix + ".npy", mmap_mode="r", allow_pickle=False)
        self.rows = pd.read_csv(prefix + "_meta.csv", keep_default_na=False)
        required = {"SCAN", "INT", "CAL", "PROCSCAN", "PROCEDURE", "SOURCE",
                    "SAMPLER", "RESTFREQ", "FREQRES", "EXPOSURE", "CENTFREQ"}
        missing = required - set(self.rows)
        if missing:
            raise ValueError(f"Missing metadata fields: {sorted(missing)}")
        if self.data.ndim != 2 or len(self.rows) != self.data.shape[0]:
            raise ValueError("Expected one metadata row per two-dimensional array row")
        if self.data.dtype.kind != "f":
            raise ValueError("Expected floating-point raw counts")
        if "NUMCHN" in self.rows and not (self.rows.NUMCHN == self.data.shape[1]).all():
            raise ValueError("NUMCHN disagrees with the array")
        self.rows["ARRAY_ROW"] = np.arange(len(self.rows))
        if "INDEX" not in self.rows:
            self.rows["INDEX"] = self.rows.ARRAY_ROW
        for field in ("CAL", "PROCSCAN"):
            self.rows[field] = self.rows[field].astype(str).str.strip().str.upper()
        if not self.rows.CAL.isin(["T", "F"]).all():
            raise ValueError("CAL must contain T/F")
        if not self.rows.PROCSCAN.isin(["ON", "OFF"]).all():
            raise ValueError("PROCSCAN must contain ON/OFF")
        if not self.rows.PROCEDURE.isin(["OffOn", "OnOff"]).all():
            raise ValueError("Only OffOn/OnOff procedures are supported")
        if not (np.isfinite(self.rows.FREQRES) & (self.rows.FREQRES > 0)).all():
            raise ValueError("FREQRES must be finite and positive")
        if not (np.isfinite(self.rows.EXPOSURE) & (self.rows.EXPOSURE >= 0)).all():
            raise ValueError("EXPOSURE must be finite and nonnegative")

    def get_hdulist_from_rows(self, rows):
        result = HDUList()
        for _, row in rows.iterrows():
            # TCAL=1 is a relative unit, not an inferred kelvin calibration.
            header = astropy.io.fits.Header()
            for key, value in dict(TCAL=1.0, OBSFREQ=row.CENTFREQ,
                                   FREQRES=row.FREQRES, EXPOSURE=row.EXPOSURE).items():
                header[key] = value
            result.append(astropy.io.fits.PrimaryHDU(
                data=self.data[int(row.ARRAY_ROW)], header=header,
            ))
        return result

    def pair_up_rows(self):
        rows = self.rows.copy()
        first = (rows.PROCEDURE == "OffOn") == (rows.PROCSCAN == "OFF")
        rows["PAIRED_OFFSCAN"] = np.where(first, rows.SCAN, rows.SCAN - 1)
        grouping = ["SOURCE", "PAIRED_OFFSCAN", "SAMPLER", "RESTFREQ"]
        pairs = []
        self.unpaired_rows = 0
        for group, scan_rows in rows.groupby(grouping, sort=False):
            for _, integration in scan_rows.groupby("INT", sort=True):
                parts = {(state, cal): integration[
                    (integration.PROCSCAN == state) & (integration.CAL == cal)
                ] for state in ("ON", "OFF") for cal in ("T", "F")}
                if any(len(part) > 1 for part in parts.values()):
                    raise ValueError(f"Ambiguous duplicate scan/integration/CAL rows in {group}")
                if any(len(part) != 1 for part in parts.values()):
                    self.unpaired_rows += len(integration)
                    continue
                pairs.append(SigRefPairedRows(
                    sig=CalOnOffPairedRows(calon=parts["ON", "T"], caloff=parts["ON", "F"]),
                    ref=CalOnOffPairedRows(calon=parts["OFF", "T"], caloff=parts["OFF", "F"]),
                    metadata=dict(group=dict(zip(
                        ["source", "offscan", "sampler", "restfreq"], group))),
                ))
        return pairs


class ArrayCalibration(PositionSwitchedCalibration):
    """Relative calibration on channel coordinates; no Doppler correction.

    The inherited frequency method signatures accept unit='Hz' for compatibility
    with Pipeline. For this adapter their numerical coordinate is ALWAYS channel
    index. Output manifests explicitly identify that convention.
    """

    @classmethod
    def get_observed_frequency(cls, hdulist, loc="center", unit="Hz"):
        n = hdulist[0].data.size
        if loc == "center":
            return np.arange(n, dtype=float)
        if loc == "edge":
            return np.arange(n + 1, dtype=float) - 0.5
        raise ValueError("loc must be center or edge")

    @classmethod
    def get_corrected_frequency(cls, hdulist, loc="center", unit="Hz", method="default"):
        return cls.get_observed_frequency(hdulist, loc=loc, unit=unit)
