"""Optional streaming diagnostics; large arrays never accumulate in Pipeline."""

import json
from pathlib import Path

import numpy as np

from .spectrum import Spectrum


class DiagnosticWriter:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, stage, paired_row, spectrum=None, metadata=None, reason=None):
        rows = {}
        indices = []
        for state in ("sig", "ref"):
            for cal in ("calon", "caloff"):
                row = paired_row[state][cal].iloc[0]
                rows[f"{state}_{cal}"] = row.to_dict()
                indices.append(str(int(row.get("ARRAY_ROW", row["INDEX"]))))
        stem = "_".join(indices) + "." + stage
        arrays = {}

        def encode(value, path):
            if isinstance(value, Spectrum):
                value = value._fields
            if isinstance(value, np.ndarray):
                arrays[path] = value
                return {"npz_key": path}
            if isinstance(value, dict):
                return {str(k): encode(v, f"{path}.{k}") for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [encode(v, f"{path}.{i}") for i, v in enumerate(value)]
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value

        document = encode(dict(
            stage=stage, rows=rows, reason=reason,
            spectrum=spectrum, metadata=metadata or {},
        ), "record")
        if arrays:
            np.savez_compressed(self.directory / (stem + ".npz"), **arrays)
        (self.directory / (stem + ".json")).write_text(
            json.dumps(document, indent=2, allow_nan=False) + "\n"
        )
