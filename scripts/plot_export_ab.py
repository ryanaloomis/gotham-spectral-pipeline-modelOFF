"""Overlay final aggregated A/B output spectra; no reduction is performed."""

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gotham-matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", type=Path, help="Path to group_000 (or another output group)")
    parser.add_argument("--zoom-center", type=float, default=114000)
    parser.add_argument("--zoom-width", type=float, default=4000)
    args = parser.parse_args()
    if args.zoom_width <= 0:
        parser.error("--zoom-width must be positive")
    spectra = []
    for mode in ("measured", "modeled"):
        with np.load(args.group / mode / "spectrum.npz") as data:
            spectra.append({key: data[key] for key in data.files})
    x = spectra[0]["frequency"]
    if not np.array_equal(x, spectra[1]["frequency"]):
        raise ValueError("Spectra have different channel coordinates")
    valid_masks = []
    for spectrum in spectra:
        # Existing Spectrum flags: VALID_DATA=32; invalid reasons occupy bits 0–3.
        valid_masks.append(
            ((spectrum["flag"] & 32) != 0) & ((spectrum["flag"] & 15) == 0)
            & np.isfinite(spectrum["intensity"]) & np.isfinite(spectrum["noise"])
            & (spectrum["noise"] > 0)
        )
    if not all(mask.any() for mask in valid_masks):
        raise ValueError("One final output has no valid channels")
    left = args.zoom_center - args.zoom_width / 2
    right = args.zoom_center + args.zoom_width / 2
    zoom = np.any(np.stack(valid_masks), axis=0) & (x >= left) & (x <= right)
    if not zoom.any():
        raise ValueError("Zoom has no valid channels; choose another --zoom-center")
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), constrained_layout=True)
    colors = ("#5475a5", "#dc702c")
    labels = (
        r"Measured reference: $T_{\rm sys}(\rm ON- OFF)/OFF$",
        r"Smooth reference: $T_{\rm sys}(\rm ON-M)/M$",
    )
    for index, (spectrum, label, color) in enumerate(zip(spectra, labels, colors)):
        valid = valid_masks[index]
        y = np.where(valid, spectrum["intensity"], np.nan)
        axes[0].plot(x, y, color=color, lw=0.4, alpha=0.8,
                     label=label, rasterized=True)
        window = (x >= left) & (x <= right)
        axes[1].plot(x[window], y[window], color=color, lw=0.8, alpha=0.85, label=label)
    axes[0].set_title("Full band • final pipeline spectra", loc="left", fontsize=11)
    axes[0].set_xlim(x[valid].min(), x[valid].max())
    axes[1].set_title("Central zoom • native channels, no smoothing", loc="left", fontsize=11)
    axes[1].set_xlim(left, right)
    for ax in axes:
        ax.set_xlabel("Native channel index (no Doppler correction)")
        ax.set_ylabel("Calibrated line intensity (Tcal units)")
        ax.axhline(0, color="black", lw=0.5, alpha=0.3)
        ax.grid(alpha=0.15)
        ax.legend(loc="upper right", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Final position-switched calibrated spectra", fontsize=15)
    directory = args.group / "plots"
    directory.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        path = directory / ("spectra_overlay." + extension)
        fig.savefig(path, dpi=180)
        print(path.resolve())
    plt.close(fig)


if __name__ == "__main__":
    main()
