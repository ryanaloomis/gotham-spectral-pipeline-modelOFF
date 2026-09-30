"""Reference construction in CAL-combined raw counts, before calibration.

Builders consume and return arrays in native channel order. They neither change
frequency frames nor estimate Tsys. Model uncertainty is separate from the
conditional, narrow-scale radiometer noise returned by calibration.
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.interpolate import LSQUnivariateSpline


class ReferenceFitError(ValueError):
    """The reference cannot be estimated with adequate support."""


@dataclass
class ReferenceResult:
    counts: np.ndarray
    shape: np.ndarray
    normalization: float
    fit_mask: np.ndarray
    valid_mask: np.ndarray
    diagnostics: dict = field(default_factory=dict)


class ReferenceBuilder(Protocol):
    def build(self, *, off_counts, on_counts, fit_mask=None) -> ReferenceResult:
        ...


@dataclass(frozen=True)
class SmoothOffReference:
    """Broad robust cubic spline with fixed knots in native channel coordinates.

    Shape is fitted independently for each paired OFF; no cross-integration
    cache is used. The robust median of shape on supported channels is one.
    """

    knot_spacing: int = 512
    fit_bin: int = 32
    edge_channels: int = 1024
    clip_sigma: float = 5.0
    iterations: int = 4
    normalization: str = "off"

    def __post_init__(self):
        if self.fit_bin < 1 or self.knot_spacing < max(32, 4 * self.fit_bin):
            raise ValueError("knot_spacing must be >= 32 and >= 4 * fit_bin")
        if self.edge_channels < 0 or self.clip_sigma <= 0 or self.iterations < 1:
            raise ValueError("Invalid edge, clipping, or iteration setting")
        if self.normalization not in ("off", "on", "none"):
            raise ValueError("normalization must be off, on, or none")

    def build(self, *, off_counts, on_counts, fit_mask=None):
        off = np.asarray(off_counts, dtype=float)
        on = np.asarray(on_counts, dtype=float)
        if off.ndim != 1 or on.shape != off.shape:
            raise ReferenceFitError("ON and OFF must be matching one-dimensional arrays")
        valid = np.isfinite(off) & (off > 0)
        if fit_mask is not None:
            if np.shape(fit_mask) != off.shape:
                raise ReferenceFitError("fit_mask shape does not match OFF")
            valid &= np.asarray(fit_mask, dtype=bool)
        edge = self.edge_channels
        if edge:
            valid[:edge] = False
            valid[-edge:] = False
        x = np.arange(off.size, dtype=float)
        # Median compression prevents isolated raw-channel spikes from dominating.
        bx, by = [], []
        for start in range(edge, off.size - edge, self.fit_bin):
            stop = min(start + self.fit_bin, off.size - edge)
            good = valid[start:stop]
            if good.sum() < max(2, (stop - start + 1) // 2):
                continue
            bx.append(np.median(x[start:stop][good]))
            by.append(np.median(off[start:stop][good]))
        bx, by = np.asarray(bx), np.asarray(by)
        if bx.size < 8:
            raise ReferenceFitError("Too few supported bins")
        knots = np.arange(bx[0] + self.knot_spacing, bx[-1], self.knot_spacing)
        keep = np.ones(bx.size, dtype=bool)

        def fit():
            xx, yy = bx[keep], by[keep]
            boundaries = np.r_[bx[0], knots, bx[-1]]
            support, _ = np.histogram(xx, bins=boundaries)
            if xx.size <= knots.size + 4 or np.any(support < 2):
                raise ReferenceFitError("Insufficient real data in a spline knot interval")
            try:
                return LSQUnivariateSpline(xx, yy, knots, bbox=[bx[0], bx[-1]], k=3)
            except ValueError as exc:
                raise ReferenceFitError(str(exc)) from exc

        for _ in range(self.iterations):
            spline = fit()
            residual = by - spline(bx)
            center = np.median(residual[keep])
            scale = 1.4826 * np.median(np.abs(residual[keep] - center))
            floor = 100 * np.finfo(float).eps * np.max(np.abs(by[keep]))
            if not np.isfinite(scale) or scale <= floor:
                break
            new_keep = keep & (np.abs(residual - center) <= self.clip_sigma * scale)
            if np.array_equal(keep, new_keep):
                break
            keep = new_keep
        spline = fit()
        # Clamp unsupported edges to boundary values, and exclude them using
        # valid_mask. Finite padding avoids poisoning the existing RFI detector's
        # chunk means with artificial NaNs. These channels never enter science fits.
        supported = (x >= bx[keep][0]) & (x <= bx[keep][-1])
        smooth = spline(np.clip(x, bx[keep][0], bx[keep][-1]))
        if np.any(~np.isfinite(smooth[supported]) | (smooth[supported] <= 0)):
            raise ReferenceFitError("Nonpositive or nonfinite spline reference")
        # Exclude clipped compressed bins from scalar normalization as well.
        fit_valid = valid & supported
        for center in bx[~keep]:
            fit_valid[np.abs(x - center) <= self.fit_bin / 2] = False
        shape_scale = float(np.median(smooth[fit_valid]))
        shape = smooth / shape_scale
        normalization_mask = fit_valid.copy()
        if self.normalization == "none":
            amplitude = shape_scale
        else:
            source = off if self.normalization == "off" else on
            normalization_mask &= np.isfinite(source) & (source > 0)
            ratio = source / shape
            for _ in range(self.iterations):
                values = ratio[normalization_mask]
                if values.size < max(8, int(0.1 * supported.sum())):
                    raise ReferenceFitError("Too few channels for scalar normalization")
                amplitude = float(np.median(values))
                scale = 1.4826 * np.median(np.abs(values - amplitude))
                floor = 100 * np.finfo(float).eps * abs(amplitude)
                if scale <= floor:
                    break
                updated = normalization_mask & (np.abs(ratio - amplitude) <= self.clip_sigma * scale)
                if np.array_equal(updated, normalization_mask):
                    break
                normalization_mask = updated
            if normalization_mask.sum() < max(8, int(0.1 * supported.sum())):
                raise ReferenceFitError("Too few channels for scalar normalization")
            amplitude = float(np.median(ratio[normalization_mask]))
        if not np.isfinite(amplitude) or amplitude <= 0:
            raise ReferenceFitError("Invalid scalar normalization")
        return ReferenceResult(
            counts=amplitude * shape,
            shape=shape,
            normalization=amplitude,
            fit_mask=fit_valid,
            valid_mask=supported,
            diagnostics=dict(
                kind="robust_cubic_spline", normalization_source=self.normalization,
                knot_spacing_channels=self.knot_spacing, fit_bin=self.fit_bin,
                edge_channels=edge, clip_sigma=self.clip_sigma, iterations=self.iterations,
                knots=spline.get_knots(), coefficients=spline.get_coeffs(), degree=3,
                bin_channel=bx, bin_counts=by, bin_keep=keep,
                shape_scale=shape_scale, normalization_mask=normalization_mask,
                uncertainty="conditional_on_fixed_model_and_normalization",
            ),
        )
