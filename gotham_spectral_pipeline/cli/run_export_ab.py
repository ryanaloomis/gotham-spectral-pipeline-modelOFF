"""Measured/model A/B reduction of raw NPY/CSV exports in relative units."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .. import __version__
from ..array_input import ArrayInput, ArrayCalibration
from ..diagnostics import DiagnosticWriter
from ..pipeline import Pipeline
from ..spectrum import Spectrum, SpectrumAggregator
from .reference_options import add_model_options, make_builder


def name():
    return "run_export_ab"


def help():
    return "Compare measured and smooth OFF using raw NPY/CSV, in Tcal units and channel coordinates."


def configure_parser(parser):
    parser.add_argument("--input-prefix", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--limit-pairs", type=int, help="Only reduce this many complete pairs, in input order")
    parser.add_argument("--flag-head-tail-channel-number", type=int, default=1024)
    parser.add_argument("--max-rfi-channel", type=int, default=256)
    add_model_options(parser)


def _save_output(output, directory):
    for field in ("spectrum", "exposure", "total_exposure"):
        value = getattr(output, field, None)
        if value is not None:
            value.to_npz(directory / (field + ".npz"))
    (directory / "status.json").write_text(json.dumps(dict(
        success=output.success, reason=output.reason,
        dropped=dict(output.integration_dropped_reason),
    ), indent=2) + "\n")


def _load_diagnostic(path):
    with np.load(path, allow_pickle=False) as data:
        return Spectrum(**{field: data["record.spectrum." + field]
                           for field in ("frequency", "intensity", "noise", "flag")})


def _compare_common(directory):
    """Compare post-baseline spectra on identical integrations/channels.

    Each branch has already used its own unmodified baseline fitting procedure.
    These products isolate selection differences, not differences in fitted baselines.
    """
    a_dir, b_dir = [directory / mode / "diagnostics" for mode in ("measured", "modeled")]
    filenames = sorted(set(p.name for p in a_dir.glob("*.baseline.npz")) &
                       set(p.name for p in b_dir.glob("*.baseline.npz")))
    aggregators = [SpectrumAggregator(SpectrumAggregator.LinearTransformer(1.0)) for _ in range(2)]
    records = []
    for filename in filenames:
        spectra = [_load_diagnostic(d / filename) for d in (a_dir, b_dir)]
        a, b = spectra
        if not np.array_equal(a.frequency, b.frequency):
            raise ValueError("Common comparison requires identical channel coordinates")
        mask = np.ones(a.intensity.size, dtype=bool)
        for spectrum in spectra:
            mask &= (spectrum.flagged(Spectrum.FlagReason.VALID_DATA) &
                     np.isfinite(spectrum.intensity) & np.isfinite(spectrum.noise) &
                     (spectrum.noise > 0))
        if mask.sum() < 2:
            continue
        adjacent = mask[:-1] & mask[1:]
        row = dict(pair=filename.removesuffix(".baseline.npz"), channels=int(mask.sum()))
        for mode, spectrum, aggregator in zip(("measured", "modeled"), spectra, aggregators):
            row[mode + "_std"] = float(np.std(spectrum.intensity[mask]))
            diff = np.diff(spectrum.intensity)[adjacent]
            row[mode + "_difference_rms"] = (
                float(1.4826 * np.median(np.abs(diff - np.median(diff))) / np.sqrt(2))
                if diff.size else None
            )
            flags = np.where(mask, Spectrum.FlagReason.VALID_DATA.value, Spectrum.FlagReason.NAN.value)
            # Zero invalid intensities avoids inf*0/NaN leakage inside aggregation.
            aggregator.merge(Spectrum(
                intensity=np.where(mask, spectrum.intensity, 0.0), frequency=spectrum.frequency,
                noise=np.where(mask, spectrum.noise, np.inf), flag=flags,
            ))
        records.append(row)
    common = directory / "common"
    common.mkdir()
    if records:
        for mode, aggregator in zip(("measured", "modeled"), aggregators):
            aggregator.get_spectrum().to_npz(common / (mode + ".npz"))
        with (common / "per_integration.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (common / "status.json").write_text(json.dumps(dict(
        common_integrations=len(records),
        comparison="Intersection of accepted post-baseline integrations and valid channels",
        baseline="Each branch fitted independently using the existing pipeline",
        difference_rms="Robust adjacent-channel difference / sqrt(2); channel correlations affect interpretation",
    ), indent=2) + "\n")
    return len(records)


def main(args):
    builder = make_builder(args)
    if args.limit_pairs is not None and args.limit_pairs < 1:
        raise ValueError("limit-pairs must be positive")
    if args.flag_head_tail_channel_number < 0:
        raise ValueError("flag-head-tail-channel-number must be nonnegative")
    directory = args.output_directory
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use an empty output directory to avoid mixing reductions")
    source = ArrayInput(args.input_prefix)
    pairs = source.pair_up_rows()
    if args.limit_pairs is not None:
        pairs = pairs[:args.limit_pairs]
    if not pairs:
        raise ValueError("No complete ON/OFF CAL pairs")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        version=__version__, input_prefix=str(source.path.resolve()),
        options={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        coordinate="native_channel_index", intensity_unit="Tcal", tcal_assumed=1.0,
        doppler_correction=False, opacity_correction=False, beam_correction=False,
        tsys_thresholds="relative: positive only; no kelvin lookup",
        pairing="scan pair plus INT; array position is separate from original INDEX",
        complete_pairs_selected=len(pairs), unpaired_rows=source.unpaired_rows,
        model_uncertainty="Noise is conditional on fixed model and normalization; covariance omitted",
        shape_scope="one fit per paired OFF integration",
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    options = Pipeline.Options.from_dict(dict(
        channel_width=1.0,
        flag_head_tail_channel_number=args.flag_head_tail_channel_number,
        max_rfi_channel=args.max_rfi_channel,
        Tsys_dynamic_threshold=False, Tsys_dynamic_threshold_selector="LookupTable",
        Tsys_min_threshold=0.0, Tsys_max_threshold=float("inf"), Tsys_min_success_rate=0.0,
    ))
    groups = {}
    for pair in pairs:
        info = pair.metadata["group"]
        key = (info["source"], info["sampler"], info["restfreq"])
        groups.setdefault(key, []).append(pair)
    failures = 0
    for number, (key, group_pairs) in enumerate(groups.items()):
        group_directory = directory / f"group_{number:03d}"
        group_directory.mkdir()
        (group_directory / "group.json").write_text(json.dumps(dict(
            source=str(key[0]), sampler=str(key[1]), restfreq=float(key[2]),
        ), indent=2) + "\n")
        for mode, reference in (("measured", None), ("modeled", builder)):
            mode_directory = group_directory / mode
            sink = DiagnosticWriter(mode_directory / "diagnostics")
            output = Pipeline(
                source, None, None, group_pairs, options,
                calibration=ArrayCalibration, reference_builder=reference, diagnostics=sink,
            ).calibrate().get_output()
            _save_output(output, mode_directory)
            print(f"{key}: {mode}: {output.reason}; dropped={dict(output.integration_dropped_reason)}")
            failures += not output.success
        common_count = _compare_common(group_directory)
        print(f"{key}: {common_count} integrations in common comparison")
        failures += common_count == 0
    print(f"Outputs: {directory.resolve()}; coordinate=channel, intensity=Tcal")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=help())
    configure_parser(parser)
    raise SystemExit(main(parser.parse_args()))
