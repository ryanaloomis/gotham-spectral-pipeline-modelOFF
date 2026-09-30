# Smooth OFF experiment

The default SDFITS reduction still uses the original measured-reference arithmetic.
The optional `SmoothOffReference` fits CAL-combined raw OFF counts before calibration
and supplies both the numerator reference and denominator of `Tsys * (ON - M) / M`.
It does not change the measured-OFF Tsys estimate. Measured diode-state metadata
continues to drive the existing time-domain RFI detector.

## Your NPY/CSV session

Run from this repository in the pipeline's Python >=3.10 environment (dependencies
are listed in `environment.yml`). No additional package is required by this change.

Start with a limited reduction:

```bash
python -m gotham_spectral_pipeline run_export_ab \
  --input-prefix /Users/rloomis/Science/GOTHAM/AGBT19B_047_102_SAMPLERA1_0 \
  --output-directory output/model_off_smoke \
  --limit-pairs 2
```

Then run the session with OFF normalization:

```bash
python -m gotham_spectral_pipeline run_export_ab \
  --input-prefix /Users/rloomis/Science/GOTHAM/AGBT19B_047_102_SAMPLERA1_0 \
  --output-directory output/model_off_session \
  --normalization off
```

To test ON normalization, repeat with `--normalization on` and a new output
directory. `--normalization none` preserves the fitted OFF amplitude directly.
Output directories must be empty to prevent stale results entering comparisons.
The full diagnostic output may occupy several GB; arrays are compressed and
written per integration rather than retained in memory.

The export adapter:

- Memory-maps the NPY and associates its array positions with CSV row positions.
  Original `INDEX` identifiers are retained separately.
- Matches OFF/ON scans and explicit `INT` values, then requires both CAL states.
  It rejects duplicate matches and counts unmatched rows in the manifest.
- Uses `TCAL=1` as a relative unit and recomputes `Tsys/Tcal` from diode states.
  The CSV's placeholder `TSYS=1` is not used as a temperature measurement.
- Uses **native channel index** as the numerical coordinate. The generic NPZ
  field remains named `frequency`, but its unit here is **channels**, not Hz.
- Runs the existing RFI, baseline, and aggregation machinery separately for
  measured and modeled spectra, with positive relative Tsys required.
- Applies no Doppler, opacity, or beam correction to this incomplete export.
  Source/sampler/rest-frequency groups remain separate.

These channel-aligned averages are diagnostic products, not final sky-frequency
science spectra. Missing TCAL, VFRAME, and full WCS prevent reproducing the native
physical calibration. The native SDFITS path below retains those corrections.

## Outputs

`manifest.json` records input path, code version, settings, units, and assumptions.
Each `group_NNN` has:

```text
group.json
measured/
  status.json
  spectrum.npz, exposure.npz, total_exposure.npz   # when available
  diagnostics/
modeled/
  status.json
  spectrum.npz, exposure.npz, total_exposure.npz
  diagnostics/
common/
  status.json
  measured.npz, modeled.npz                     # if common data survive
  per_integration.csv
```

Diagnostics are named by the four input-row identifiers and processing stage:

- `calibration`: spectrum before flagging/baseline; Tsys/Tcal; measured diode
  diagnostics; for models, raw ON/OFF, fitted shape/reference, residual, knots,
  coefficients, fit mask, support mask, scalar normalization, and normalization mask.
- `flagged`: spectrum with RFI and validity flags, before the RFI-count rejection.
- `legacy_rfi` (smooth-reference mode): measured-reference spectrum after the
  historical per-integration RFI decisions.
- `model_rfi` (smooth-reference mode): modeled-reference spectrum after running
  the same per-integration RFI checks on its own calibrated values.
- `selection`: acceptance or rejection reasons after pre-baseline processing.
- `baseline`: accepted spectrum after baseline subtraction, plus fitted baseline.
- `reference_failure` / `baseline_failure`: failure details when applicable.

Each JSON document maps array values to named keys in its accompanying NPZ.
Model arrays use **native channel order** and include their corresponding
coordinates; `Spectrum` arrays use its usual ascending coordinate order.

`common` uses only integrations accepted through baseline fitting in both runs,
and intersects their valid channels. It aggregates each branch using its own
noise weights on that shared selection. Baselines were fitted independently;
the comparison does not force identical baseline fits. The CSV includes standard
deviation and a robust adjacent-channel difference estimate divided by sqrt(2).
The latter probes narrow-scale noise but is affected by channel correlations.
The command returns nonzero if either branch fails or no common integrations survive.

## Native SDFITS

Use the existing command and arguments. Additional options are:

```text
--reference-mode measured|smooth     default: measured
--normalization off|on|none          default: off
--knot-spacing 512                  native channels
--fit-bin 32                        channels per median-compressed bin
--model-edge-channels 1024
--model-clip-sigma 5
--model-iterations 4
--diagnostics-directory PATH         optional; must be empty
```

Run measured and smooth modes with separate output prefixes and diagnostic
directories. Normal native Doppler, beam, opacity, and Tsys selection behavior is
unchanged. The exported-data command shares the model settings above.

## Model and uncertainty conventions

- One shape is fitted to each paired OFF. Reusing one shape per scan, as in the
  prototype, is not implemented in this first version.
- Fixed cubic knots prohibit adaptive channel-scale fitting. The defaults give
  roughly 0.732 MHz spacing for this session. Validate that scale empirically.
- Compression uses medians, followed by iterative clipping. Every knot interval
  must retain at least two real compressed bins. Unsupported gaps or invalid
  models fail explicitly; there is no automatic measured-OFF fallback.
- Unsupported edges are clamped numerically and marked `CHUNK_EDGES`, so they
  cannot contribute to baseline fitting or final aggregation.
- In smooth-reference mode, time/frequency RFI checks run on both measured- and
  modeled-reference spectra. The model's mask is the union of both results plus
  its own unsupported-edge and NaN flags. This preserves historical detections
  while allowing the model-specific detector to flag newly exposed features.
- After combining integrations, smooth-reference mode runs the same frequency-
  domain detector once more on the aggregate to catch narrow features that were
  individually weak but become significant in the combined spectrum. It flags detected channels
  invalid in the final spectrum and zeros their output exposure. This detector
  can still confuse narrow astronomical lines with birdies; inspect the saved
  flags and protect expected sky-line channels when interpreting results.
- Shape has median one on fitted channels. `off` and `on` fit a robust scalar
  target/shape ratio; `none` uses the original fitted amplitude. ON normalization
  can absorb broadband source emission as well as gain changes.
- Modeled noise uses the existing CAL-combined ON radiometer estimate, rescaled
  by measured OFF/model. Measured OFF thermal variance is not added to it.
  This is conditional on a fixed model and normalization, not a full covariance
  calculation. Model, gain, and Tsys uncertainty are not propagated. Validate the
  apparent noise improvement and line recovery rather than assuming sqrt(2).

The builder contract lives in `reference.py` and is independent of FITS,
`Spectrum`, and the pipeline. An alternative model supplies a `ReferenceResult`
in native raw-count/channel order, including explicit support and fit masks.

## Checks

Small isolated numerical checks (NumPy/SciPy only):

```bash
python tests/test_reference.py
```

Full-environment contract checks (small synthetic arrays; no session reduction):

```bash
python -m unittest discover -s tests -v
```

These check legacy arithmetic, reference substitution, descending-grid ordering,
diode metadata preservation, and array-row/integration matching. Full-environment
checks and the session reductions were left for you to run. For scientific
validation, compare both normalization modes and multiple knot spacings, inspect
fit residuals/rejections, and measure injected-line recovery and noise correlation
before interpreting a reduction in RMS as increased sensitivity.
