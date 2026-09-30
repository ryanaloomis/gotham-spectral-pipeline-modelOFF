# Smooth OFF reference

The normal SDFITS pipeline uses the measured OFF spectrum by default. To use a
smooth model of the CAL-combined OFF bandpass, pass `--reference-mode smooth` to
`run_pipeline`:

```bash
python -m gotham_spectral_pipeline run_pipeline \
  --sdfits input.fits \
  --channel_width 1.0 \
  --reference-mode smooth
```

The model is a robust cubic spline fit in native channel coordinates. It uses
median-compressed OFF counts and fixed, broad knots, then estimates one scalar
normalization from the measured OFF. The calibration remains
`Tsys * (ON - reference) / reference`; only the reference spectrum changes.
The default mode and all downstream baseline, correction, and aggregation steps
remain the existing pipeline behavior.

In smooth mode, the pipeline runs the existing per-integration RFI checks on
both measured-reference and modeled-reference spectra, then applies the union
of their flags to the modeled spectrum. After integrations are combined, it
runs the frequency-domain RFI detector on the aggregate and flags additional
narrow features before saving. As with any automatic RFI detector, inspect
flags near expected astronomical lines.

The spline implementation and its `ReferenceBuilder` interface are isolated in
`gotham_spectral_pipeline/reference.py`. Its defaults currently use 512-channel
knot spacing, 32-channel median bins, 1024 excluded channels at each band edge,
and four robust-clipping iterations. Unsupported edges are marked invalid; a
failed fit drops that integration rather than silently reverting to measured
OFF calibration.

Small numerical and calibration checks are in `tests/test_reference.py` and
`tests/test_calibration_reference.py`.
