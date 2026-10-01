import argparse
import collections
import loguru
import math
import numpy
import sys
import time
import traceback
from typing_extensions import Self

import tqdm  # type: ignore

from . import SigRefPairedRows, SigRefPairedHDUList
from . import GbtTsysLookupTable, GbtTsysHybridSelector, TsysThresholdSelector
from . import BeamEfficiency, PositionSwitchedCalibration, SDFits, ZenithOpacity
from . import Exposure, ExposureAggregator
from . import Spectrum, SpectrumAggregator
from .reference import ReferenceFitError

__all__ = [
    "Pipeline",
]


class Pipeline:

    class Input:
        sdfits: SDFits
        zenith_opacity: ZenithOpacity | None
        beam_efficiency: BeamEfficiency | None
        paired_rows: list[SigRefPairedRows]

    class FilteredIntegration:
        paired_row: SigRefPairedRows
        exposure: Exposure
        spectrum: Spectrum
        reference_residual: Spectrum | None
        correction_factor: Spectrum | None

    class PreBaselineOutput:
        filtered_integrations: list["Pipeline.FilteredIntegration"]

    class BaselineOutput:
        filtered_integrations: list["Pipeline.FilteredIntegration"]

    class PostBaselineOutput:
        pass

    class Output:
        success: bool = False
        reason: str = "Unknown reason"
        integration_dropped_reason: collections.Counter[str]

        total_exposure: Exposure
        exposure: Exposure
        spectrum: Spectrum
        reference_residual: Spectrum | None

        def __init__(self):
            self.integration_dropped_reason = collections.Counter()

    class Options:
        channel_width: float
        flag_head_tail_channel_number: int
        max_rfi_channel: int
        Tsys_dynamic_threshold: bool
        Tsys_dynamic_threshold_selector: str
        Tsys_min_threshold: float
        Tsys_max_threshold: float
        Tsys_min_success_rate: float

        @classmethod
        def from_dict(cls, d: dict) -> Self:
            res = cls()
            for field_name, field_type in cls.__annotations__.items():
                if field_name not in d:
                    continue
                if not isinstance(d[field_name], field_type):
                    loguru.logger.error(
                        f"Field {field_name} has type {type(d[field_name])}. Expected {field_type}"
                    )
                    continue
                setattr(res, field_name, d[field_name])
            return res

        @classmethod
        def from_namespace(cls, ns: argparse.Namespace) -> Self:
            return cls.from_dict(vars(ns))

    class Halt(RuntimeError):
        pass

    _input: Input
    _pre_baseline_output: PreBaselineOutput
    _baseline_output: BaselineOutput
    _post_baseline_output: PostBaselineOutput
    _output: Output
    _options: Options

    def __init__(
        self,
        sdfits: SDFits,
        zenith_opacity: ZenithOpacity | None,
        beam_efficiency: BeamEfficiency | None,
        paired_rows: list[SigRefPairedRows],
        options: Options,
        *,
        calibration=PositionSwitchedCalibration,
        reference_builder=None,
    ):
        self.calibration = calibration
        self.reference_builder = reference_builder
        self.sdfits = sdfits
        self.zenith_opacity = zenith_opacity
        self.beam_efficiency = beam_efficiency
        self.paired_rows = paired_rows

        self._input = self.Input()
        self._pre_baseline_output = self.PreBaselineOutput()
        self._baseline_output = self.BaselineOutput()
        self._post_baseline_output = self.PostBaselineOutput()
        self._output = self.Output()
        self._timings: collections.defaultdict[str, float] = collections.defaultdict(float)

        self._input.sdfits = sdfits
        self._input.zenith_opacity = zenith_opacity
        self._input.beam_efficiency = beam_efficiency
        self._input.paired_rows = paired_rows
        self._options = options

    def _get_tsys_threshold_selector(self) -> TsysThresholdSelector | None:
        if self._options.Tsys_dynamic_threshold:
            if self._options.Tsys_dynamic_threshold_selector == "LookupTable":
                return GbtTsysLookupTable()
            elif self._options.Tsys_dynamic_threshold_selector == "Hybrid":
                return GbtTsysHybridSelector()
            else:
                loguru.logger.error(
                    f"Unexpected Tsys_dynamic_threshold_selector = {self._options.Tsys_dynamic_threshold_selector}."
                )
                return None
        return None

    def _check_tsys_threshold(
        self,
        spectrum_metadata: dict,
        tsys_threshold_selector: TsysThresholdSelector | None,
        tsys_stats: dict[str, int],
    ) -> bool:
        Tsys_min_threshold = self._options.Tsys_min_threshold
        Tsys_max_threshold = self._options.Tsys_max_threshold
        if tsys_threshold_selector is not None:
            dynamic_threshold = tsys_threshold_selector.get_threshold(
                spectrum_metadata["ObsFreq"]
            )
            if dynamic_threshold is None:
                loguru.logger.error("Tsys threshold selector returned None.")
                return False
            Tsys_min_threshold *= dynamic_threshold
            Tsys_max_threshold *= dynamic_threshold

        tsys_stats["total"] += 1
        if not spectrum_metadata["Tsys"] > Tsys_min_threshold:
            self._output.integration_dropped_reason["Tsys exceeds min threshold"] += 1
            return False
        if not spectrum_metadata["Tsys"] < Tsys_max_threshold:
            self._output.integration_dropped_reason["Tsys exceeds max threshold"] += 1
            return False
        tsys_stats["succeed"] += 1
        return True

    def _check_tsys_stats(self, tsys_stats: dict[str, int]) -> bool:
        if tsys_stats["total"] == 0:
            tsys_success_rate = 0.0
        else:
            tsys_success_rate = tsys_stats["succeed"] / tsys_stats["total"]
        if tsys_success_rate < self._options.Tsys_min_success_rate:
            return False
        return True

    def _check_num_rfi_channel(
        self,
        spectrum: Spectrum,
    ) -> bool:
        assert spectrum.flag is not None
        if self._options.max_rfi_channel >= 0:
            rfi_in_body_count = (
                ~spectrum.flagged(Spectrum.FlagReason.CHUNK_EDGES)
                & spectrum.flagged(
                    Spectrum.FlagReason.FREQUENCY_DOMAIN_RFI
                    | Spectrum.FlagReason.TIME_DOMAIN_RFI
                )
            ).sum()
            if rfi_in_body_count > self._options.max_rfi_channel:
                self._output.integration_dropped_reason["Too many RFI channels"] += 1
                return False
        return True

    def _get_correction_factor(
        self, sigrefpair: SigRefPairedHDUList
    ) -> Spectrum | None:
        correction_factors = []
        if self._input.zenith_opacity is not None:
            opacity_correction_factor = (
                self.calibration.get_opacity_correction_factor(
                    sigrefpair["sig"]["caloff"], self._input.zenith_opacity
                )
            )
            if opacity_correction_factor is None:
                raise self.Halt("Opacity temperature correction enabled but failed.")
            correction_factors.append(opacity_correction_factor)
        if self._input.beam_efficiency is not None:
            efficiency_correction_factor = (
                self.calibration.get_efficiency_correction_factor(
                    sigrefpair["sig"]["caloff"], self._input.beam_efficiency
                )
            )
            if efficiency_correction_factor is None:
                raise self.Halt("Beam efficiency correction enabled but failed.")
            correction_factors.append(efficiency_correction_factor)
        if len(correction_factors) == 0:
            return None
        return math.prod(correction_factors)  # type: ignore

    def _get_debug_indices(self, paired_row: SigRefPairedRows):
        return {
            f"{sigref},{calonoff}": int(paired_row[sigref][calonoff]["INDEX"].iloc[0])
            for sigref in paired_row
            for calonoff in paired_row[sigref]
        }

    def _log_timing_report(self):
        if not self._timings:
            return
        report = ", ".join(
            f"{name}={elapsed:.0f}" if name.endswith("_terms")
            else f"{name}={elapsed:.3f}s"
            for name, elapsed in sorted(self._timings.items())
        )
        loguru.logger.info(
            f"Pipeline timing for {len(self._input.paired_rows)} paired integrations: {report}"
        )

    def _run_stage_pre_baseline(self) -> bool:
        total_exposure_aggregator = ExposureAggregator(
            ExposureAggregator.LinearTransformer(self._options.channel_width)
        )

        tsys_threshold_selector = self._get_tsys_threshold_selector()
        tsys_stats = dict(succeed=0, total=0)

        self._pre_baseline_output.filtered_integrations = list()
        for paired_row in tqdm.tqdm(
            self._input.paired_rows, dynamic_ncols=True, smoothing=0.0, leave=False
        ):
            try:
                timing_started = time.perf_counter()
                sigrefpair = paired_row.get_paired_hdu(self._input.sdfits)
                exposure = self.calibration.get_exposure(sigrefpair["sig"])
                if exposure is None:
                    self._timings["pre.io_and_checks"] += time.perf_counter() - timing_started
                    self._output.integration_dropped_reason["No exposure returned"] += 1
                    continue
                total_exposure_aggregator.merge(exposure)

                discarded = self.calibration.should_be_discarded(sigrefpair)
                self._timings["pre.io_and_checks"] += time.perf_counter() - timing_started
                if discarded:
                    self._output.integration_dropped_reason["Failed prechecks"] += 1
                    continue

                timing_started = time.perf_counter()
                (
                    spectrum,
                    spectrum_metadata,
                ) = self.calibration.get_calibrated_spectrum(
                    sigrefpair, freq_kwargs=dict(unit="Hz"), return_metadata=True,
                    reference_builder=self.reference_builder,
                    timings=self._timings,
                )
                self._timings["pre.calibration"] += time.perf_counter() - timing_started
                if spectrum is None:
                    self._output.integration_dropped_reason[
                        "No calibrated spectrum returned"
                    ] += 1
                    continue

                if not self._check_tsys_threshold(
                    spectrum_metadata,
                    tsys_threshold_selector,
                    tsys_stats,
                ):
                    continue

                timing_started = time.perf_counter()
                if self.reference_builder is None:
                    # Keep the historical measured-reference flagging path
                    # unchanged when the standard calibration is selected.
                    (
                        spectrum.flag_nan()
                        .flag_head_tail(
                            nchannel=self._options.flag_head_tail_channel_number
                        )
                        .flag_time_domain_rfi(spectrum_metadata)
                        .flag_frequency_domain_rfi()
                        .flag_valid_data()
                    )
                else:
                    # Run the historical checks on both calibrated spectra.
                    # The model branch keeps either branch's RFI decision and
                    # its own unsupported-edge/NaN flags.
                    measured_spectrum = spectrum_metadata["measured_calibrated"]
                    (
                        measured_spectrum.flag_nan()
                        .flag_head_tail(
                            nchannel=self._options.flag_head_tail_channel_number
                        )
                        .flag_time_domain_rfi(spectrum_metadata)
                        .flag_frequency_domain_rfi()
                        .flag_valid_data()
                    )
                    (
                        spectrum.flag_nan()
                        .flag_head_tail(
                            nchannel=self._options.flag_head_tail_channel_number
                        )
                        .flag_frequency_domain_rfi()
                    )
                    assert spectrum.flag is not None
                    assert measured_spectrum.flag is not None
                    invalid_reasons = (
                        Spectrum.FlagReason.NAN
                        | Spectrum.FlagReason.CHUNK_EDGES
                        | Spectrum.FlagReason.FREQUENCY_DOMAIN_RFI
                        | Spectrum.FlagReason.TIME_DOMAIN_RFI
                    ).value
                    spectrum.flag[:] = (
                        (spectrum.flag | measured_spectrum.flag) & invalid_reasons
                    )
                    spectrum.flag_valid_data()
                self._timings["pre.rfi"] += time.perf_counter() - timing_started

                if not self._check_num_rfi_channel(spectrum):
                    continue

                exposure.exposure[
                    ~spectrum.flagged(Spectrum.FlagReason.VALID_DATA)
                ] = 0.0

                filtered_integration = self.FilteredIntegration()
                filtered_integration.paired_row = paired_row
                filtered_integration.exposure = exposure
                filtered_integration.spectrum = spectrum
                filtered_integration.reference_residual = spectrum_metadata.get(
                    "reference_residual"
                )
                timing_started = time.perf_counter()
                filtered_integration.correction_factor = self._get_correction_factor(
                    sigrefpair
                )
                self._timings["pre.correction"] += time.perf_counter() - timing_started
                self._pre_baseline_output.filtered_integrations.append(
                    filtered_integration
                )
            except ReferenceFitError as e:
                self._output.integration_dropped_reason["Reference model fit failed"] += 1
            except self.Halt as e:
                loguru.logger.critical(*e.args)
                tqdm.tqdm.write(*e.args)
                sys.exit(1)
            except Exception:
                loguru.logger.critical(
                    f"Uncaught exception while working on {self._get_debug_indices(paired_row)}\n{traceback.format_exc()}"
                )
                self._output.integration_dropped_reason["Uncaught exception"] += 1

        self._output.total_exposure = total_exposure_aggregator.get_spectrum()

        if not self._check_tsys_stats(tsys_stats):
            self._output.reason = "Tsys threshold success rate less than required."
            return False

        if len(self._pre_baseline_output.filtered_integrations) == 0:
            self._output.reason = "All integrations are filtered out."
            return False

        return True

    def _run_stage_baseline(self) -> bool:
        timing_started = time.perf_counter()
        pre_baseline_aggregated_spectrum = (
            SpectrumAggregator(
                SpectrumAggregator.LinearTransformer(self._options.channel_width)
            )
            .merge_all(
                spectrum - spectrum.from_callable(baseline_result[0])
                for spectrum, baseline_result in (
                    (
                        filtered_integration.spectrum,
                        filtered_integration.spectrum.fit_baseline(
                            method="polynomial",
                            polynomial_options=dict(degree=20),
                            timings=self._timings,
                            timing_prefix="baseline.signal_prepass",
                        ),
                    )
                    for filtered_integration in self._pre_baseline_output.filtered_integrations
                )
                if baseline_result is not None
            )
            .get_spectrum()
            # Flag strong signals where baseline fitting is affected a lot
            .flag_signal(
                nadjacent=31,
                ignore_flags=Spectrum.FlagReason.CHUNK_EDGES
                | Spectrum.FlagReason.FREQUENCY_DOMAIN_RFI
                | Spectrum.FlagReason.TIME_DOMAIN_RFI,
            )
            # Flag weak signals that require more accurate baseline fitting
            .flag_signal(
                nadjacent=dict(baseline=255, chisq=31),
                ignore_flags=Spectrum.FlagReason.CHUNK_EDGES
                | Spectrum.FlagReason.FREQUENCY_DOMAIN_RFI
                | Spectrum.FlagReason.TIME_DOMAIN_RFI
                | Spectrum.FlagReason.SIGNAL,
            )
        )
        self._timings["baseline.signal_prepass"] += time.perf_counter() - timing_started

        self._baseline_output.filtered_integrations = list()
        timing_started = time.perf_counter()
        for filtered_integration in tqdm.tqdm(
            self._pre_baseline_output.filtered_integrations,
            dynamic_ncols=True,
            smoothing=0.0,
            leave=False,
        ):
            paired_row = filtered_integration.paired_row
            exposure = filtered_integration.exposure
            spectrum = filtered_integration.spectrum
            reference_residual = filtered_integration.reference_residual
            correction_factor = filtered_integration.correction_factor
            try:
                spectrum.copy_flags(
                    Spectrum.FlagReason.SIGNAL, pre_baseline_aggregated_spectrum
                )

                residual_threshold = 0.1
                if self.reference_builder is not None:
                    residual_threshold *= math.sqrt(2)

                fit_timing_started = time.perf_counter()
                baseline_result = spectrum.fit_baseline(
                    method="hybrid",
                    polynomial_options=dict(max_degree=20),
                    lomb_scargle_options=dict(
                        min_num_terms=0, max_num_terms=40, max_cycle=32
                    ),
                    residual_threshold=residual_threshold,
                    timings=self._timings,
                    timing_prefix="baseline.integration",
                )
                self._timings["baseline.integration_fit"] += (
                    time.perf_counter() - fit_timing_started
                )
                if baseline_result is None:
                    self._output.integration_dropped_reason[
                        "Failed baseline fitting"
                    ] += 1
                    continue

                baseline, _ = baseline_result
                evaluation_timing_started = time.perf_counter()
                baseline_substracted_spectrum = spectrum - spectrum.from_callable(
                    baseline
                )
                self._timings["baseline.integration_evaluation"] += (
                    time.perf_counter() - evaluation_timing_started
                )
                if correction_factor is not None:
                    correction_timing_started = time.perf_counter()
                    baseline_substracted_spectrum *= correction_factor
                    if reference_residual is not None:
                        reference_residual *= correction_factor
                    self._timings["baseline.integration_correction"] += (
                        time.perf_counter() - correction_timing_started
                    )

                filtered_integration = self.FilteredIntegration()
                filtered_integration.paired_row = paired_row
                filtered_integration.exposure = exposure
                filtered_integration.spectrum = baseline_substracted_spectrum
                filtered_integration.reference_residual = reference_residual
                self._baseline_output.filtered_integrations.append(filtered_integration)
            except self.Halt as e:
                loguru.logger.critical(*e.args)
                tqdm.tqdm.write(*e.args)
                sys.exit(1)
            except Exception:
                message = (
                    f"Baseline exception for {self._get_debug_indices(paired_row)}\n"
                    f"{traceback.format_exc()}"
                )
                loguru.logger.critical(message)
                tqdm.tqdm.write(message)
                self._output.integration_dropped_reason["Uncaught exception"] += 1

        self._timings["baseline.integration_fits"] += time.perf_counter() - timing_started

        if len(self._baseline_output.filtered_integrations) == 0:
            tqdm.tqdm.write(
                f"ZERO BASELINE SURVIVORS: "
                f"{len(self._pre_baseline_output.filtered_integrations)} entered baseline stage; "
                f"drops={dict(self._output.integration_dropped_reason)}"
            )
            self._output.reason = "All integrations failed baseline fitting."
            return False
        return True

    def _run_stage_post_baseline(self) -> bool:
        timing_started = time.perf_counter()
        self._output.exposure = (
            ExposureAggregator(
                ExposureAggregator.LinearTransformer(self._options.channel_width)
            )
            .merge_all(self._output.total_exposure.split(), init_only=True)
            .merge_all(
                filtered_integration.exposure
                for filtered_integration in self._baseline_output.filtered_integrations
            )
            .get_spectrum()
        )
        self._timings["post.exposure_aggregation"] += time.perf_counter() - timing_started
        timing_started = time.perf_counter()
        self._output.spectrum = (
            SpectrumAggregator(
                SpectrumAggregator.LinearTransformer(self._options.channel_width)
            )
            .merge_all(
                filtered_integration.spectrum
                for filtered_integration in self._baseline_output.filtered_integrations
            )
            .get_spectrum()
        )
        self._timings["post.science_aggregation"] += time.perf_counter() - timing_started
        if self.reference_builder is not None:
            timing_started = time.perf_counter()
            self._output.reference_residual = (
                SpectrumAggregator(
                    SpectrumAggregator.LinearTransformer(self._options.channel_width)
                )
                .merge_all(
                    filtered_integration.reference_residual
                    for filtered_integration in self._baseline_output.filtered_integrations
                    if filtered_integration.reference_residual is not None
                )
                .get_spectrum()
            )
            self._timings["post.reference_residual_aggregation"] += (
                time.perf_counter() - timing_started
            )
        else:
            self._output.reference_residual = None
        return True

    @staticmethod
    def flag_persistent_model_rfi(
        reference_residual: Spectrum,
        spectrum: Spectrum | Exposure,
        exposure: Exposure | None = None,
    ):
        """Use the aggregated OFF residual to flag persistent narrow RFI.

        The two-argument form is retained for callers of the old helper. The
        smooth-reference pipeline uses the three-argument form, in which the
        detector never examines the science-spectrum morphology or SIGNAL mask.
        """
        legacy_mode = exposure is None
        if legacy_mode:
            exposure = spectrum  # type: ignore[assignment]
            spectrum = reference_residual
            assert isinstance(spectrum, Spectrum)
            reference_residual = spectrum
        assert isinstance(spectrum, Spectrum)
        assert isinstance(exposure, Exposure)
        assert reference_residual.frequency is not None
        assert reference_residual.noise is not None
        assert reference_residual.flag is not None
        assert spectrum.frequency is not None
        assert spectrum.flag is not None

        valid = (
            reference_residual.flagged(Spectrum.FlagReason.VALID_DATA)
            & numpy.isfinite(reference_residual.intensity)
            & numpy.isfinite(reference_residual.noise)
            & (reference_residual.noise > 0)
        )
        z = numpy.abs(reference_residual.intensity / reference_residual.noise)
        persistent_birdies = valid & (z > 5.0)

        if legacy_mode:
            # Compatibility for the old direct helper API only. The actual
            # smooth-reference pipeline always supplies an independent residual.
            persistent_birdies &= ~spectrum.flagged(Spectrum.FlagReason.SIGNAL)

        if not numpy.array_equal(reference_residual.frequency, spectrum.frequency):
            raise ValueError("Reference residual and science spectra use different grids")
        if not numpy.array_equal(exposure.frequency, spectrum.frequency):
            raise ValueError("Science spectrum and exposure use different grids")

        overlap_signal = persistent_birdies & spectrum.flagged(Spectrum.FlagReason.SIGNAL)
        newly_invalidated = persistent_birdies & spectrum.flagged(
            Spectrum.FlagReason.VALID_DATA
        )
        spectrum.flag[persistent_birdies] |= (
            Spectrum.FlagReason.FREQUENCY_DOMAIN_RFI.value
        )
        spectrum.flag[persistent_birdies] &= ~Spectrum.FlagReason.VALID_DATA.value
        exposure.exposure[persistent_birdies] = 0.0
        loguru.logger.info(
            "Persistent OFF-residual birdies: "
            f"{int(persistent_birdies.sum())} channels, "
            f"{int(overlap_signal.sum())} overlap SIGNAL, "
            f"{int(newly_invalidated.sum())} newly invalidated science channels."
        )

    def calibrate(self) -> Self:
        timing_started = time.perf_counter()
        if not self._run_stage_pre_baseline():
            self._timings["stage.pre_baseline"] += time.perf_counter() - timing_started
            self._log_timing_report()
            return self
        self._timings["stage.pre_baseline"] += time.perf_counter() - timing_started
        timing_started = time.perf_counter()
        if not self._run_stage_baseline():
            self._timings["stage.baseline"] += time.perf_counter() - timing_started
            self._log_timing_report()
            return self
        self._timings["stage.baseline"] += time.perf_counter() - timing_started
        timing_started = time.perf_counter()
        if not self._run_stage_post_baseline():
            self._timings["stage.post_baseline"] += time.perf_counter() - timing_started
            self._log_timing_report()
            return self
        self._timings["stage.post_baseline"] += time.perf_counter() - timing_started
        self._output.success = True
        self._output.reason = "Success"
        self._log_timing_report()
        return self

    def get_output(self) -> Output:
        return self._output
