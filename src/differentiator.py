""" Numerical differentiation and spectral error-analysis utilities: turns noisy
measurements into an end-to-end error budget for the simulated gravity signal. """

import h5py

from typing import Any
from pathlib import Path

from matplotlib.pylab import seed
import numpy as np
from findiff import Diff
from pycbc import types, psd
from scipy.spatial.transform import Rotation


from plotter import Plotter

# Define constants
LOWEST_FREQUENCY_RESOLUTION_HZ = 5e-6


def save_to_hdf5(
    noise_model_version: int,
    file_name: str,
    frequencies: np.ndarray,
    asd_values: np.ndarray,
    spectrum_name: str,
    ordinate_label: str,
) -> Path:
    """Save a frequency/ASD spectrum under the versioned output directory.

    Args:
        noise_model_version (int): Noise model version (1 or 2), used to pick
            the output subdirectory.
        file_name (str): Filename to save the HDF5 file as, within that
            directory.
        frequencies (np.ndarray): 1D array of frequencies [Hz].
        asd_values (np.ndarray): 1D array of ASD values, matching frequencies
            in shape.
        spectrum_name (str): spectrum name, stored as an HDF5
            attribute for reference.
        ordinate_label (str): Unit label for the ASD values, stored as an HDF5 attribute.

    Returns:
        Path: Path to the HDF5 file that was written.

    Raises:
        ValueError: If frequencies and asd_values have different shapes.
    """

    frequencies = np.asarray(frequencies, dtype=float).reshape(-1)
    asd_values = np.asarray(asd_values, dtype=float).reshape(-1)

    if frequencies.shape != asd_values.shape:
        raise ValueError("frequencies and asd_values must have the same shape.")

    output_directory = Path(f"./output/spectra/version_{noise_model_version}")
    output_directory.mkdir(parents=True, exist_ok=True)
    file_path = output_directory / file_name

    with h5py.File(file_path, "w") as h5_file:
        h5_file.create_dataset("frequencies", data=frequencies)
        h5_file.create_dataset("asd", data=asd_values)
        h5_file.attrs["spectrum_name"] = spectrum_name
        h5_file.attrs["frequency_units"] = "Hz"
        h5_file.attrs["asd_units"] = ordinate_label
        h5_file.attrs["noise_model_version"] = int(noise_model_version)

    return file_path


def compute_acceleration(
        position: np.ndarray,
        time: np.ndarray,
        accuracy: int
        ) -> tuple[np.ndarray, dict]:
    """Compute acceleration by double-differentiating a position time history.

    Args:
        position (np.ndarray): Shape (N, 3). Cartesian position vectors sampled
            at time.
        time (np.ndarray): Shape (N,). Time values corresponding to position,
            uniform or non-uniform.
        accuracy (int): Order of accuracy for the finite-difference
            approximation.

    Returns:
        tuple[np.ndarray, dict]: (acceleration, metadata), where acceleration
            is shape (N, 3) and metadata contains a "stencil_points" entry
            describing the finite-difference stencil used at each sample.
    """

    num_epochs = position.shape[0]
    acceleration = np.zeros_like(position)
    metadata = {}

    # second derivative operator along time axis
    second_derivative_operator = Diff(0, time, acc=accuracy)**2

    # Compute acceleration for each spatial component
    for i in range(3):
        acceleration[:, i] = second_derivative_operator(position[:, i])

    # Capture stencils used at each k (1D shape only)
    stencil_points = second_derivative_operator.stencil((num_epochs,))

    metadata["stencil_points"] = stencil_points

    return acceleration, metadata


def differentiate_inter_satellite_range(
    ranges: np.ndarray,
    time: np.ndarray,
    accuracy: int,
    derivative_order: int,
) -> np.ndarray:
    """Differentiate an inter-satellite range history with a requested finite-difference accuracy.

    Args:
        ranges (np.ndarray): 1D inter-satellite range history [m].
        time (np.ndarray): 1D time values matching ranges, uniform or
            non-uniform.
        accuracy (int): Finite-difference accuracy order.
        derivative_order (int): 1 for range-rate, 2 for range-acceleration.

    Returns:
        np.ndarray: The differentiated range history, same shape as ranges.

    Raises:
        ValueError: If derivative_order is not 1 or 2.
    """

    ranges = np.asarray(ranges, dtype=float).reshape(-1)
    time = np.asarray(time, dtype=float).reshape(-1)

    if derivative_order == 1:
        derivative_operator = Diff(0, time, acc=accuracy)
    elif derivative_order == 2:
        derivative_operator = Diff(0, time, acc=accuracy)**2
    else:
        raise ValueError("Only first- and second-order derivatives are supported.")

    return np.asarray(derivative_operator(ranges), dtype=float)


def build_orbital_phasing_period_exclusion_mask(
    time: np.ndarray,
    guidance_log: list[dict[str, Any]] | None,
) -> np.ndarray:
    """Build a boolean mask marking samples that fall within any orbit-phasing maneuver.

    Samples during and between an entry and exit burn are excluded from noise
    differentiation and spectral estimation, since the impulsive maneuvers
    introduce discontinuities that finite-difference stencils and Welch PSD
    estimation cannot handle correctly.

    Args:
        time (np.ndarray): 1D array of epochs to build the mask for.
        guidance_log (list[dict[str, Any]] | None): Guidance maneuver records, each expected to
            have a "trigger_epoch" and one of "actual_phasing_end_epoch",
            "exit_epoch", or "final_phasing_epoch" marking the maneuver's end.
            None or an empty list is treated as "no maneuvers occurred."

    Returns:
        np.ndarray: Boolean mask, same shape as time, True where the sample
            falls within any maneuver's trigger-to-exit window (inclusive).
    """

    time = np.asarray(time, dtype=float).reshape(-1)
    exclusion_mask = np.zeros(time.shape[0], dtype=bool)

    if not guidance_log:
        return exclusion_mask

    for maneuver_record in guidance_log:
        trigger_epoch = float(maneuver_record["trigger_epoch"])
        phasing_end_epoch = float(
            maneuver_record.get(
                "actual_phasing_end_epoch",
                maneuver_record.get(
                    "exit_epoch",
                    maneuver_record["final_phasing_epoch"],
                ),
            )
        )
        exclusion_mask |= (time >= trigger_epoch) & (time <= phasing_end_epoch)

    return exclusion_mask


def get_contiguous_valid_propagation_segments(valid_propagation_segments_mask: np.ndarray) -> list[np.ndarray]:
    """Split a boolean validity mask into contiguous runs of valid sample indices.

    Args:
        valid_propagation_segments_mask (np.ndarray): Boolean mask over
            samples, True where the sample is usable.

    Returns:
        list[np.ndarray]: One integer index array per contiguous run of True
            values in the mask, in ascending order. Empty if no True values
            are present.
    """

    valid_indices = np.flatnonzero(np.asarray(valid_propagation_segments_mask, dtype=bool))
    if valid_indices.size == 0:
        return []

    split_indices = np.where(np.diff(valid_indices) > 1)[0] + 1
    return [np.asarray(propagation_segment, dtype=int) for propagation_segment in np.split(valid_indices, split_indices)]


def get_segment_length_from_frequency_resolution(
    time_step: float,
    lowest_frequency_resolution_hz: float,
) -> tuple[int, float]:
    """Convert a target Welch-estimated PSD lowest-frequency resolution into a segment length in samples.

    Args:
        time_step (float): Sample spacing, in seconds.
        lowest_frequency_resolution_hz (float): Desired lowest resolvable
            frequency [Hz], which sets the minimum Welch segment duration
            (duration = 1 / lowest_frequency_resolution_hz).

    Returns:
        tuple[int, float]: (segment_length, requested_segment_duration_seconds),
            where segment_length is the segment length in samples (rounded to
            the nearest integer) and requested_segment_duration_seconds is the
            exact (pre-rounding) segment duration.

    Raises:
        ValueError: If time_step or lowest_frequency_resolution_hz is not
            positive, or if the resulting segment length rounds to less than
            one sample.
    """

    if time_step <= 0.0:
        raise ValueError("time_step must be positive.")
    if lowest_frequency_resolution_hz <= 0.0:
        raise ValueError("lowest_frequency_resolution_hz must be positive.")

    requested_segment_duration_seconds = 1.0 / lowest_frequency_resolution_hz
    raw_segment_length = requested_segment_duration_seconds / time_step
    segment_length = int(np.rint(raw_segment_length))

    if segment_length < 1:
        raise ValueError("The requested segment length is shorter than one sample.")

    return segment_length, requested_segment_duration_seconds


def build_welch_estimation_segment_settings(
    valid_propagation_segments_mask: np.ndarray,
    time_step: float,
    segment_length: int,
    segment_stride: int,
    *,
    emit_warnings: bool = False,
    warning_context: str = "Welch estimation",
) -> list[dict[str, Any]]:
    """Select contiguous propagation arcs long enough for the requested Welch estimation segment settings.

    An arc must be at least segment_length + segment_stride samples long to
    support pycbc's "median-mean" Welch averaging method; shorter arcs are
    skipped.

    Args:
        valid_propagation_segments_mask (np.ndarray): Boolean mask over
            samples, True where usable (not excluded by a maneuver).
        time_step (float): Sample spacing, in seconds.
        segment_length (int): Welch estimation segment length, in samples.
        segment_stride (int): Welch estimation segment stride, in samples.
        emit_warnings (bool): If True, computes diagnostic information about
         skipped arcs. Defaults to False.
        warning_context (str): Label describing the calling context, used only
            in the diagnostic message. Defaults to
            "Welch estimation".

    Returns:
        list[dict[str, Any]]: One settings dict per usable arc, each with keys
            "arc_idx", "propagation_arc" (index array), "segment_length",
            "segment_stride", "segment_duration_seconds", and
            "lowest_frequency_resolution_hz".

    Raises:
        ValueError: If segment_length or segment_stride is less than one
            sample.
    """

    if segment_length < 1:
        raise ValueError("segment_length must be at least one sample.")
    if segment_stride < 1:
        raise ValueError("segment_stride must be at least one sample.")

    requested_segment_duration_seconds = segment_length * time_step
    lowest_frequency_resolution_hz = 1.0 / requested_segment_duration_seconds
    minimum_segment_length_for_median_mean = segment_length + segment_stride
    minimum_segment_duration_for_median_mean_seconds = minimum_segment_length_for_median_mean * time_step
    welch_segment_settings: list[dict[str, Any]] = []

    for arc_idx, propagation_arc in enumerate(
        get_contiguous_valid_propagation_segments(valid_propagation_segments_mask),
        start=1,
    ):
        propagation_arc_length = int(propagation_arc.size)
        propagation_arc_duration_seconds = propagation_arc_length * time_step

        if propagation_arc_length < minimum_segment_length_for_median_mean:
            if emit_warnings:
                attainable_lowest_frequency_resolution_hz = 1.0 / propagation_arc_duration_seconds
                start_index = int(propagation_arc[0])
                end_index = int(propagation_arc[-1])
                # print(
                #     "\nWARNING: "
                #     f"{warning_context} will skip propagation segment {arc_idx} "
                #     f"(sample indices {start_index}-{end_index}, {propagation_arc_length} samples, "
                #     f"{propagation_arc_duration_seconds:.1f} s) because it is shorter than the required "
                #     f"required segment duration for the median-mean estimation of {minimum_segment_duration_for_median_mean_seconds:.1f} s "
                #     f"({segment_length} samples plus a stride of {segment_stride} samples at dt={time_step:.1f} s, "
                #     f"lowest frequency resolution={lowest_frequency_resolution_hz:.6e} Hz). "
                #     f"This propagation segment can support at most {propagation_arc_duration_seconds:.1f} s, "
                #     f"which corresponds to an attainable lowest frequency resolution of "
                #     f"{attainable_lowest_frequency_resolution_hz:.6e} Hz.\n"
                # )
            continue

        welch_segment_settings.append(
            {
                "arc_idx": arc_idx,
                "propagation_arc": propagation_arc,
                "segment_length": segment_length,
                "segment_stride": segment_stride,
                "segment_duration_seconds": requested_segment_duration_seconds,
                "lowest_frequency_resolution_hz": lowest_frequency_resolution_hz,
            }
        )

    return welch_segment_settings


def apply_orbital_phasing_period_exclusion_mask(
    values: np.ndarray,
    exclusion_mask: np.ndarray,
) -> np.ndarray:
    """Replace excluded samples in a value array with NaN.

    Args:
        values (np.ndarray): Array of values, first axis indexed by sample.
        exclusion_mask (np.ndarray): Boolean mask, True where the sample
            should be replaced with NaN.

    Returns:
        np.ndarray: A copy of values (as float) with excluded samples set to
            NaN.

    Raises:
        ValueError: If values and exclusion_mask have a different number of
            samples.
    """

    values = np.asarray(values, dtype=float).copy()
    exclusion_mask = np.asarray(exclusion_mask, dtype=bool).reshape(-1)

    if values.shape[0] != exclusion_mask.shape[0]:
        raise ValueError("values and exclusion_mask must have the same number of samples.")

    values[exclusion_mask] = np.nan
    return values


def get_welch_segment_settings_from_propagation_segments(
    valid_propagation_segments_mask: np.ndarray,
    time_step: float,
    target_lowest_frequency_resolution_hz: float = LOWEST_FREQUENCY_RESOLUTION_HZ,
    warning_context: str = "Welch estimation",
) -> list[dict[str, Any]]:
    """Build per-arc Welch estimation settings for a requested lowest-frequency resolution, with a 50% overlap stride.

    Args:
        valid_propagation_segments_mask (np.ndarray): Boolean mask over
            samples, True where usable.
        time_step (float): Sample spacing, in seconds.
        target_lowest_frequency_resolution_hz (float): Desired lowest
            resolvable frequency [Hz]. Defaults to LOWEST_FREQUENCY_RESOLUTION_HZ.
        warning_context (str): Label describing the calling context, used in
            diagnostic messages. Defaults to "Welch estimation".

    Returns:
        list[dict[str, Any]]: Per-arc Welch estimation segment settings, as returned by
            build_welch_estimation_segment_settings.

    Raises:
        ValueError: If no valid propagation arc is long enough to support the
            requested resolution.
    """

    segment_length, requested_segment_duration_seconds = get_segment_length_from_frequency_resolution(
        time_step=time_step,
        lowest_frequency_resolution_hz=target_lowest_frequency_resolution_hz,
    )
    segment_stride = segment_length // 2
    welch_segment_settings = build_welch_estimation_segment_settings(
        valid_propagation_segments_mask=valid_propagation_segments_mask,
        time_step=time_step,
        segment_length=segment_length,
        segment_stride=segment_stride,
        emit_warnings=True,
        warning_context=warning_context,
    )

    if not welch_segment_settings:
        raise ValueError(
            "No valid propagation segment is long enough for the requested Welch estimation settings: "
            f"T_segment={requested_segment_duration_seconds:.1f} s, "
            f"lowest_frequency_resolution={target_lowest_frequency_resolution_hz:.6e} Hz."
        )

    return welch_segment_settings


def compute_weighted_average_psd_across_segments(
    values: np.ndarray,
    valid_propagation_segments_mask: np.ndarray,
    time_step: float,
    segment_length: int | None = None,
    segment_stride: int | None = None,
    welch_segment_settings: list[dict[str, Any]] | None = None,
    emit_warnings: bool = False,
    warning_context: str = "Welch estimation",
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a Welch PSD independently per propagation arc, then combine with sample-count weighting.

    Each contiguous valid propagation arc is Welch-PSD-estimated separately
    (so no stencil or segment spans an excluded maneuver gap), then the
    per-arc PSDs are averaged, weighted by each arc's sample count, requiring
    all arcs to share the same frequency grid.

    Args:
        values (np.ndarray): 1D signal to estimate the PSD of.
        valid_propagation_segments_mask (np.ndarray): Boolean mask over
            samples, True where usable.
        time_step (float): Sample spacing, in seconds.
        segment_length (int | None): Welch estimation segment length, in samples. Must be
            given together with segment_stride if welch_segment_settings is
            not provided.
        segment_stride (int | None): Welch estimationsegment stride, in samples.
        welch_segment_settings (list[dict[str, Any]] | None): Precomputed
            per-arc settings, as an alternative to segment_length/segment_stride.
        emit_warnings (bool): Passed through to build_welch_estimation_segment_settings
            if segment_length/segment_stride path is used. Defaults to False.
        warning_context (str): Label for diagnostic messages. Defaults to
            "Welch estimation".

    Returns:
        tuple[np.ndarray, np.ndarray]: (frequency_grid, weighted_average_psd),
            the shared frequency grid and the sample-count-weighted average
            PSD across all usable arcs.

    Raises:
        ValueError: If values and valid_propagation_segments_mask have
            different shapes, if neither or both of
            (welch_segment_settings) and (segment_length, segment_stride) are
            provided, or if no valid arc supports the requested settings.
        RuntimeError: If different arcs' Welch PSD estimates land on different
            frequency grids (the segment settings should prevent this).
    """

    values = np.asarray(values, dtype=float).reshape(-1)
    valid_propagation_segments_mask = np.asarray(valid_propagation_segments_mask, dtype=bool).reshape(-1)

    if values.shape != valid_propagation_segments_mask.shape:
        raise ValueError("values and valid_propagation_segments_mask must have the same shape.")

    if welch_segment_settings is None:
        if segment_length is None or segment_stride is None:
            raise ValueError(
                "Either welch_segment_settings must be provided, or both segment_length and segment_stride must be set."
            )
        welch_segment_settings = build_welch_estimation_segment_settings(
            valid_propagation_segments_mask=valid_propagation_segments_mask,
            time_step=time_step,
            segment_length=segment_length,
            segment_stride=segment_stride,
            emit_warnings=emit_warnings,
            warning_context=warning_context,
        )
    elif segment_length is not None or segment_stride is not None:
        raise ValueError("Provide either welch_segment_settings or segment_length/segment_stride, not both.")

    weighted_psd_sum = None
    frequency_grid = None
    total_weight = 0.0

    for welch_segment_setting in welch_segment_settings:
        propagation_arc = np.asarray(welch_segment_setting["propagation_arc"], dtype=int)
        segment_length = int(welch_segment_setting["segment_length"])
        segment_stride = int(welch_segment_setting["segment_stride"])

        propagation_arc_time_series = types.timeseries.TimeSeries(
            values[propagation_arc],
            delta_t=time_step,
        )
        estimated_psd = psd.welch(
            propagation_arc_time_series,
            seg_len=segment_length,
            seg_stride=segment_stride,
            avg_method="median-mean",
        )

        propagation_arc_frequencies = estimated_psd.sample_frequencies.numpy()
        propagation_arc_psd_values = estimated_psd.numpy()
        propagation_arc_weight = float(propagation_arc.size)

        if frequency_grid is None:
            frequency_grid = propagation_arc_frequencies
            weighted_psd_sum = propagation_arc_weight * propagation_arc_psd_values
        else:
            if not np.allclose(frequency_grid, propagation_arc_frequencies):
                raise RuntimeError(
                    "Welch estimated PSDs frequency grids differ across propagation arcs. "
                    "Ensure the selected propagation arcs share the same segment duration."
                )
            weighted_psd_sum += propagation_arc_weight * propagation_arc_psd_values

        total_weight += propagation_arc_weight

    if frequency_grid is None or weighted_psd_sum is None or total_weight == 0.0:
        raise ValueError("No valid propagation arc is long enough for the requested Welch settings.")

    return frequency_grid, weighted_psd_sum / total_weight


def compute_log_frequency_binned_psd(
    frequencies: np.ndarray,
    psd_values: np.ndarray,
    bin_width_decades: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a PSD by averaging within uniform log10-frequency bins.

    Args:
        frequencies (np.ndarray): 1D frequency array [Hz].
        psd_values (np.ndarray): 1D PSD values matching frequencies.
        bin_width_decades (float): Bin width in log10-frequency decades.
            Defaults to 0.05.

    Returns:
        tuple[np.ndarray, np.ndarray]: (smoothed_frequencies, smoothed_psd),
            each entry being the mean frequency/PSD value within one populated
            log-frequency bin, in ascending frequency order.

    Raises:
        ValueError: If frequencies and psd_values have different shapes,
            bin_width_decades is not positive, or no finite positive-frequency
            sample is available.
        RuntimeError: If binning produces no populated bins (should not occur
            given the input validation above).
    """

    frequencies = np.asarray(frequencies, dtype=float).reshape(-1)
    psd_values = np.asarray(psd_values, dtype=float).reshape(-1)

    if frequencies.shape != psd_values.shape:
        raise ValueError("frequencies and the corresponding PSD values must have the same shape.")
    if bin_width_decades <= 0.0:
        raise ValueError("The bin width in decades must be positive.")

    # Welch PSD estimates include the 0 Hz bin, which cannot be represented in log10-frequency space.
    valid_samples_mask = np.isfinite(frequencies) & np.isfinite(psd_values) & (frequencies > 0.0)
    if not np.any(valid_samples_mask):
        raise ValueError(
            "Log-frequency PSD binning requires at least one finite sample with positive frequency."
        )

    frequencies = frequencies[valid_samples_mask]
    psd_values = psd_values[valid_samples_mask]

    sort_indices = np.argsort(frequencies)
    sorted_frequencies_input = frequencies[sort_indices]
    sorted_psd_input = psd_values[sort_indices]

    log_frequencies = np.log10(sorted_frequencies_input)
    if np.isclose(log_frequencies[0], log_frequencies[-1]):
        return (
            np.array([sorted_frequencies_input[0]], dtype=float),
            np.array([sorted_psd_input[0]], dtype=float),
        )

    log_frequency_bin_edges = np.arange(
        log_frequencies[0],
        log_frequencies[-1] + bin_width_decades,
        bin_width_decades,
    )
    if log_frequency_bin_edges.size < 2:
        log_frequency_bin_edges = np.array(
            [log_frequencies[0], log_frequencies[-1]],
            dtype=float,
        )

    smoothed_frequencies: list[float] = []
    smoothed_psd_values: list[float] = []

    for bin_idx, (bin_lower_edge, bin_upper_edge) in enumerate(
        zip(log_frequency_bin_edges[:-1], log_frequency_bin_edges[1:])
    ):
        if bin_idx == log_frequency_bin_edges.size - 2:
            samples_in_bin_mask = (
                (log_frequencies >= bin_lower_edge)
                & (log_frequencies <= bin_upper_edge)
            )
        else:
            samples_in_bin_mask = (
                (log_frequencies >= bin_lower_edge)
                & (log_frequencies < bin_upper_edge)
            )

        if not np.any(samples_in_bin_mask):
            continue

        frequencies_in_bin = sorted_frequencies_input[samples_in_bin_mask]
        psd_values_in_bin = sorted_psd_input[samples_in_bin_mask]

        smoothed_frequencies.append(
            float(np.mean(frequencies_in_bin))
        )
        smoothed_psd_values.append(float(np.mean(psd_values_in_bin)))

    if not smoothed_frequencies:
        raise RuntimeError("The log-frequency PSD binning produced no populated bins.")

    return (
        np.asarray(smoothed_frequencies, dtype=float),
        np.asarray(smoothed_psd_values, dtype=float),
    )


def compute_root_mean_square_over_finite_values(values: np.ndarray) -> float:
    """Compute the root-mean-square of an array's finite entries, ignoring NaN/inf.

    Args:
        values (np.ndarray): Array of values, potentially containing NaN
            entries.

    Returns:
        float: RMS computed over only the finite entries of values.
    """

    values = np.asarray(values, dtype=float).reshape(-1)
    finite_mask = np.isfinite(values)

    return float(np.sqrt(np.mean(values[finite_mask] ** 2)))


def differentiate_inter_satellite_range_piecewise(
    ranges: np.ndarray,
    time: np.ndarray,
    accuracy: int,
    derivative_order: int,
    valid_propagation_segments_mask: np.ndarray,
) -> np.ndarray:
    """Differentiate a range history arc-by-arc so no finite-difference stencil spans an excluded gap.

    Applies finite differentiation independently within each
    contiguous run of valid samples, leaving excluded samples as
    NaN in the output.

    Args:
        ranges (np.ndarray): 1D inter-satellite range history [m].
        time (np.ndarray): 1D time values matching ranges.
        accuracy (int): Finite-difference accuracy order.
        derivative_order (int): 1 for range-rate, 2 for range-acceleration.
        valid_propagation_segments_mask (np.ndarray): Boolean mask over
            samples, True where usable.

    Returns:
        np.ndarray: Differentiated range history, same shape as ranges, with
            NaN at excluded samples.

    Raises:
        ValueError: If ranges, time, and valid_propagation_segments_mask do
            not all share the same shape.
    """

    ranges = np.asarray(ranges, dtype=float).reshape(-1)
    time = np.asarray(time, dtype=float).reshape(-1)
    valid_propagation_segments_mask = np.asarray(valid_propagation_segments_mask, dtype=bool).reshape(-1)

    if not (ranges.shape == time.shape == valid_propagation_segments_mask.shape):
        raise ValueError("ranges, time, and valid_propagation_segments_mask must have the same shape.")

    differentiated_ranges = np.full(ranges.shape, np.nan, dtype=float)

    for segment_indices in get_contiguous_valid_propagation_segments(valid_propagation_segments_mask):

        differentiated_ranges[segment_indices] = differentiate_inter_satellite_range(
            ranges=ranges[segment_indices],
            time=time[segment_indices],
            accuracy=accuracy,
            derivative_order=derivative_order,
        )

    return differentiated_ranges


def compute_log_root_mean_square_misfit(
    frequencies: np.ndarray,
    numerical_asd: np.ndarray,
    analytical_asd: np.ndarray,
    minimum_frequency_hz: float = 1e-5,
) -> float:
    """Compute the RMS log10-space misfit between numerical and analytical ASD curves.

    Args:
        frequencies (np.ndarray): 1D frequency array [Hz], matching
            numerical_asd and analytical_asd in shape.
        numerical_asd (np.ndarray): ASD estimated via Welch PSD estimation.
        analytical_asd (np.ndarray): ASD predicted analytically (by
            scaling a reference ASD by (2*pi*f)^derivative_order).
        minimum_frequency_hz (float): Frequencies at or below this value are
            excluded from the misfit, to avoid the ill-defined low-frequency/DC
            region. Defaults to 1e-5.

    Returns:
        float: RMS of log10(numerical_asd) - log10(analytical_asd) over the
            frequency band where both ASDs are nonzero and above
            minimum_frequency_hz.
    """

    frequencies = np.asarray(frequencies, dtype=float).reshape(-1)
    numerical_asd = np.asarray(numerical_asd, dtype=float).reshape(-1)
    analytical_asd = np.asarray(analytical_asd, dtype=float).reshape(-1)

    valid_frequency_band_mask = (
        (numerical_asd != 0.0) 
        & (analytical_asd != 0.0)
        & (frequencies > minimum_frequency_hz)
    )

    log_difference = (
        np.log10(numerical_asd[valid_frequency_band_mask])
        - np.log10(analytical_asd[valid_frequency_band_mask])
    )
    return float(np.sqrt(np.mean(log_difference**2)))


def run_psd_estimation_parameter_sensitivity_analysis(
    range_noise_debiased: np.ndarray,
    differentiated_range_noise: np.ndarray,
    valid_propagation_segments_mask: np.ndarray,
    time_step: float,
    derivative_order: int,
    num_trials: int,
    seed: int,
    minimum_log_rms_misfit_frequency_hz: float = 1e-5,
) -> dict[str, Any]:
    """Search Welch estimationsegment-length/stride settings for the best analytical-vs-numerical ASD fit.

    Randomly samples (segment_length, segment_stride) pairs supported by the
    available propagation arcs, estimates the Welch ASD of both
    range_noise_debiased and differentiated_range_noise for each candidate,
    and scores each candidate by how closely
    (2*pi*f)^derivative_order * ASD(range_noise_debiased) matches the directly
    estimated ASD(differentiated_range_noise) in log-space. The candidate with
    the smallest such misfit is returned as the best fit.

    Args:
        range_noise_debiased (np.ndarray): 1D debiased KBR range-noise history.
        differentiated_range_noise (np.ndarray): 1D range-noise history already
            differentiated to derivative_order (rate or acceleration).
        valid_propagation_segments_mask (np.ndarray): Boolean mask over
            samples, True where usable.
        time_step (float): Sample spacing, in seconds.
        derivative_order (int): 1 for range-rate, 2 for range-acceleration;
            used in the analytical ASD scaling law.
        num_trials (int): Target number of distinct (segment_length,
            segment_stride) candidates to evaluate.
        seed (int): Seed for the random candidate sampler, for reproducibility.
        minimum_log_rms_misfit_frequency_hz (float): Passed to
            compute_log_root_mean_square_misfit as its low-frequency cutoff.
            Defaults to 1e-5.

    Returns:
        dict[str, Any]: A dict with keys:
            - "records" (list[dict[str, float]]): One entry per evaluated
              candidate, with segment_length, segment_stride,
              segment_duration_seconds, lowest_frequency_resolution_hz,
              number_of_used_propagation_segments, and log_rms_misfit.
            - "best_record" (dict[str, Any]): The lowest-misfit candidate's
              record, plus "frequencies", "range_noise_debiased_asd",
              "numerical_differentiated_asd", and "analytical_differentiated_asd"
              arrays.

    Raises:
        ValueError: If the three input arrays have mismatched shapes, if no
            valid propagation segment exists, or if no segment is long enough
            to support the minimum required Welch estimation settings.
        RuntimeError: If no candidate produced a valid result.
    """

    range_noise_debiased = np.asarray(range_noise_debiased, dtype=float).reshape(-1)
    differentiated_range_noise = np.asarray(differentiated_range_noise, dtype=float).reshape(-1)
    valid_propagation_segments_mask = np.asarray(valid_propagation_segments_mask, dtype=bool).reshape(-1)

    if not (
        range_noise_debiased.shape
        == differentiated_range_noise.shape
        == valid_propagation_segments_mask.shape
    ):
        raise ValueError(
            "range_noise_debiased, differentiated_range_noise, and valid_propagation_segments_mask must have the same shape."
        )

    valid_propagation_segments = get_contiguous_valid_propagation_segments(valid_propagation_segments_mask)
    if not valid_propagation_segments:
        raise ValueError("At least one valid propagation segment is required for the PSD sensitivity analysis.")

    minimum_segment_length_from_resolution, _ = get_segment_length_from_frequency_resolution(
        time_step=time_step,
        lowest_frequency_resolution_hz=LOWEST_FREQUENCY_RESOLUTION_HZ,
    )
    longest_valid_segment_length = max(
        propagation_segment.size
        for propagation_segment in valid_propagation_segments
    )
    if longest_valid_segment_length < minimum_segment_length_from_resolution:
        raise ValueError(
            "No valid propagation segment is long enough to support the minimum Welch segment duration "
            f"required by the target lowest frequency resolution of "
            f"{LOWEST_FREQUENCY_RESOLUTION_HZ:.6e} Hz."
        )

    min_segment_length = minimum_segment_length_from_resolution
    min_segment_stride = min_segment_length // 2
    minimum_segment_length_for_median_mean = min_segment_length + min_segment_stride
    if longest_valid_segment_length < minimum_segment_length_for_median_mean:
        raise ValueError(
            "No valid propagation segment is long enough to support Welch estimation median-mean averaging "
            "at the requested lowest frequency resolution. "
            f"The longest valid segment has {longest_valid_segment_length} samples, but at least "
            f"{minimum_segment_length_for_median_mean} samples are required "
            f"({min_segment_length} segment samples plus {min_segment_stride} stride samples)."
        )

    max_segment_length = longest_valid_segment_length
    starting_segment_length = min_segment_length
    starting_segment_stride = min_segment_stride

    rng = np.random.default_rng(seed)
    candidate_parameters_pairs = {(starting_segment_length, starting_segment_stride)}

    # Cap the number of sampling attempts so this loop cannot spin forever if
    # the valid arcs are too short to yield num_trials distinct, supportable
    # (segment_length, segment_stride) pairs.
    max_sampling_attempts =  num_trials * 100
    sampling_attempts = 0
    while len(candidate_parameters_pairs) < num_trials:
        sampling_attempts += 1
        if sampling_attempts > max_sampling_attempts:
            break

        segment_length = int(rng.integers(min_segment_length, max_segment_length + 1))
        # Stride between 50% and 100% of segment_length keeps segment overlap
        # within pycbc's supported range for "median-mean" Welch estimation method's averaging.
        segment_stride = int(rng.integers(segment_length // 2, segment_length + 1))

        if not any(
            propagation_segment.size >= segment_length + segment_stride
            for propagation_segment in valid_propagation_segments
        ):
            continue

        candidate_parameters_pairs.add((segment_length, segment_stride))

    records: list[dict[str, float]] = []
    best_candidate: dict[str, Any] | None = None

    for segment_length, segment_stride in sorted(candidate_parameters_pairs):
        candidate_welch_segment_settings = build_welch_estimation_segment_settings(
            valid_propagation_segments_mask=valid_propagation_segments_mask,
            time_step=time_step,
            segment_length=segment_length,
            segment_stride=segment_stride,
            emit_warnings=True,
        )
        estimated_frequencies_asd_range_noise_debiased, estimated_psd_range_noise_debiased = (
            compute_weighted_average_psd_across_segments(
                values=range_noise_debiased,
                valid_propagation_segments_mask=valid_propagation_segments_mask,
                time_step=time_step,
                welch_segment_settings=candidate_welch_segment_settings,
            )
        )
        estimated_values_asd_range_noise_debiased = np.sqrt(estimated_psd_range_noise_debiased)

        estimated_frequencies_asd_differentiated_range_noise, estimated_psd_differentiated_range_noise = (
            compute_weighted_average_psd_across_segments(
                values=differentiated_range_noise,
                valid_propagation_segments_mask=valid_propagation_segments_mask,
                time_step=time_step,
                welch_segment_settings=candidate_welch_segment_settings,
            )
        )
        estimated_values_asd_differentiated_range_noise = np.sqrt(estimated_psd_differentiated_range_noise)

        if not np.allclose(estimated_frequencies_asd_range_noise_debiased, estimated_frequencies_asd_differentiated_range_noise):
            raise RuntimeError("Welch estimated PSDs frequency grids differ for the same segment settings.")

        analytical_asd_differentiation = (2.0 * np.pi * estimated_frequencies_asd_range_noise_debiased) ** derivative_order * estimated_values_asd_range_noise_debiased
        log_root_mean_square_misfit = compute_log_root_mean_square_misfit(
            estimated_frequencies_asd_range_noise_debiased,
            estimated_values_asd_differentiated_range_noise,
            analytical_asd_differentiation,
            minimum_frequency_hz=minimum_log_rms_misfit_frequency_hz,
        )

        result = {
            "segment_length": float(segment_length),
            "segment_stride": float(segment_stride),
            "segment_duration_seconds": float(segment_length * time_step),
            "lowest_frequency_resolution_hz": float(1.0 / (segment_length * time_step)),
            "number_of_used_propagation_segments": float(len(candidate_welch_segment_settings)),
            "log_rms_misfit": log_root_mean_square_misfit,
        }
        records.append(result)

        if best_candidate is None or log_root_mean_square_misfit < best_candidate["log_rms_misfit"]:
            best_candidate = {
                **result,
                "frequencies": estimated_frequencies_asd_range_noise_debiased,
                "range_noise_debiased_asd": estimated_values_asd_range_noise_debiased,
                "numerical_differentiated_asd": estimated_values_asd_differentiated_range_noise,
                "analytical_differentiated_asd": analytical_asd_differentiation,
            }

    if best_candidate is None:
        raise RuntimeError("The PSD estimation parameter sensitivity analysis did not produce any valid candidate.")

    return {
        "records": records,
        "best_record": best_candidate,
    }


def propagate_observation_errors_to_lgds(
    time_step: float,   
    kbr_range_noise: np.ndarray,
    kbr_range_noise_debiased: np.ndarray,
    eci_velocity_data: list[np.ndarray],
    eci_gps_velocity_noise: dict[str, np.ndarray],
    eci_position_data: list[np.ndarray],
    eci_gps_position_noise: dict[str, np.ndarray],
    accelerometer_observations_sf: dict[str, np.ndarray],
    dependent_variables_array: np.ndarray,
    noisy_attitude_time_series: dict[str, dict[str, types.TimeSeries]],
    guidance_log: list[dict[str, Any]] | None,
    plotter: Plotter,
    noise_model_version: int,
    reference_orbital_period: float | None = None,
    accuracy_orders: list[int] | None = None,
    sensitivity_analysis_runs: int = 10000,
    sensitivity_analysis_seed: int = 42,
) -> None:
    """Propagate KBR range and related noise into a line-of-sight gravity difference (LGD) error budget.

    End-to-end pipeline: 
     - excludes samples within any orbit-phasing maneuver window
     - estimates the KBR debiased range-noise ASD via Welch's method
     - numerically differentiates the noisy range history to
       range-rate/range-acceleration at several finite-difference accuracy
       orders, picking the accuracy with lowest RMS error against the reference
       (error-free) range-rate/acceleration derived from dependent_variables_array,
     - cross-checks the differentiated range-noise ASD against an analytical
       (2*pi*f)^n-scaled prediction via a Welch estimation parameter sensitivity search, and
     - reconstructs the noisy and error-free line-of-sight gravity difference
       (LGD) using GPS position/velocity noise, accelerometer observations, and
       noisy attitude, then estimates and saves the resulting LGD error ASD.


    Args:
        time_step (float): Sample spacing, in seconds.
        kbr_range_noise (np.ndarray): Noisy (biased) KBR range-noise history [m].
        kbr_range_noise_debiased (np.ndarray): Debiased KBR range-noise history [m].
        eci_velocity_data (list[np.ndarray]): [target_velocity, chaser_velocity],
            each an (N, 3) ECI velocity history [m/s].
        eci_gps_velocity_noise (dict[str, np.ndarray]): Per-satellite ("GRACE C",
            "GRACE D") GPS-derived velocity noise histories.
        eci_position_data (list[np.ndarray]): [target_position, chaser_position],
            each an (N, 3) ECI position history [m].
        eci_gps_position_noise (dict[str, np.ndarray]): Per-satellite GPS-derived
            position noise histories.
        accelerometer_observations_sf (dict[str, np.ndarray]): Per-satellite
            accelerometer observations in the satellite frame.
        dependent_variables_array (np.ndarray): Dependent-variable history from
            propagation.
        noisy_attitude_time_series (dict[str, dict[str, types.TimeSeries]]):
            Per-satellite noisy yaw/pitch/roll pointing-angle time series.
        guidance_log (list[dict[str, Any]] | None): Orbit-phasing maneuver
            records, used to exclude affected samples.
        plotter (Plotter): Plotter instance used to save diagnostic plots.
        noise_model_version (int): Noise model version (1 or 2), used when
            saving spectra to the versioned output directory.
        reference_orbital_period (float | None): Mean orbital period, in
            seconds, used only to annotate plots with an orbital-period
            reference line. Optional.
        accuracy_orders (list[int] | None): Finite-difference accuracy orders
            to evaluate; defaults to [2, 4, ..., 20] if not given.
        sensitivity_analysis_runs (int): Number of Welch estimation parameter candidates
            to evaluate in run_psd_estimation_parameter_sensitivity_analysis.
            Defaults to 10000.
        sensitivity_analysis_seed (int): Base seed for the sensitivity
            analysis; the second (range-acceleration) analysis uses
            sensitivity_analysis_seed + 1. Defaults to 42.

    Returns:
        None

    Raises:
        ValueError: If eci_position_data/eci_velocity_data do not each contain
            exactly two histories, if no maneuver-free samples are available,
            or if reference_orbital_period is provided but not positive.
        KeyError: If eci_gps_position_noise, eci_gps_velocity_noise, or
            accelerometer_observations_sf is missing an entry for "GRACE C" or
            "GRACE D".
    """

    if accuracy_orders is None:
        accuracy_orders = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    time = dependent_variables_array[:, 0]
    orbital_phasing_period_exclusion_mask = build_orbital_phasing_period_exclusion_mask(time, guidance_log)

    kbr_range_noise = np.asarray(kbr_range_noise, dtype=float).reshape(-1)
    kbr_range_noise_debiased = np.asarray(kbr_range_noise_debiased, dtype=float).reshape(-1)
    dependent_variables_array = np.asarray(dependent_variables_array, dtype=float)

    if len(eci_position_data) != 2 or len(eci_velocity_data) != 2:
        raise ValueError("Exactly two position histories and two velocity histories are required.")

    valid_propagation_segments_mask = ~orbital_phasing_period_exclusion_mask

    if not np.any(valid_propagation_segments_mask):
        raise ValueError("No maneuver-free propagation samples are available for analysis.")

    if reference_orbital_period is not None and reference_orbital_period <= 0.0:
        raise ValueError("reference_orbital_period must be positive when provided.")

    target_position = apply_orbital_phasing_period_exclusion_mask(
        eci_position_data[0],
        orbital_phasing_period_exclusion_mask,
    )
    chaser_position = apply_orbital_phasing_period_exclusion_mask(
        eci_position_data[1],
        orbital_phasing_period_exclusion_mask,
    )
    target_velocity = apply_orbital_phasing_period_exclusion_mask(
        eci_velocity_data[0],
        orbital_phasing_period_exclusion_mask,
    )
    chaser_velocity = apply_orbital_phasing_period_exclusion_mask(
        eci_velocity_data[1],
        orbital_phasing_period_exclusion_mask,
    )

    relative_position = target_position - chaser_position
    inter_satellite_range = np.linalg.norm(relative_position, axis=1)

    noisy_inter_satellite_range = inter_satellite_range + kbr_range_noise
    noisy_inter_satellite_range_debiased = inter_satellite_range + kbr_range_noise_debiased

    los_unit_vector = relative_position / inter_satellite_range[:, None]
    relative_velocity = target_velocity - chaser_velocity
    reference_range_rate = np.einsum("ij,ij->i", los_unit_vector, relative_velocity)

    # NOTE: Column indices below are tied to the fixed order of
    # dependent_variables_to_save in main.py: the columns referenced to below hold
    # GRACE C's and GRACE D's total_acceleration entries respectively.
    target_total_acceleration = apply_orbital_phasing_period_exclusion_mask(
        dependent_variables_array[:, 47:50],
        orbital_phasing_period_exclusion_mask,
    )
    chaser_total_acceleration = apply_orbital_phasing_period_exclusion_mask(
        dependent_variables_array[:, 50:53],
        orbital_phasing_period_exclusion_mask,
    )
    relative_total_acceleration = target_total_acceleration - chaser_total_acceleration
    error_free_los_relative_total_acceleration = np.einsum(
        "ij,ij->i",
        los_unit_vector,
        relative_total_acceleration,
    )

    centrifugal_term = (
        np.einsum("ij,ij->i", relative_velocity, relative_velocity) - reference_range_rate**2
    ) / inter_satellite_range
    reference_range_acceleration = error_free_los_relative_total_acceleration + centrifugal_term

    # ==========================================================
    # KBR range-noise (debiased) ASD via Welch estimation
    # ==========================================================

    kbr_debiased_range_noise_welch_segment_settings = get_welch_segment_settings_from_propagation_segments(
        valid_propagation_segments_mask=valid_propagation_segments_mask,
        time_step=time_step,
        target_lowest_frequency_resolution_hz=LOWEST_FREQUENCY_RESOLUTION_HZ,
        warning_context="KBR debiased range-noise Welch ASD estimation",
    )
    estimated_frequencies_asd_kbr_range_noise_debiased, estimated_values_psd_kbr_range_noise_debiased = (
        compute_weighted_average_psd_across_segments(
            values=kbr_range_noise_debiased,
            valid_propagation_segments_mask=valid_propagation_segments_mask,
            time_step=time_step,
            welch_segment_settings=kbr_debiased_range_noise_welch_segment_settings,
        )
    )
    estimated_values_asd_kbr_range_noise_debiased = np.sqrt(estimated_values_psd_kbr_range_noise_debiased)
    save_to_hdf5(
        noise_model_version=noise_model_version,
        file_name="range_error_debiased_asd.h5",
        frequencies=estimated_frequencies_asd_kbr_range_noise_debiased,
        asd_values=estimated_values_asd_kbr_range_noise_debiased,
        spectrum_name="KBR range error debiased ASD",
        ordinate_label="m Hz^(-1/2)",
    )
    smoothed_frequencies_asd_kbr_range_noise_debiased, smoothed_values_psd_kbr_range_noise_debiased = (
        compute_log_frequency_binned_psd(
            frequencies=estimated_frequencies_asd_kbr_range_noise_debiased,
            psd_values=estimated_values_psd_kbr_range_noise_debiased,
        )
    )
    smoothed_values_asd_kbr_range_noise_debiased = np.sqrt(smoothed_values_psd_kbr_range_noise_debiased)

    plotter.plot_welch_estimated_asd(
        frequencies=estimated_frequencies_asd_kbr_range_noise_debiased,
        asd_values=estimated_values_asd_kbr_range_noise_debiased,
        smoothed_frequencies=smoothed_frequencies_asd_kbr_range_noise_debiased,
        smoothed_asd_values=smoothed_values_asd_kbr_range_noise_debiased,
        file_name="range_noise_debiased_welch_estimated_asd.png",
        ordinate_label=r"ASD [m Hz$^{-1/2}$]",
        title="KBR Range Noise (Debiased) ASD",
        line_label="Welch Estimated ASD",
        x_limit_inf=1e-5,
        x_limit_sup=1e-1,
        reference_orbital_period_seconds=reference_orbital_period,
    )

    # ==========================================================
    # First derivative accuracy assessment
    # ==========================================================

    first_derivative_results: dict[int, dict[str, Any]] = {}

    for accuracy in accuracy_orders:
        numerical_range_rate = differentiate_inter_satellite_range_piecewise(
            ranges=noisy_inter_satellite_range,
            time=time,
            accuracy=accuracy,
            derivative_order=1,
            valid_propagation_segments_mask=valid_propagation_segments_mask,
        )
        error = numerical_range_rate - reference_range_rate

        first_derivative_results[accuracy] = {
            "numerical_range_rate": numerical_range_rate,
            "error": error,
            "absolute_error": np.abs(error),
            "error_rms": compute_root_mean_square_over_finite_values(error),
        }

    best_candidate_first_derivative_accuracy = min(
        first_derivative_results,
        key=lambda accuracy: first_derivative_results[accuracy]["error_rms"],
    )

    plotter.plot_numerical_derivative_statistics(
        scenario="GRACE-FO",
        time=time,
        results=first_derivative_results,
        file_name="grace_fo_range_rate_finite_difference_error.png",
        ylabel=r"$|\epsilon_{\dot{\rho}}(t)|$ [m/s]",
        title="Noisy Inter-Satellite Range-Rate Error",
        rms_unit_label=r"\mathrm{m\,s^{-1}}",
    )

    print("=================================")
    print("First-derivative accuracy assessment")
    for accuracy in accuracy_orders:
        print(
            f"  p={accuracy:2d}: RMS error = "
            f"{first_derivative_results[accuracy]['error_rms']:.6e} m/s"
        )
    print(f"Selected first-derivative accuracy order: p={best_candidate_first_derivative_accuracy}")
    print("=================================\n")

    # ==========================================================
    # Second derivative accuracy assessment
    # ==========================================================

    second_derivative_results: dict[int, dict[str, Any]] = {}

    for accuracy in accuracy_orders:
        numerical_range_acceleration = differentiate_inter_satellite_range_piecewise(
            ranges=noisy_inter_satellite_range,
            time=time,
            accuracy=accuracy,
            derivative_order=2,
            valid_propagation_segments_mask=valid_propagation_segments_mask,
        )
        error = numerical_range_acceleration - reference_range_acceleration

        second_derivative_results[accuracy] = {
            "numerical_range_acceleration": numerical_range_acceleration,
            "error": error,
            "absolute_error": np.abs(error),
            "error_rms": compute_root_mean_square_over_finite_values(error),
        }

    best_candidate_second_derivative_accuracy = min(
        second_derivative_results,
        key=lambda accuracy: second_derivative_results[accuracy]["error_rms"],
    )

    best_candidate_numerical_range_rate = first_derivative_results[
        best_candidate_first_derivative_accuracy
    ]["numerical_range_rate"]
    best_candidate_numerical_range_acceleration = second_derivative_results[
        best_candidate_second_derivative_accuracy
    ]["numerical_range_acceleration"]

    plotter.plot_numerical_derivative_statistics(
        scenario="GRACE-FO",
        time=time,
        results=second_derivative_results,
        file_name="grace_fo_range_acceleration_finite_difference_error.png",
        ylabel=r"$|\epsilon_{\ddot{\rho}}(t)|$ [m/s$^2$]",
        title="Noisy Inter-Satellite Range-Acceleration Error",
        rms_unit_label=r"\mathrm{m\,s^{-2}}",
    )

    print("=================================")
    print("Second-derivative accuracy assessment")
    for accuracy in accuracy_orders:
        print(
            f"  p={accuracy:2d}: RMS error = "
            f"{second_derivative_results[accuracy]['error_rms']:.6e} m/s^2"
        )
    print(f"Selected second-derivative accuracy order: p={best_candidate_second_derivative_accuracy}")
    print("=================================\n")

    # ==========================================================
    # Numerical vs analytical differentiation of the range-noise ASD
    # ==========================================================

    numerical_range_rate_noise = differentiate_inter_satellite_range_piecewise(
        kbr_range_noise,
        time,
        best_candidate_first_derivative_accuracy,
        derivative_order=1,
        valid_propagation_segments_mask=valid_propagation_segments_mask,
    )
    numerical_range_acceleration_noise = differentiate_inter_satellite_range_piecewise(
        kbr_range_noise,
        time,
        best_candidate_second_derivative_accuracy,
        derivative_order=2,
        valid_propagation_segments_mask=valid_propagation_segments_mask,
    )

    minimum_log_rms_misfit_frequency_hz = 1e-5

    first_derivative_spectral_sensitivity_analysis = run_psd_estimation_parameter_sensitivity_analysis(
        range_noise_debiased=kbr_range_noise_debiased,
        differentiated_range_noise=numerical_range_rate_noise,
        valid_propagation_segments_mask=valid_propagation_segments_mask,
        time_step=time_step,
        derivative_order=1,
        num_trials=sensitivity_analysis_runs,
        seed=sensitivity_analysis_seed,
        minimum_log_rms_misfit_frequency_hz=minimum_log_rms_misfit_frequency_hz,
    )
    second_derivative_spectral_sensitivity_analysis = run_psd_estimation_parameter_sensitivity_analysis(
        range_noise_debiased=kbr_range_noise_debiased,
        differentiated_range_noise=numerical_range_acceleration_noise,
        valid_propagation_segments_mask=valid_propagation_segments_mask,
        time_step=time_step,
        derivative_order=2,
        num_trials=sensitivity_analysis_runs,
        seed=sensitivity_analysis_seed + 1,
        minimum_log_rms_misfit_frequency_hz=minimum_log_rms_misfit_frequency_hz,
    )

    save_to_hdf5(
        noise_model_version=noise_model_version,
        file_name="range_rate_error_analytical_differentiated_asd.h5",
        frequencies=first_derivative_spectral_sensitivity_analysis["best_record"]["frequencies"],
        asd_values=first_derivative_spectral_sensitivity_analysis["best_record"]["analytical_differentiated_asd"],
        spectrum_name="Range-rate error analytical differentiated ASD",
        ordinate_label="m s^(-1) Hz^(-1/2)",
    )
    save_to_hdf5(
        noise_model_version=noise_model_version,
        file_name="range_acceleration_error_analytical_differentiated_asd.h5",
        frequencies=second_derivative_spectral_sensitivity_analysis["best_record"]["frequencies"],
        asd_values=second_derivative_spectral_sensitivity_analysis["best_record"]["analytical_differentiated_asd"],
        spectrum_name="Range-acceleration error analytical differentiated ASD",
        ordinate_label="m s^(-2) Hz^(-1/2)",
    )

    plotter.plot_spectral_sensitivity_analysis_results(
        spectral_sensitivity_analysis=first_derivative_spectral_sensitivity_analysis["records"],
        file_name="grace_fo_range_rate_spectral_sensitivity_analysis.png",
        title="Sensitivity Analysis for Range-Rate ASD Fit",
        colorbar_label="Log-RMS ASD Misfit",
    )
    plotter.plot_spectral_sensitivity_analysis_results(
        spectral_sensitivity_analysis=second_derivative_spectral_sensitivity_analysis["records"],
        file_name="grace_fo_range_acceleration_spectral_sensitivity_analysis.png",
        title="Sensitivity Analysis for Range-Acceleration ASD Fit",
        colorbar_label="Log-RMS ASD Misfit",
    )

    plotter.plot_welch_estimated_asd_comparison(
        estimated_frequencies=first_derivative_spectral_sensitivity_analysis["best_record"]["frequencies"],
        estimated_asd_values=first_derivative_spectral_sensitivity_analysis["best_record"]["numerical_differentiated_asd"],
        reference_frequencies=first_derivative_spectral_sensitivity_analysis["best_record"]["frequencies"],
        reference_asd_values=first_derivative_spectral_sensitivity_analysis["best_record"]["analytical_differentiated_asd"],
        file_name="grace_fo_range_rate_noise_asd_comparison.png",
        ordinate_label=r"ASD [m s$^{-1}$ Hz$^{-1/2}$]",
        title="Range-Rate Noise ASD: Numerical vs Analytical Differentiation",
        estimated_label=r"Numerical",
        reference_label=r"Analytical",
        x_limit_inf=1e-5,
        x_limit_sup=1e-1,
        reference_orbital_period_seconds=reference_orbital_period,
    )

    # Smooth the numerical differentiation derived range rate ASD
    smoothed_numerical_range_rate_asd_frequencies, smoothed_numerical_range_rate_psd_values = compute_log_frequency_binned_psd(
        frequencies=first_derivative_spectral_sensitivity_analysis["best_record"]["frequencies"],
        psd_values=first_derivative_spectral_sensitivity_analysis["best_record"]["numerical_differentiated_asd"] ** 2,
    )
    smoothed_numerical_range_rate_asd = np.sqrt(smoothed_numerical_range_rate_psd_values)
    # Smooth the analytical differentiation derived range rate ASD
    smoothed_analytical_range_rate_asd_frequencies, smoothed_analytical_range_rate_psd_values = compute_log_frequency_binned_psd(
        frequencies=first_derivative_spectral_sensitivity_analysis["best_record"]["frequencies"],
        psd_values=first_derivative_spectral_sensitivity_analysis["best_record"]["analytical_differentiated_asd"] ** 2,
    )
    smoothed_analytical_range_rate_asd = np.sqrt(smoothed_analytical_range_rate_psd_values)
    plotter.plot_welch_estimated_asd_comparison(
        estimated_frequencies=smoothed_analytical_range_rate_asd_frequencies,
        estimated_asd_values=smoothed_analytical_range_rate_asd,
        reference_frequencies=smoothed_numerical_range_rate_asd_frequencies,
        reference_asd_values=smoothed_numerical_range_rate_asd,
        file_name="grace_fo_range_rate_noise_estimated_smoothed_asd_comparison.png",
        ordinate_label=r"ASD [m s$^{-1}$ Hz$^{-1/2}$]",
        title="Range-Rate Noise ASD: Welch vs Log-Binned PSD Smoothing",
        estimated_label="Analytical",
        reference_label="Numerical",
        x_limit_inf=1e-5,
        x_limit_sup=1e-1,
        reference_orbital_period_seconds=reference_orbital_period,
    )

    plotter.plot_welch_estimated_asd_comparison(
        estimated_frequencies=second_derivative_spectral_sensitivity_analysis["best_record"]["frequencies"],
        estimated_asd_values=second_derivative_spectral_sensitivity_analysis["best_record"]["numerical_differentiated_asd"],
        reference_frequencies=second_derivative_spectral_sensitivity_analysis["best_record"]["frequencies"],
        reference_asd_values=second_derivative_spectral_sensitivity_analysis["best_record"]["analytical_differentiated_asd"],
        file_name="grace_fo_range_acceleration_noise_asd_comparison.png",
        ordinate_label=r"ASD [m s$^{-2}$ Hz$^{-1/2}$]",
        title="Range-Acceleration Noise ASD: Numerical vs Analytical Differentiation",
        estimated_label=r"Numerical",
        reference_label=r"Analytical",
        x_limit_inf=1e-5,
        x_limit_sup=1e-1,
        reference_orbital_period_seconds=reference_orbital_period,
    )

    # Smooth the numerical differentiation derived range acceleration ASD
    smoothed_numerical_range_acceleration_frequencies, smoothed_numerical_range_acceleration_psd_values = compute_log_frequency_binned_psd(
        frequencies=second_derivative_spectral_sensitivity_analysis["best_record"]["frequencies"],
        psd_values=second_derivative_spectral_sensitivity_analysis["best_record"]["numerical_differentiated_asd"] ** 2,
    )
    smoothed_numerical_range_acceleration_asd_values = np.sqrt(smoothed_numerical_range_acceleration_psd_values)

    # Smooth the analytical differentiation derived range acceleration ASD
    smoothed_analytical_range_acceleration_frequencies, smoothed_analytical_range_acceleration_psd_values = compute_log_frequency_binned_psd(
        frequencies=second_derivative_spectral_sensitivity_analysis["best_record"]["frequencies"],
        psd_values=second_derivative_spectral_sensitivity_analysis["best_record"]["analytical_differentiated_asd"] ** 2,
    )
    smoothed_analytical_range_acceleration_asd = np.sqrt(smoothed_analytical_range_acceleration_psd_values)

    plotter.plot_welch_estimated_asd_comparison(
        estimated_frequencies=smoothed_analytical_range_acceleration_frequencies,
        estimated_asd_values=smoothed_analytical_range_acceleration_asd,
        reference_frequencies=smoothed_numerical_range_acceleration_frequencies,
        reference_asd_values=smoothed_numerical_range_acceleration_asd_values,
        file_name="grace_fo_range_acceleration_noise_smoothed_asd_comparison.png",
        ordinate_label=r"ASD [m s$^{-2}$ Hz$^{-1/2}$]",
        title="Range-Acceleration Noise ASD: Welch vs Log-Binned PSD Smoothing",
        estimated_label="Analytical",
        reference_label="Numerical",
        x_limit_inf=1e-5,
        x_limit_sup=1e-1,
        reference_orbital_period_seconds=reference_orbital_period,
    )

    print("=================================")
    print(
        "Best Welch estimation fit for the range-rate ASD comparison: "
        f"segment_length={int(first_derivative_spectral_sensitivity_analysis['best_record']['segment_length'])}, "
        f"segment_stride={int(first_derivative_spectral_sensitivity_analysis['best_record']['segment_stride'])}, "
        f"lowest_frequency_resolution="
        f"{first_derivative_spectral_sensitivity_analysis['best_record']['lowest_frequency_resolution_hz']:.6e} Hz, "
        f"log-RMS misfit={first_derivative_spectral_sensitivity_analysis['best_record']['log_rms_misfit']:.6e}"
    )
    print(
        "Best Welch estimation fit for the range-acceleration ASD comparison: "
        f"segment_length={int(second_derivative_spectral_sensitivity_analysis['best_record']['segment_length'])}, "
        f"segment_stride={int(second_derivative_spectral_sensitivity_analysis['best_record']['segment_stride'])}, "
        f"lowest_frequency_resolution="
        f"{second_derivative_spectral_sensitivity_analysis['best_record']['lowest_frequency_resolution_hz']:.6e} Hz, "
        f"log-RMS misfit={second_derivative_spectral_sensitivity_analysis['best_record']['log_rms_misfit']:.6e}"
    )
    print("=================================\n")

    # ==========================================================
    # LGD error propagation
    # ==========================================================

    satellite_labels = ("GRACE C", "GRACE D")
    for satellite_label in satellite_labels:
        if satellite_label not in eci_gps_position_noise:
            raise KeyError(f"Missing GPS derived position noise for {satellite_label}.")
        if satellite_label not in eci_gps_velocity_noise:
            raise KeyError(f"Missing GPS derived velocity noise for {satellite_label}.")
        if satellite_label not in accelerometer_observations_sf:
            raise KeyError(f"Missing accelerometer observations  for {satellite_label}.")

    # NOTE: Column indices below are tied to the fixed order of
    # dependent_variables_to_save.
    error_free_target_non_gravitational_acceleration = apply_orbital_phasing_period_exclusion_mask(
        dependent_variables_array[:, 3:6]
        + dependent_variables_array[:, 11:14],
        orbital_phasing_period_exclusion_mask,
    )
    error_free_chaser_non_gravitational_acceleration = apply_orbital_phasing_period_exclusion_mask(
        dependent_variables_array[:, 6:9]
        + dependent_variables_array[:, 14:17],
        orbital_phasing_period_exclusion_mask,
    )
    error_free_relative_non_gravitational_acceleration = (
        error_free_target_non_gravitational_acceleration - error_free_chaser_non_gravitational_acceleration
    )
    error_free_los_relative_non_gravitational_acceleration = np.einsum(
        "ij,ij->i",
        error_free_relative_non_gravitational_acceleration,
        los_unit_vector,
    )
    error_free_lgd = (
        error_free_los_relative_total_acceleration
        - error_free_los_relative_non_gravitational_acceleration
    )

    noisy_target_position = (
        target_position
        + np.asarray(eci_gps_position_noise["GRACE C"], dtype=float)
    )
    noisy_chaser_position = (
        chaser_position
        + np.asarray(eci_gps_position_noise["GRACE D"], dtype=float)
    )
    noisy_relative_position = noisy_target_position - noisy_chaser_position
    noisy_los_unit_vector = noisy_relative_position / np.linalg.norm(
        noisy_relative_position,
        axis=1,
        keepdims=True,
    )

    noisy_target_velocity = (
        target_velocity
        + np.asarray(eci_gps_velocity_noise["GRACE C"], dtype=float)
    )
    noisy_chaser_velocity = (
        chaser_velocity
        + np.asarray(eci_gps_velocity_noise["GRACE D"], dtype=float)
    )
    noisy_relative_velocity = noisy_target_velocity - noisy_chaser_velocity

    noisy_centrifugal_term = (
        np.einsum("ij,ij->i", noisy_relative_velocity, noisy_relative_velocity)
        - best_candidate_numerical_range_rate**2
    ) / noisy_inter_satellite_range
    
    noisy_los_relative_total_acceleration = (
        best_candidate_numerical_range_acceleration
        - noisy_centrifugal_term
    )

    noisy_non_gravitational_acceleration = {}

    for satellite in satellite_labels:

        # Noisy attitude angles
        roll_noisy  = np.asarray(noisy_attitude_time_series[satellite]["roll"], dtype=float)
        pitch_noisy = np.asarray(noisy_attitude_time_series[satellite]["pitch"], dtype=float)
        yaw_noisy   = np.asarray(noisy_attitude_time_series[satellite]["yaw"], dtype=float)

        noisy_pointing_angles = np.column_stack([yaw_noisy, pitch_noisy, roll_noisy])

        rot_matrices_sf_to_losf_noisy = Rotation.from_euler(
            "ZYX",
            noisy_pointing_angles,
            degrees=False,
        ).as_matrix()

        # Noisy positions
        primary_position_noisy = noisy_target_position if satellite == "GRACE C" else noisy_chaser_position
        secondary_position_noisy = noisy_chaser_position if satellite == "GRACE C" else noisy_target_position

        # Compute noisy LOSF to J2000 matrices for both satellites 
        los_vector_j2000_noisy = (
            secondary_position_noisy - primary_position_noisy
        ) / np.linalg.norm(
            secondary_position_noisy - primary_position_noisy,
            axis=-1,
            keepdims=True,
        )

        x_losf_noisy = los_vector_j2000_noisy

        y_losf_noisy = np.cross(x_losf_noisy, primary_position_noisy,) / np.linalg.norm(
            np.cross(x_losf_noisy, primary_position_noisy),
            axis=-1,
            keepdims=True,
        )


        z_losf_noisy = np.cross(x_losf_noisy, y_losf_noisy) / np.linalg.norm(
            np.cross(x_losf_noisy, y_losf_noisy),
            axis=-1,
            keepdims=True,
        )

        rot_matrices_losf_to_j2000_noisy = np.stack(
            [x_losf_noisy, y_losf_noisy, z_losf_noisy],
            axis=-1,
        )
        
        # Accelerometer observations rotated to J2000
        noisy_non_gravitational_acceleration[satellite] = np.einsum(
            "nij,njk,nk->ni",
            rot_matrices_losf_to_j2000_noisy,
            rot_matrices_sf_to_losf_noisy,
            np.asarray(accelerometer_observations_sf[satellite], dtype=float),
        )

    # Project the accelerometer observations back to inertial space using the available
    # noisy satellite-frame attitude history. The observation histories already contain the
    # scale-factor, bias, misalignment, and random-noise effects introduced upstream.
    noisy_target_non_gravitational_acceleration = apply_orbital_phasing_period_exclusion_mask(
        noisy_non_gravitational_acceleration["GRACE C"],
        orbital_phasing_period_exclusion_mask,
    )
    noisy_chaser_non_gravitational_acceleration = apply_orbital_phasing_period_exclusion_mask(
        noisy_non_gravitational_acceleration["GRACE D"],
        orbital_phasing_period_exclusion_mask,
    )
    noisy_relative_non_gravitational_acceleration = (
        noisy_target_non_gravitational_acceleration
        - noisy_chaser_non_gravitational_acceleration
    )
    noisy_los_relative_non_gravitational_acceleration = np.einsum(
        "ij,ij->i",
        noisy_relative_non_gravitational_acceleration,
        noisy_los_unit_vector,
    )

    noisy_lgd = (
        noisy_los_relative_total_acceleration
        - noisy_los_relative_non_gravitational_acceleration
    )

    lgd_error = noisy_lgd - error_free_lgd

    plotter.plot_lgd_error_propagation_time_series(
        time=time,
        lgd_error=lgd_error,
        file_name="grace_fo_lgd_error_propagation_time_series.png",
    )

    lgd_welch_segment_settings = (
        get_welch_segment_settings_from_propagation_segments(
            valid_propagation_segments_mask=valid_propagation_segments_mask,
            time_step=time_step,
            target_lowest_frequency_resolution_hz=LOWEST_FREQUENCY_RESOLUTION_HZ,
            warning_context="LGD error ASD Welch estimation",
        )
    )

    def estimate_amplitude_spectral_density(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Estimate a signal's ASD using the LGD-specific Welch estimation segment settings.

        Args:
            signal (np.ndarray): 1D signal to estimate the ASD of, sampled at
                the enclosing function's time_step and aligned with
                valid_propagation_segments_mask.

        Returns:
            tuple[np.ndarray, np.ndarray]: (frequencies, asd_values).
        """
        frequencies, estimated_psd = compute_weighted_average_psd_across_segments(
            values=signal,
            valid_propagation_segments_mask=valid_propagation_segments_mask,
            time_step=time_step,
            welch_segment_settings=lgd_welch_segment_settings,
        )
        return frequencies, np.sqrt(estimated_psd)

    lgd_error_frequencies, lgd_error_asd = estimate_amplitude_spectral_density(lgd_error)
    save_to_hdf5(
        noise_model_version=noise_model_version,
        file_name="lgd_error_asd.h5",
        frequencies=lgd_error_frequencies,
        asd_values=lgd_error_asd,
        spectrum_name="LGD error ASD",
        ordinate_label="m s^(-2) Hz^(-1/2)",
    )
    smoothed_lgd_error_frequencies, smoothed_lgd_error_psd_values = compute_log_frequency_binned_psd(
        frequencies=lgd_error_frequencies,
        psd_values=lgd_error_asd ** 2,
    )
    smoothed_lgd_error_asd = np.sqrt(smoothed_lgd_error_psd_values)

    plotter.plot_lgd_error_propagation_asd(
        frequencies=lgd_error_frequencies,
        lgd_error_asd=lgd_error_asd,
        smoothed_frequencies=smoothed_lgd_error_frequencies,
        smoothed_lgd_error_asd=smoothed_lgd_error_asd,
        file_name="grace_fo_lgd_error_propagation_asd.png",
        reference_orbital_period_seconds=reference_orbital_period,
    )

    print("=================================")
    print("LGD error propagation summary")
    print(
        f"  RMS propagated LGD error = {compute_root_mean_square_over_finite_values(lgd_error):.6e} m/s^2"
    )
    print(
        "  LGD spectrum Welch estimation settings: "
        f"segment_length={int(lgd_welch_segment_settings[0]['segment_length'])}, "
        f"segment_stride={int(lgd_welch_segment_settings[0]['segment_stride'])}, "
        f"lowest_frequency_resolution={lgd_welch_segment_settings[0]['lowest_frequency_resolution_hz']:.6e} Hz"
    )
    print("=================================\n")

    return None
