""" Compares saved noise model version 1 vs version 2 error spectra and plots their difference. """

from __future__ import annotations

import argparse
import h5py

from pathlib import Path
from typing import Any

import numpy as np

from plotter import Plotter


# Maps each spectrum file to the axis label used when plotting its
# version 1 vs version 2 absolute difference.
COMPARISON_DATA = {
    "range_error_debiased_asd.h5": {
        "difference_label": r"$|\Delta ASD|$ [m Hz$^{-1/2}$]",
    },
    "lgd_error_asd.h5": {
        "difference_label": r"$|\Delta ASD|$ [m s$^{-2}$ Hz$^{-1/2}$]",
    },
}

def load_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a saved frequency/ASD spectrum from its HDF5 file.

    Args:
        path (Path): Path to an HDF5 file with "frequencies" and "asd"
            datasets, as written by differentiator.save_to_hdf5.

    Returns:
        tuple[np.ndarray, np.ndarray]: (frequencies, asd_values), each a 1D
            array of matching length.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If the "frequencies" and "asd" datasets have different
            shapes.
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing spectrum file: {path}")

    with h5py.File(path, "r") as h5_file:
        frequencies = np.asarray(h5_file["frequencies"], dtype=float)
        asd_values = np.asarray(h5_file["asd"], dtype=float)

    frequencies = frequencies.reshape(-1)
    asd_values = asd_values.reshape(-1)

    if frequencies.shape != asd_values.shape:
        raise ValueError(f"Invalid spectrum shape in {path}: frequencies and ASD values differ.")

    return frequencies, asd_values

def interpolate_asd_to_frequency_grid(
    frequencies: np.ndarray,
    asd_values: np.ndarray,
    reference_frequencies: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate ASD values onto a target frequency grid.

    Args:
        frequencies (np.ndarray): 1D array of frequencies the source ASD
            values are sampled at, in ascending order.
        asd_values (np.ndarray): 1D array of ASD values matching frequencies.
        reference_frequencies (np.ndarray): 1D array of frequencies to
            interpolate onto.

    Returns:
        np.ndarray: ASD values interpolated at the reference frequencies.
    """

    return np.interp(reference_frequencies, frequencies, asd_values)


def compute_absolute_spectrum_difference(
    version_a_frequencies: np.ndarray,
    version_a_asd_values: np.ndarray,
    version_b_frequencies: np.ndarray,
    version_b_asd_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Compute the absolute difference between two ASD spectra on a common frequency grid.

    If both spectra already share the same frequency grid, compares them
    directly. Otherwise, restricts to the overlapping frequency range and
    interpolates version b's ASD onto version a's frequency grid before
    differencing.

    Args:
        version_a_frequencies (np.ndarray): Frequency grid for version a's ASD.
        version_a_asd_values (np.ndarray): ASD values for version a.
        version_b_frequencies (np.ndarray): Frequency grid for version b's ASD.
        version_b_asd_values (np.ndarray): ASD values for version b.

    Returns:
        tuple[np.ndarray, np.ndarray, str]: (common_frequencies,
            absolute_difference, grid_description), where grid_description
            describes which grid-matching path was used, for logging purposes.

    Raises:
        ValueError: If the two spectra's frequency ranges do not overlap.
    """

    if (
        version_a_frequencies.shape == version_b_frequencies.shape
        and np.allclose(version_a_frequencies, version_b_frequencies)
    ):
        return (
            version_a_frequencies,
            np.abs(version_b_asd_values - version_a_asd_values),
            "native shared frequency grid",
        )

    lower_frequency = max(version_a_frequencies[0], version_b_frequencies[0])
    upper_frequency = min(version_a_frequencies[-1], version_b_frequencies[-1])
    common_frequency_mask = (
        (version_a_frequencies >= lower_frequency)
        & (version_a_frequencies <= upper_frequency)
    )

    if not np.any(common_frequency_mask):
        raise ValueError("The two spectra do not have an overlapping positive-frequency range.")

    common_frequencies = version_a_frequencies[common_frequency_mask]
    version_a_asd_common_frequencies = version_a_asd_values[common_frequency_mask]
    version_b_asd_common_frequencies = interpolate_asd_to_frequency_grid(
        frequencies=version_b_frequencies,
        asd_values=version_b_asd_values,
        reference_frequencies=common_frequencies,
    )

    return (
        common_frequencies,
        np.abs(version_b_asd_common_frequencies - version_a_asd_common_frequencies),
        "version b interpolated onto version a frequency grid",
    )


if __name__ == "__main__":
    # Compares the fixed set of spectra in COMPARISON_DATA between the two
    # noise model variants output directories and plots each pair's difference.
    spectra_directory_a = Path(f"./output/spectra/version_1")
    spectra_directory_b = Path(f"./output/spectra/version_2")
    plots_output_directory = Path(f"./output/plots/version_comparison")
    plots_output_directory.mkdir(parents=True, exist_ok=True)
    plotter = Plotter(output_path=plots_output_directory)

    difference_records: list[dict[str, Any]] = []

    for file_name, plot_settings in COMPARISON_DATA.items():
        frequencies_a, asd_values_a = load_spectrum(spectra_directory_a / file_name)
        frequencies_b, asd_values_b = load_spectrum(spectra_directory_b / file_name)
        common_frequencies, absolute_difference, grid_description = compute_absolute_spectrum_difference(
            version_a_frequencies=frequencies_a,
            version_a_asd_values=asd_values_a,
            version_b_frequencies=frequencies_b,
            version_b_asd_values=asd_values_b,
        )

        plot_file_name = f"{Path(file_name).stem}_absolute_difference.png"
        plotter.plot_noise_version_asd_absolute_difference(
            frequencies=common_frequencies,
            absolute_difference=absolute_difference,
            file_name=plot_file_name,
            difference_label=plot_settings["difference_label"],
            version_a=1,
            version_b=2,
        )

        print(f"\nSaved {plots_output_directory / plot_file_name} ({grid_description}).")

