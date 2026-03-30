""" Noise Generator Module """

from typing import List
from helpers import compute_rtn_basis_history, rtn_basis, transform_vector_history_inertial_to_rtn
from pathlib import Path
from plotter import Plotter

import json
import numpy as np
from tudatpy import math 
from pycbc import types, noise, psd
from scipy.spatial.transform import Rotation

class NoiseGenerator:
    """Class to generate different types of noise for satellite measurements. """

    @staticmethod
    def generate_gps_position_noise(
        plotter: Plotter,
        num_epochs: int,
        state_vector: np.ndarray,
        seed: int,
        noise_model_version: int,
        satellite_name: str,
        sigma_rtn: np.ndarray | None = None,
        relative_position_error_asd_json_path: Path | None = None,
        ) -> np.ndarray:
        """ Generate GPS position measurement noise in ECI reference frame."""

        if noise_model_version == 1:

            eci_position_errors = np.empty((num_epochs, 3))
            rtn_position_errors = np.empty((num_epochs, 3))

            # NOTE: The GPS noise model is defined through a one-sided ASD
            # equal to S(f) = S = 1 cm/(Hz^1/2). When converting a continuous
            # white-noise PSD to a discrete-time sequence sampled every time_step,
            # the variance must account for the effective bandwidth introduced by
            # sampling. For a one-sided PSD this bandwidth is fs/2, with fs = 1/time_step.
            # Therefore:
            #
            #     sigma^2 = S^2 * fs / 2 = S^2 / (2*time_step)
            #

            rng = np.random.default_rng(seed)
            position_errors_rtn = rng.normal(0.0, sigma_rtn, size=(num_epochs, 3))

            for k in range(num_epochs):
                r = state_vector[k, 0:3]
                v = state_vector[k, 3:6]
                C_rtn_eci = rtn_basis(r, v)

                # Transform errors from RTN to ECI frame
                position_errors_eci = position_errors_rtn[k, :] @ C_rtn_eci.T

                eci_position_errors[k, :] = position_errors_eci
                rtn_position_errors[k, :] = position_errors_rtn[k, :]

            Plotter.plot_rtn_error_projections(
                plotter,
                samples_rtn=rtn_position_errors,
                sigma_rtn=sigma_rtn,
                file_name=f"{satellite_name}_gps_noise_rtn_projections.png"
            )

        else:

            Plotter.plot_relative_position_error_asd(
                plotter,
                file_name="relative_position_error_asd.png",
                relative_position_error_asd_json_path=relative_position_error_asd_json_path,
                )
            
            with open(relative_position_error_asd_json_path, 'r') as file:
                asd_data = json.load(file)

            frequencies = np.array([float(entry['x']) for entry in asd_data])
            
            # NOTE: The conversion from the relative position error ASD to the
            # absolute position error ASD is valid only under the following assumptions.
            # 1. KBR measurement noise is negligible in the frequency band of interest,
            # so the spectrum is attributed entirely to KO-derived relative orbit error.
            # 2. First-order linearization is valid, i.e. the relative error is small compared
            # to the inter-satellite distance.
            # 3. Both satellites contribute equally to the relative error (identical statistics).
            # 4. The absolute position errors of the two satellites are uncorrelated.
            # 5. The absolute position error of each satellite is isotropic and uncorrelated
            # across Cartesian components.
            # Under these assumptions, the absolute position error ASD of a single satellite
            # is obtained by scaling the relative ASD by a factor 1/sqrt(2).  
            asd_values = np.array([float(entry['y']) for entry in asd_data]) / np.sqrt(2.0)

            # Create a dictionary for the interpolator
            data_to_interpolate = dict(zip(frequencies, asd_values))

            # Set the interpolator settings (linear interpolation)
            interpolator_settings = math.interpolators.linear_interpolation()
            interpolator = math.interpolators.create_one_dimensional_scalar_interpolator(data_to_interpolate, interpolator_settings)

            # Define the coarser time grid and relative number of samples
            time_step = 5.0  # seconds
            num_samples = num_epochs   

            # Initialize position error vectors time histories
            eci_position_errors = np.empty((num_samples, 3))
            rtn_position_errors = np.empty((num_samples, 3))

            # Create a regular frequency span and interpolate ASD values
            delta_f = 1.0 / (num_samples * time_step)
            frequencies_uniform_span = np.arange(frequencies.min(), frequencies.max(), delta_f)
            asd_interpolated = np.array([interpolator.interpolate(freq) for freq in frequencies_uniform_span])

            # Convert ASD to PSD
            psd_interpolated = types.frequencyseries.FrequencySeries(asd_interpolated**2, delta_f)

            # Build a full inertial position-error history.
            eci_noise_time_series = []
            for component_idx in range(3):
                eci_noise_time_series.append(
                    noise.gaussian.noise_from_psd(
                        num_samples,
                        time_step,
                        psd_interpolated,
                        seed + component_idx,
                    )
                )

            eci_position_errors[:, :] = np.column_stack(
                [
                    np.asarray(component_time_series.numpy(), dtype=float)
                    for component_time_series in eci_noise_time_series
                ]
            )

            rtn_position_errors[:, :] = transform_vector_history_inertial_to_rtn(
                eci_position_errors,
                state_vector[:, 0:3],
                state_vector[:, 3:6],
            )


            # Estimate PSD of time series via Welch
            segment_len = int(num_samples / 8)

            # 50% overlap
            seg_stride = segment_len // 2

            estimated_frequencies = []
            estimated_psd_values = []
            for component_time_series in eci_noise_time_series:
                estimated_psd = psd.welch(
                    component_time_series,
                    seg_len=segment_len,
                    seg_stride=seg_stride,
                )
                estimated_frequencies.append(estimated_psd.sample_frequencies.numpy())
                estimated_psd_values.append(estimated_psd.numpy())

            input_frequencies = psd_interpolated.sample_frequencies.numpy()
            input_psd_values = psd_interpolated.numpy()    
            
            # Plot comparison of original and interpolated data
            plotter.plot_linear_interpolation_comparison(
                frequencies,
                asd_values,
                frequencies_uniform_span,
                asd_interpolated,
                file_name=f"{satellite_name}_absolute_position_error_asd_interpolation_comparison.png",
                ordinate_label=r"ASD [m Hz$^{-1/2}$]"
            )

            plotter.plot_absolute_position_error_time_series(
                eci_noise_time_series,
                file_name=f"{satellite_name}_absolute_position_error_time_series.png",
            )

            plotter.plot_welch_estimated_psd_comparison(
                estimated_frequencies,
                estimated_psd_values,
                input_frequencies,
                input_psd_values,
                file_name=f"{satellite_name}_absolute_position_error_welch_estimated_psd_comparison.png",
                ordinate_label=r"PSD [m$^2$ Hz$^{-1}$]",
                title="Absolute Inertial Position Error PSD",
                x_limit_inf=1e-5,
                x_limit_sup=1e-2,
            )

        del rtn_position_errors

        return eci_position_errors
    
    @staticmethod
    def generate_error_free_pointing_angles(
        plotter: Plotter,
        pitch_history_json_path: Path,
        yaw_history_json_path: Path,
        roll_history_json_path: Path,
        num_epochs: int,
        satellite_label: str,
        seed: int,
        generate_plots: bool = True,
        ) -> dict[str, types.TimeSeries]:
        """ Generate error-free pointing angles based on ASD data from JSON files. """

        json_paths =[pitch_history_json_path, yaw_history_json_path, roll_history_json_path]
        file_prefixes = ['pitch', 'yaw', 'roll']
        error_free_pointing_angles_time_series = {}
        seed_variation = 0

        # Load the ASD data from the uploaded JSON file
        for path, file_prefix in zip(json_paths, file_prefixes):
            
            with open(path, 'r') as file:
                asd_data = json.load(file)

            frequencies = np.array([float(entry['x']) for entry in asd_data])
            asd_values = np.array([float(entry['y']) for entry in asd_data])

            # Create a dictionary for the interpolator
            data_to_interpolate = dict(zip(frequencies, asd_values))

            # Set the interpolator settings (linear interpolation)
            interpolator_settings = math.interpolators.linear_interpolation()
            interpolator = math.interpolators.create_one_dimensional_scalar_interpolator(data_to_interpolate, interpolator_settings)

            # Create a regular frequency span and interpolate ASD values
            time_step = 5.0  # seconds
            num_samples = num_epochs 
            delta_f = 1.0 / (num_samples * time_step)
            frequencies_uniform_span = np.arange(frequencies.min(), frequencies.max(), delta_f)
            asd_interpolated = np.array([interpolator.interpolate(freq) for freq in frequencies_uniform_span])

            # Convert ASD to PSD
            psd_interpolated = types.frequencyseries.FrequencySeries(asd_interpolated**2, delta_f)

            # Generate noise using the PSD, sample rate of 5 seconds for a time span of 31 days
            angle_time_series = noise.gaussian.noise_from_psd(num_samples, time_step, psd_interpolated, seed + seed_variation)

            error_free_pointing_angles_time_series[file_prefix] = angle_time_series

            # Estimate PSD of time series via Welch
            segment_len = int(num_samples / 31)

            # 50% overlap
            seg_stride = segment_len // 2

            estimated_psd = psd.welch(angle_time_series, seg_len=segment_len, seg_stride=seg_stride)

            # Extract frequency and PSD values from estimated PSD
            estimated_frequencies = estimated_psd.sample_frequencies.numpy()
            estimated_psd_values = estimated_psd.numpy()

            input_frequencies = psd_interpolated.sample_frequencies.numpy()
            input_psd_values = psd_interpolated.numpy()

            seed_variation += 1            
            
            if generate_plots:
                
                # Plot original vs interpolated data
                plotter.plot_linear_interpolation_comparison(
                    frequencies,
                    asd_values,
                    frequencies_uniform_span,
                    asd_interpolated,
                    file_name=f"{satellite_label}_error_free_{file_prefix}_asd_interpolation_comparison.png"
                )

                # Plot noise time series and basic stats
                plotter.plot_angle_noise_time_series(
                    angle_time_series,
                    file_name=f"{satellite_label}_error_free_{file_prefix}_time_series.png",
                )

                plotter.plot_welch_estimated_psd_comparison(
                    estimated_frequencies,
                    estimated_psd_values,
                    input_frequencies,
                    input_psd_values,
                    file_name=f"{satellite_label}_error_free_{file_prefix}_welch_estimated_psd_comparison.png",
                )
            
        return error_free_pointing_angles_time_series

    @staticmethod
    def generate_pointing_angles_noise(
        plotter: Plotter,
        position: np.ndarray,
        velocity: np.ndarray,
        error_free_attitude_sample_times: np.ndarray,
        error_free_pointing_angles_time_series: dict[str, types.TimeSeries],
        noisy_attitude_sample_times: np.ndarray,
        satellite_label: str,
        seed: int,
        noise_model_version: int,
        white_noise_asd_values: dict[str, dict[str, float]],
        bias_noise_values: dict[str, dict[str, float]],
        counterpart_position: np.ndarray | None = None,
        generate_plots: bool = True,
        ) -> dict[str, types.TimeSeries]:
        """
        Generate perturbations on the pointing angles.

        If noise_model_version == 1, additive white Gaussian noise is applied directly to the error-free pointing angles.
        Otherwise, colored noise is generated from a prescribed amplitude spectral density (ASD) model and mapped to the pointing angles.

        In both cases, a constant bias term is included in the final perturbation.
        """

        pointing_angles = ["pitch", "yaw", "roll"]
        noisy_attitude_time_series = {}

        time_step = 5.0  # seconds
        error_free_attitude_sample_times = np.asarray(error_free_attitude_sample_times, dtype=float)
        noisy_attitude_sample_times = np.asarray(noisy_attitude_sample_times, dtype=float)


        for angle in pointing_angles:
            error_free_pointing_angle_values = np.asarray(
                error_free_pointing_angles_time_series[angle],
                dtype=float,
            )

            error_free_pointing_angles_time_series[angle] = types.timeseries.TimeSeries(
                np.interp(
                    noisy_attitude_sample_times,
                    error_free_attitude_sample_times,
                    error_free_pointing_angle_values,
                    left=error_free_pointing_angle_values[0],
                    right=error_free_pointing_angle_values[-1],
                ),
                delta_t=time_step,
            )

        num_samples = len(noisy_attitude_sample_times)
        delta_f = 1.0 / (num_samples * time_step)

        pointing_angles_noise_time_series = {
            angle: types.timeseries.TimeSeries(
                np.zeros(num_samples, dtype=float),
                delta_t=time_step,
            )
            for angle in pointing_angles
        }

        # =====================================
        # POINTING ANGLES NOISE GENERATION
        # =====================================

        if noise_model_version == 1:
            for seed_variation, angle in enumerate(pointing_angles):

                # Create white noise time series 
                standard_deviation = white_noise_asd_values[satellite_label][angle] * np.sqrt(1 / (2 * time_step))

                rng = np.random.default_rng(seed + seed_variation)
                white_noise_samples = rng.normal(0.0, standard_deviation, size=num_samples)

                # Transform to TimeSeries
                white_noise_time_series = types.timeseries.TimeSeries(
                    white_noise_samples,
                    delta_t=time_step,
                )

                pointing_angles_noise_time_series[angle] = white_noise_time_series

        elif noise_model_version == 2:

            # NOTE:
            # If noise_model_version == 2, the pointing-angle perturbations are first generated in the RTN frame
            # (R: radial, T: along-track, N: cross-track) using the analytical ASD models for star-camera errors
            # defined for Next Generation Gravity Missions (NGGMs) in:
            #
            # Daras, I., & Pail, R. (2017).
            # "Treatment of temporal aliasing effects in the context of next generation satellite gravimetry missions"
            # Journal of Geophysical Research: Solid Earth, 122(9), 7343–7362.
            # DOI: 10.1002/2017JB014250
            #
            # The resulting noise rotation is then expressed in the LOSF frame by applying the time-varying
            # change-of-basis between RTN and LOSF.
            #
            # Finally, equivalent LOSF roll, pitch, and yaw angles are extracted from the resulting rotation matrix
            # and added to the nominal pointing angles following the Darbeheshti measurement model.

            frequency_interval = [1e-12, 1e-1]  # Hz
            frequencies_uniform_span = np.arange(frequency_interval[0], frequency_interval[1], delta_f)

            asd_roll = 1e-5 * np.sqrt(
                ((1e-3 / frequencies_uniform_span) ** 4)
                / (((1e-5 / frequencies_uniform_span) ** 4) + 1.0)
                + 1.0
            )

            asd_pitch_yaw = 2e-6 * np.sqrt(
                ((1e-2 / frequencies_uniform_span) ** 2)
                / (((1e-5 / frequencies_uniform_span) ** 2) + 1.0)
                + 1.0
            )

            analytical_amplitude_spectral_densities = {
                "roll": asd_roll,               # along-track (T axis)
                "pitch": asd_pitch_yaw,         # cross-track (N axis)
                "yaw": asd_pitch_yaw.copy(),    # radial (R axis)
            }

            rtn_euler_angle_histories = {}

            for angle, asd_values in analytical_amplitude_spectral_densities.items():
                analytical_psd = types.frequencyseries.FrequencySeries(asd_values**2, delta_f)

                noise_time_series = noise.gaussian.noise_from_psd(
                    num_samples,
                    time_step,
                    analytical_psd,
                    seed + len(rtn_euler_angle_histories),
                )
                rtn_euler_angle_histories[angle] = np.asarray(
                    noise_time_series,
                    dtype=float,
                )

                segment_len = int(num_samples / 8)
                seg_stride = segment_len // 2

                estimated_psd = psd.welch(
                    noise_time_series,
                    seg_len=segment_len,
                    seg_stride=seg_stride,
                )

                estimated_frequencies = estimated_psd.sample_frequencies.numpy()
                estimated_psd_values = estimated_psd.numpy()
                input_frequencies = analytical_psd.sample_frequencies.numpy()
                input_psd_values = analytical_psd.numpy()

                if generate_plots:
                    plotter.plot_linear_interpolation_comparison(
                        frequencies_uniform_span,
                        asd_values,
                        frequencies_uniform_span,
                        asd_values,
                        file_name=f"{satellite_label}_rtn_{angle}_noise_asd_comparison.png",
                        ordinate_label=r"ASD [rad Hz$^{-1/2}$]",
                    )

                    plotter.plot_angle_noise_time_series(
                        noise_time_series,
                        file_name=f"{satellite_label}_rtn_{angle}_noise_time_series.png",
                    )

                    plotter.plot_welch_estimated_psd_comparison(
                        estimated_frequencies,
                        estimated_psd_values,
                        input_frequencies,
                        input_psd_values,
                        file_name=f"{satellite_label}_rtn_{angle}_noise_welch_estimated_psd_comparison.png",
                        ordinate_label=r"PSD [rad$^2$ Hz$^{-1}$]",
                        title=f"{angle.capitalize()} Noise PSD",
                    )

                del (
                    analytical_psd,
                    noise_time_series,
                    estimated_psd,
                    estimated_frequencies,
                    estimated_psd_values,
                    input_frequencies,
                    input_psd_values,
                )

            yaw_history = rtn_euler_angle_histories["yaw"]
            pitch_history = rtn_euler_angle_histories["pitch"]
            roll_history = rtn_euler_angle_histories["roll"]

            rtn_noise_rotation_matrices = Rotation.from_euler(
                "xzy",
                np.column_stack([yaw_history, pitch_history, roll_history]),
                degrees=False,
            ).as_matrix()

            radial_unit_vector, along_track_unit_vector, normal_unit_vector = compute_rtn_basis_history(
                position,
                velocity,
            )
            rot_matrices_rtn_to_j2000 = np.stack(
                [radial_unit_vector, along_track_unit_vector, normal_unit_vector],
                axis=-1,
            )

            primary_position = np.asarray(position, dtype=float)
            secondary_position = np.asarray(counterpart_position, dtype=float)

            x_losf = (
                secondary_position - primary_position
            ) / np.linalg.norm(
                secondary_position - primary_position,
                axis=-1,
                keepdims=True,
            )
            y_losf = np.cross(x_losf, primary_position) / np.linalg.norm(
                np.cross(x_losf, primary_position),
                axis=-1,
                keepdims=True,
            )
            z_losf = np.cross(x_losf, y_losf)
            z_losf = z_losf / np.linalg.norm(z_losf, axis=-1, keepdims=True)
            rot_matrices_losf_to_j2000 = np.stack([x_losf, y_losf, z_losf], axis=-1)

            rot_matrices_j2000_to_rtn = np.transpose(rot_matrices_rtn_to_j2000, (0, 2, 1))
            rot_matrices_losf_to_rtn = rot_matrices_j2000_to_rtn @ rot_matrices_losf_to_j2000
            rot_matrices_rtn_to_losf = np.transpose(rot_matrices_losf_to_rtn, (0, 2, 1))

            losf_noise_rotation_matrices = (
                rot_matrices_rtn_to_losf
                @ rtn_noise_rotation_matrices
                @ rot_matrices_losf_to_rtn
            )

            losf_noise_euler_angles = Rotation.from_matrix(
                losf_noise_rotation_matrices
            ).as_euler("ZYX", degrees=False)

            pointing_angles_noise_time_series = {
                "yaw": types.timeseries.TimeSeries(
                    losf_noise_euler_angles[:, 0],
                    delta_t=time_step,
                ),
                "pitch": types.timeseries.TimeSeries(
                    losf_noise_euler_angles[:, 1],
                    delta_t=time_step,
                ),
                "roll": types.timeseries.TimeSeries(
                    losf_noise_euler_angles[:, 2],
                    delta_t=time_step,
                ),
            }

            if generate_plots:
                for angle, noise_time_series in pointing_angles_noise_time_series.items():
                    plotter.plot_angle_noise_time_series(
                        noise_time_series,
                        file_name=f"{satellite_label}_losf_{angle}_noise_time_series.png",
                    )

            del (
                rtn_euler_angle_histories,
                yaw_history,
                pitch_history,
                roll_history,
                rtn_noise_rotation_matrices,
                radial_unit_vector,
                along_track_unit_vector,
                normal_unit_vector,
                rot_matrices_rtn_to_j2000,
                primary_position,
                secondary_position,
                x_losf,
                y_losf,
                z_losf,
                rot_matrices_losf_to_j2000,
                rot_matrices_j2000_to_rtn,
                rot_matrices_losf_to_rtn,
                rot_matrices_rtn_to_losf,
                losf_noise_rotation_matrices,
                losf_noise_euler_angles,
            )


        # =====================================
        # NOISY ATTITUDE TIME SERIES GENERATION
        # =====================================

        for angle in pointing_angles:

            noise_time_series = pointing_angles_noise_time_series[angle]
            error_free_pointing_angle_time_series = error_free_pointing_angles_time_series[angle]

            bias_time_series = types.timeseries.TimeSeries(
                np.full(num_samples, bias_noise_values[satellite_label][angle], dtype=float),
                delta_t=time_step,
            )

            noisy_attitude_time_series[angle] = (
                noise_time_series
                + bias_time_series
                + error_free_pointing_angle_time_series
            )

            if generate_plots:

                plotter.plot_angle_noise_time_series(
                    noisy_attitude_time_series[angle],
                    file_name=f"{satellite_label}_total_noisy_{angle}_angle_time_series.png",
                )

        return noisy_attitude_time_series
    
    @staticmethod
    def generate_kbr_system_and_oscillator_noise(
        plotter: Plotter,
        num_epochs: int,
        seed: int,
        noise_model_version: int,
        )-> types.TimeSeries:
        """ Generate system and oscillator noise time series. """

        time_step = 5.0  # seconds
        delta_f = 1.0 / (num_epochs * time_step)

        if noise_model_version == 1:

            # Create white noise time series 
            standard_deviation = 2e-3

            # The frequency uniform is compute solely for plotting purposes
            frequency_interval = [0, 1/(2*time_step)]  # Hz
            frequencies_uniform_span = np.arange(frequency_interval[0], frequency_interval[1], delta_f)
            analytical_asd = np.full_like(frequencies_uniform_span, standard_deviation * np.sqrt(2 * time_step))

            rng = np.random.default_rng(seed)
            samples = rng.normal(0, standard_deviation, size=num_epochs)

            # Transform to TimeSeries
            noise_time_series = types.timeseries.TimeSeries(
                samples,
                delta_t=time_step,
            )

        else:

            # Create a regular frequency span
            frequency_interval = [delta_f, 1e-1 + delta_f]  # Hz
            frequencies_uniform_span = np.arange(frequency_interval[0], frequency_interval[1], delta_f)
            analytical_asd = 1e-6 * np.sqrt(1 + (0.0018 / frequencies_uniform_span)**4)  # [m Hz^-1/2]
            analytical_psd = (1e-6 * np.sqrt(1 + (0.0018 / frequencies_uniform_span)**4))**2  # [m^2 Hz^-1]

            # Convert ASD to PSD
            analytical_psd = types.frequencyseries.FrequencySeries(analytical_psd, delta_f)

            # Generate noise using the PSD, sample rate of 5 seconds for a time span of 31 days
            num_samples = num_epochs 
            noise_time_series = noise.gaussian.noise_from_psd(num_samples, time_step, analytical_psd, seed)

            # Estimate PSD of time series via Welch
            segment_len = int(num_samples // 3)

            # 50% overlap
            seg_stride = segment_len // 2

            estimated_psd = psd.welch(noise_time_series, seg_len=segment_len, seg_stride=seg_stride)

            # Extract frequency and PSD values from estimated PSD
            estimated_frequencies = estimated_psd.sample_frequencies.numpy()
            estimated_psd_values = estimated_psd.numpy()

            input_frequencies = analytical_psd.sample_frequencies.numpy()
            input_psd_values = analytical_psd.numpy()

            plotter.plot_welch_estimated_psd_comparison(
                estimated_frequencies,
                estimated_psd_values,
                input_frequencies,
                input_psd_values,
                file_name=f"kbr_system_and_oscillator_welch_estimated_psd_comparison.png",
                ordinate_label=r"PSD [m$^2$ Hz$^{-1}$]",
                title="KBR System and Oscillator PSD",
                x_limit_inf=0.8e-5,
                x_limit_sup=1e-1,
            )
            
        # Plot original vs interpolated data
        plotter.plot_kbr_system_and_oscillator_asd(
            frequencies_uniform_span,
            analytical_asd,
            file_name=f"kbr_system_and_oscillator_asd.png"
        )

        # Plot noise time series
        plotter.plot_kbr_system_and_oscillator_noise_time_series(
            noise_time_series,
            file_name=f"kbr_system_and_oscillator_noise_time_series.png",
        )

        return noise_time_series
    
    @staticmethod
    def generate_kbr_range_noise(
            error_free_pointing_angles_time_series: dict[str, dict[str, types.TimeSeries]],
            noisy_attitude_time_series: dict[str, dict[str, types.TimeSeries]],
            kbr_system_and_oscillator_noise_timeseries: types.TimeSeries,
            position_data: List[np.ndarray],
            eci_gps_position_noise: dict[str, np.ndarray],
            antenna_phase_center_offset_vector_sf: dict[str, np.ndarray],
            standard_deviation_guessed_antenna_phase_center_offset_vector_sf: dict[str, np.ndarray],
            bias_value: float,
            plotter: Plotter,
        ) -> np.ndarray:
        """ 
        Generate KBR range measurement noise using pointing angles noise time series,
        satellite position data, and systema and oscillator noise timeseries.
        """

        # ==================================================================
        # ANTENNA PHASE CENTRE POINTING JITTER COUPLING NOISE GENERATION 
        # ==================================================================

        # Antenna phase center pointing jitter coupling noise
        apc_pointing_jitter_coupling_noise = dict()

        position_data = {
            "GRACE C": np.asarray(position_data[0], dtype=float),
            "GRACE D": np.asarray(position_data[1], dtype=float),
        }

        num_epochs = len(kbr_system_and_oscillator_noise_timeseries)
        time_seconds = np.asarray(
            kbr_system_and_oscillator_noise_timeseries.sample_times,
            dtype=float,
        )

        for satellite in ["GRACE C", "GRACE D"]:
            counterpart_satellite = "GRACE D" if satellite == "GRACE C" else "GRACE C"

            roll  = np.asarray(error_free_pointing_angles_time_series[satellite]["roll"], dtype=float)
            pitch = np.asarray(error_free_pointing_angles_time_series[satellite]["pitch"], dtype=float)
            yaw   = np.asarray(error_free_pointing_angles_time_series[satellite]["yaw"], dtype=float)

            pointing_angles = np.column_stack([yaw, pitch, roll])
            rot_matrices_sf_to_losf = Rotation.from_euler(
                "ZYX",
                pointing_angles,
                degrees=False,
            ).as_matrix()

            primary_position = position_data[satellite]
            secondary_position = position_data[counterpart_satellite]

            # Compute LOSF to J2000 matrices for both satellites
            los_vector_j2000 = (secondary_position - primary_position) / np.linalg.norm(
                secondary_position - primary_position,
                axis=-1,
                keepdims=True,
            )

            x_losf = los_vector_j2000
            
            
            y_losf = np.cross(los_vector_j2000, primary_position) / np.linalg.norm(
                np.cross(los_vector_j2000, primary_position),
                axis=-1,
                keepdims=True,
            )
            
            z_losf = np.cross(x_losf, y_losf) / np.linalg.norm(np.cross(x_losf, y_losf), axis=-1, keepdims=True)
            rot_matrices_losf_to_j2000 = np.stack(
                [x_losf, y_losf, z_losf],
                axis=-1,
            )

            antenna_phase_center_offset_vector_j2000 = np.einsum(
                "nij,njk,k->ni",
                rot_matrices_losf_to_j2000,
                rot_matrices_sf_to_losf,
                antenna_phase_center_offset_vector_sf[satellite],
            )

            apc_pointing_jitter_coupling_noise[satellite] = -np.einsum(
                "ni,ni->n",
                los_vector_j2000,
                antenna_phase_center_offset_vector_j2000,
            )

            plotter.plot_apc_pointing_jitter_coupling_time_series_demeaned(
                apc_pointing_jitter_coupling_noise=apc_pointing_jitter_coupling_noise[satellite],
                time_seconds=time_seconds,
                satellite_label=satellite,
                file_name=f"{satellite}_apc_pointing_jitter_coupling_noise_demeaned.png",
            )

            # Release full-epoch intermediates before processing the next satellite.
            del (
                roll,
                pitch,
                yaw,
                pointing_angles,
                rot_matrices_sf_to_losf,
                primary_position,
                secondary_position,
                los_vector_j2000,
                x_losf,
                y_losf,
                z_losf,
                rot_matrices_losf_to_j2000,
                antenna_phase_center_offset_vector_j2000,
            )

        
        # ======================================
        # ANTENNA OFFSET CORRECTION GENERATION 
        # ======================================

        residual_apc_coupling_jitter_noise = dict()

        random_seed = {
            "GRACE C": 90,
            "GRACE D": 91,
        }


        for satellite in ["GRACE C", "GRACE D"]:
            counterpart_satellite = "GRACE D" if satellite == "GRACE C" else "GRACE C"

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
            primary_position_noisy = (
                position_data[satellite] + eci_gps_position_noise[satellite]
            )
            secondary_position_noisy = (
                position_data[counterpart_satellite]
                + eci_gps_position_noise[counterpart_satellite]
            )

            # Compute noisy LOSF to J2000 matrices for both satellites 
            los_vector_j2000_noisy = (
                secondary_position_noisy - primary_position_noisy
            ) / np.linalg.norm(
                secondary_position_noisy - primary_position_noisy,
                axis=-1,
                keepdims=True,
            )

            x_losf_noisy = los_vector_j2000_noisy

            los_cross_track_noisy = np.cross(
                x_losf_noisy,
                primary_position_noisy,
            )
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

            rng = np.random.default_rng(random_seed[satellite])
            guessed_antenna_phase_center_offset_vector_sf_errors = rng.normal(
                0.0,
                standard_deviation_guessed_antenna_phase_center_offset_vector_sf[satellite],
                size=(num_epochs, 3),
            )
            
            guessed_antenna_phase_center_offset_vector_sf = (
                antenna_phase_center_offset_vector_sf[satellite]
                + guessed_antenna_phase_center_offset_vector_sf_errors
            )
            # Guessed APC offset rotated to J2000
            guessed_antenna_phase_center_offset_vector_j2000 = np.einsum(
                "nij,njk,nk->ni",
                rot_matrices_losf_to_j2000_noisy,
                rot_matrices_sf_to_losf_noisy,
                guessed_antenna_phase_center_offset_vector_sf,
            )

            # Estimated antenna offset correction
            estimated_antenna_offset_correction = -np.einsum(
                "ni,ni->n",
                los_vector_j2000_noisy,
                guessed_antenna_phase_center_offset_vector_j2000,
            )

            # Residual APC contribution after correction
            residual_apc_coupling_jitter_noise[satellite] = (
                apc_pointing_jitter_coupling_noise[satellite]
                - estimated_antenna_offset_correction
            )

            plotter.plot_residual_apc_coupling_jitter_noise_time_series(
                residual_apc_coupling_jitter_noise=residual_apc_coupling_jitter_noise[satellite],
                time_seconds=time_seconds,
                satellite=satellite,
                file_name=f"{satellite}_residual_apc_time_series.png",
            )

            del (
                roll_noisy,
                pitch_noisy,
                yaw_noisy,
                noisy_pointing_angles,
                rot_matrices_sf_to_losf_noisy,
                primary_position_noisy,
                secondary_position_noisy,
                los_vector_j2000_noisy,
                los_cross_track_noisy,
                x_losf_noisy,
                y_losf_noisy,
                z_losf_noisy,
                rot_matrices_losf_to_j2000_noisy,
                guessed_antenna_phase_center_offset_vector_sf_errors,
                guessed_antenna_phase_center_offset_vector_sf,
                guessed_antenna_phase_center_offset_vector_j2000,
                estimated_antenna_offset_correction,
            )

        # =========================
        # TOTAL KBR RANGE NOISE 
        # =========================   

        bias = 0.0 * kbr_system_and_oscillator_noise_timeseries + bias_value

        # Final length checks
        if len(apc_pointing_jitter_coupling_noise["GRACE C"]) != num_epochs\
              or len(apc_pointing_jitter_coupling_noise["GRACE D"]) != num_epochs\
                or len(bias) != num_epochs:
            raise ValueError(
                f"TimeSeries length mismatch."
            )

        total_kbr_range_noise = (residual_apc_coupling_jitter_noise["GRACE C"] + \
                                residual_apc_coupling_jitter_noise["GRACE D"] + \
                                bias + \
                                np.asarray(kbr_system_and_oscillator_noise_timeseries, dtype=float))

        print("=================================")
        print(f"Total KBR range noise stats: mean={np.mean(total_kbr_range_noise):.3e} m, std={np.std(total_kbr_range_noise):.3e} m")
        print(f"KBR range noise contributions stats:")
        for satellite in ["GRACE C", "GRACE D"]:
            print(f"  {satellite} APC residual pointing jitter coupling noise: mean={np.mean(residual_apc_coupling_jitter_noise[satellite]):.3e} m, std={np.std(residual_apc_coupling_jitter_noise[satellite]):.3e} m")
        print(f"  KBR system and oscillator noise: mean={np.mean(kbr_system_and_oscillator_noise_timeseries):.3e} m, std={np.std(kbr_system_and_oscillator_noise_timeseries):.3e} m")
        print("=================================\n")

        return total_kbr_range_noise
