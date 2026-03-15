from __future__ import annotations

from dataclasses import dataclass
from scipy.integrate import quad
from scipy.optimize import brentq
from tudatpy.astro import element_conversion

import math
import numpy as np


@dataclass
class OrbitPhaseShiftingWindow:
    start_time: float
    end_time: float
    acceleration_inertial: np.ndarray


class Guidance:
    """
    Simple orbit phase shifting guidance for keeping a deputy/reference separation bounded.

    Strategy
    --------
    - Monitor the current relative distance between controlled_satellite and
      reference_satellite.
    - If the distance deviates from the nominal one by more than distance_threshold,
      compute a phase correction from the current along-track error.
    - Convert the needed phase correction into a semi-major-axis offset using the
      orbital period linearization with respect to semi-major axis.
      where N is the integer number of phasing revolutions.
    - Execute two equal-and-opposite short tangential burns:
        * burn 1: enter phasing orbit
        * burn 2: exit phasing orbit after N revolutions

    Notes
    -----
    - The function is safe under Tudat's NaN-reset protocol for custom models.
    - The trigger uses range deviation, but the correction itself is based on the
      along-track error, since that is what the phasing maneuver primarily fixes.
    """

    def __init__(
        self,
        bodies,
        controlled_satellite: str,
        reference_satellite: str,
        target_range: float,
        n_revolutions: int,
        distance_threshold: float,
        initial_time: float,
        burn_duration: float = 10.0,
    ):
        self.bodies = bodies
        self.controlled_satellite = controlled_satellite
        self.reference_satellite = reference_satellite
        self.controlled_body = bodies.get(controlled_satellite)
        self.reference_body = bodies.get(reference_satellite)
        self.central_body = bodies.get("Earth")
        self.central_gravitational_parameter = float(self.central_body.gravitational_parameter)
        self.target_range = target_range
        self.distance_threshold = float(distance_threshold)
        self.burn_duration = float(burn_duration)
        self.n_revolutions = n_revolutions
        self.initial_time = initial_time

        self.current_time = float("nan")
        self.current_acceleration = np.zeros(3)

        self.active_thrust_firing_windows: list[OrbitPhaseShiftingWindow] = []
        self.last_exit_burn_end_time = -np.inf


    def get_acceleration(self, current_time: float) -> np.ndarray:
        """
        Return the inertial custom acceleration at current_time.
        """
        self._update_guidance(current_time)
        return self.current_acceleration

    # ============================================= 
    #               Main Update Logic
    # ============================================= 
    def _update_guidance(self, current_time: float) -> None:
        # Tudat uses NaN to signal the start of a fresh full state-derivative evaluation.
        if math.isnan(current_time):
            self.current_time = float("nan")
            self.current_acceleration = np.zeros(3)
            return

        # Avoid recomputing if Tudat calls this custom model multiple times for the same
        # state-derivative evaluation.
        if current_time == self.current_time:
            return
        

        self.current_time = current_time
        self.current_acceleration = np.zeros(3)

        controlled_state = np.asarray(self.controlled_body.state, dtype=float).copy()
        reference_state = np.asarray(self.reference_body.state, dtype=float).copy()

        controlled_position = controlled_state[:3]
        controlled_velocity = controlled_state[3:]
        reference_position = reference_state[:3]
        reference_velocity = reference_state[3:]

        relative_position_inertial = controlled_position - reference_position
        relative_velocity_inertial = controlled_velocity - reference_velocity

        rtn_to_inertial_matrix = self._rtn_to_inertial_matrix(reference_position, reference_velocity)
        inertial_to_rtn_matrix = rtn_to_inertial_matrix.T

        relative_position_rtn = inertial_to_rtn_matrix @ relative_position_inertial
        relative_velocity_rtn = inertial_to_rtn_matrix @ relative_velocity_inertial

        current_range = float(np.linalg.norm(relative_position_rtn))
        range_error = current_range - self.target_range

        # If we are currently in one of the scheduled quasi-impulsive thrust firings time windows, apply it.
        for i, window in enumerate(self.active_thrust_firing_windows):
            if window.start_time <= current_time < window.end_time:
                self.current_acceleration = window.acceleration_inertial
                # if len(self.active_thrust_firing_windows) == 2 and i == 0:
                #     print("Applying thrust firing 1")
                # else:
                #     print("Applying thrust firing 2")
                # return

        # Remove already completed thrust firing windows
        self.active_thrust_firing_windows = [window for window in self.active_thrust_firing_windows if window.end_time > current_time]

        # Do not start a new maneuver before the previous one has completed
        if current_time - (self.last_exit_burn_end_time + 3600) <= 0.0:
            return

        # Trigger on total range deviation
        if abs(range_error) <= self.distance_threshold:
            return

        # Plan a phasing maneuver
        self._plan_orbit_phasing_maneuver(
            current_time=current_time,
            reference_state=reference_state,
            controlled_state=controlled_state,
            range_error=range_error,
        )

        for window in self.active_thrust_firing_windows:
            if window.start_time <= current_time < window.end_time:
                self.current_acceleration = window.acceleration_inertial
                return
            
    # ============================================= 
    #           Maneuver Planning Logic
    # ============================================= 
    def _plan_orbit_phasing_maneuver(
        self,
        current_time: float,
        reference_state: np.ndarray,
        controlled_state: np.ndarray,
        range_error: float,
    ) -> None:
        
        reference_position = reference_state[:3]
        reference_velocity = reference_state[3:]
        controlled_position = controlled_state[:3]
        controlled_velocity = controlled_state[3:]

        controlled_orbital_elements = element_conversion.cartesian_to_keplerian(
            controlled_state, 
            self.central_gravitational_parameter
        )

        controlled_semi_major_axis = controlled_orbital_elements[0]
        controlled_eccentricity = controlled_orbital_elements[1]
        controlled_true_anomaly = controlled_orbital_elements[5]

        # Convert the needed range correction into a phase angle using an arc-length inversion.
        true_anomaly_shift = self._arc_length_to_true_anomaly_shift(
                semi_major_axis=controlled_semi_major_axis,
                eccentricity=controlled_eccentricity,
                initial_true_anomaly=controlled_true_anomaly,
                arc_length=-range_error,
            )

        # Controlled satellite orbital period
        initial_orbital_period = 2.0 * math.pi * math.sqrt(
            controlled_semi_major_axis**3 / self.central_gravitational_parameter)

        # linearization:
        delta_semi_major_axis = (-true_anomaly_shift * controlled_semi_major_axis) / (3.0 * math.pi * self.n_revolutions)

        # If the correction is too small, skip it.
        if abs(delta_semi_major_axis) < 1e-6:
            return

        phasing_orbit_semi_major_axis = controlled_semi_major_axis + delta_semi_major_axis

        initial_position_norm = float(np.linalg.norm(controlled_position))
        initial_velocity_norm = float(np.linalg.norm(controlled_velocity))

        vis_viva_argument = self.central_gravitational_parameter * (2.0 / initial_position_norm - 1.0 / phasing_orbit_semi_major_axis)
        if vis_viva_argument <= 0.0:
            return

        phasing_orbit_initial_velocity_norm = math.sqrt(vis_viva_argument)

        # Tangential ΔV to enter phasing orbit
        first_delta_v = phasing_orbit_initial_velocity_norm - initial_velocity_norm

        # Ignore numerically irrelevant maneuvers
        if abs(first_delta_v) < 1e-9:
            return

        controlled_tangential_direction = controlled_velocity / np.linalg.norm(controlled_velocity)
        first_inertial_acceleration = (first_delta_v / self.burn_duration) * controlled_tangential_direction

        phasing_orbit_orbital_period = 2.0 * math.pi * math.sqrt(phasing_orbit_semi_major_axis**3 / self.central_gravitational_parameter)
        exit_burn_start_time = current_time + self.n_revolutions * phasing_orbit_orbital_period

        # Equal and opposite exit quasi impulsive firing
        second_delta_v = -first_delta_v
        second_inertial_acceleration = (second_delta_v / self.burn_duration) * controlled_tangential_direction

        self.active_thrust_firing_windows = [
            OrbitPhaseShiftingWindow(
                start_time=current_time,
                end_time=current_time + self.burn_duration,
                acceleration_inertial=first_inertial_acceleration,
            ),
            OrbitPhaseShiftingWindow(
                start_time=exit_burn_start_time,
                end_time=exit_burn_start_time + self.burn_duration,
                acceleration_inertial=second_inertial_acceleration,
            ),
        ]

        self.last_exit_burn_end_time = exit_burn_start_time + self.burn_duration

    # =====================================
    #           Geometry Helpers
    # =====================================
    @staticmethod
    def _rtn_to_inertial_matrix(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """
        Return the RTN to J2000 rotation matrix based on a reference state.
        Columns are the inertial components of R, T, N unit vectors.
        """
        radial_unit_vector = position / np.linalg.norm(position)
        angular_momentum = np.cross(position, velocity)
        normal_unit_vector = angular_momentum / np.linalg.norm(angular_momentum)
        along_track_unit_vector = np.cross(normal_unit_vector, radial_unit_vector)
        return np.column_stack((radial_unit_vector, along_track_unit_vector, normal_unit_vector))

    # =============================================================
    #           Arc-length to true anomaly shift conversion
    # =============================================================
    @staticmethod
    def _arc_length_to_true_anomaly_shift(
        semi_major_axis: float,
        eccentricity: float,
        initial_true_anomaly: float,
        arc_length: float,
        tol_abs: float = 1e-13,
        tol_rel: float = 1e-13,
    ) -> float:
        """
        Given an osculating ellipse and a requested arc length, solve for the
        true-anomaly increment or decrement such that the orbital prograde or retrograde arc length from the initial
        true anomaly to the shifted true anomaly equals the prograde (positive sign) or retrograde (negative sign) arc_length.
        """
        
        if arc_length == 0:
            return 0.0

        if eccentricity >= 1.0:
            raise ValueError("Only elliptic orbits are supported.")

        semi_latus_rectum = semi_major_axis * (1.0 - eccentricity**2)

        def integrand(nu: float) -> float:
            denom = 1.0 + eccentricity * math.cos(nu)
            radius = semi_latus_rectum / denom
            dr_dnu = (semi_latus_rectum * eccentricity * math.sin(nu)) / (denom**2)
            return math.sqrt(radius**2 + dr_dnu**2)

        def get_arc_length(delta_nu: float) -> float:
            arc_length, _ = quad(
                integrand,
                initial_true_anomaly,
                initial_true_anomaly + delta_nu,
                epsabs=tol_abs,
                epsrel=tol_rel,
            )
            return arc_length

        def root_function(delta_nu: float) -> float:
            return get_arc_length(delta_nu) - arc_length
        
        initial_radius = semi_latus_rectum / (1.0 + eccentricity * math.cos(initial_true_anomaly))
        delta_nu_guess = arc_length / initial_radius

        lower_bound = min(0.5 * delta_nu_guess, 1.5 * delta_nu_guess)
        upper_bound = max(0.5 * delta_nu_guess, 1.5 * delta_nu_guess)

        max_abs = 2.0 * math.pi - 1e-9

        if arc_length > 0.0:
            lower_bound = max(0.0, lower_bound)
            upper_bound = min(max_abs, upper_bound)
        else:
            lower_bound = max(-max_abs, lower_bound)
            upper_bound = min(0.0, upper_bound)

        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound

        func_value_lower_bound = root_function(lower_bound)
        func_value_upper_bound = root_function(upper_bound)

        expand = 1.5
        iteration = 0

        while (
            func_value_lower_bound * func_value_upper_bound > 0.0
            and abs(lower_bound) < max_abs
            and abs(upper_bound) < max_abs
            and iteration < 50
        ):
            lower_bound *= expand
            upper_bound *= expand

            if arc_length > 0.0:
                lower_bound = max(0.0, min(max_abs, lower_bound))
                upper_bound = max(0.0, min(max_abs, upper_bound))
            else:
                lower_bound = max(-max_abs, min(0.0, lower_bound))
                upper_bound = max(-max_abs, min(0.0, upper_bound))

            if lower_bound > upper_bound:
                lower_bound, upper_bound = upper_bound, lower_bound

            func_value_lower_bound = root_function(lower_bound)
            func_value_upper_bound = root_function(upper_bound)
            iteration += 1

        if func_value_lower_bound * func_value_upper_bound > 0.0:
            raise RuntimeError("Could not bracket Δν for the requested arc length within one revolution.")

        delta_nu = brentq(
            root_function,
            lower_bound,
            upper_bound,
            xtol=1e-13,
            rtol=1e-13,
            maxiter=200,
        )

        return float(delta_nu)