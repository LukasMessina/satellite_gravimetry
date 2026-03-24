from __future__ import annotations

from dataclasses import dataclass

from scipy.integrate import quad
from scipy.optimize import brentq
from tudatpy.astro import element_conversion

import math
import numpy as np


@dataclass(frozen=True)
class OrbitPhaseShiftingManeuver:
    trigger_epoch: float
    current_range: float
    range_error: float
    phasing_orbit_semi_major_axis: float
    phasing_orbit_period: float
    phasing_duration: float
    entry_delta_v: float
    exit_delta_v: float

    @property
    def final_phasing_epoch(self) -> float:
        return self.trigger_epoch + self.phasing_duration


class Guidance:
    """
    Orbit phase shifting planner for keeping a deputy/reference separation bounded.

    Strategy
    --------
    - Monitor the current relative distance between controlled_satellite and
      reference_satellite.
    - If the distance deviates from the nominal one by more than distance_threshold,
      compute a phase correction from the current range error.
    - Convert the needed phase correction into a semi-major-axis offset using the
      orbital period linearization with respect to semi-major axis, over
      n_revolutions phasing revolutions.
    - Return two equal-and-opposite tangential impulsive maneuvers:
        * burn 1: enter phasing orbit
        * burn 2: exit phasing orbit after N revolutions
    """

    def __init__(
        self,
        bodies,
        controlled_satellite: str,
        reference_satellite: str,
        target_range: float,
        n_revolutions: int,
        distance_threshold: float,
        cooldown_duration: float = 3600.0,
    ):
        self.controlled_satellite = controlled_satellite
        self.reference_satellite = reference_satellite
        self.central_body = bodies.get("Earth")
        self.central_gravitational_parameter = float(self.central_body.gravitational_parameter)
        self.target_range = float(target_range)
        self.distance_threshold = float(distance_threshold)
        self.n_revolutions = int(n_revolutions)
        self.cooldown_duration = float(cooldown_duration)

    def get_intersatellite_range_error(
        self,
        controlled_state: np.ndarray,
        reference_state: np.ndarray,
    ) -> tuple[float, float]:
        controlled_state = np.asarray(controlled_state, dtype=float)
        reference_state = np.asarray(reference_state, dtype=float)
        current_range = float(np.linalg.norm(controlled_state[:3] - reference_state[:3]))
        range_error = current_range - self.target_range
        return current_range, range_error

    def plan_orbit_phasing_maneuver_strategy(
        self,
        current_time: float,
        controlled_state: np.ndarray,
        reference_state: np.ndarray,
    ) -> OrbitPhaseShiftingManeuver | None:
        controlled_state = np.asarray(controlled_state, dtype=float)
        reference_state = np.asarray(reference_state, dtype=float)

        current_range, range_error = self.get_intersatellite_range_error(controlled_state, reference_state)
        if abs(range_error) <= self.distance_threshold:
            return None

        controlled_position = controlled_state[:3]
        controlled_velocity = controlled_state[3:]

        controlled_orbital_elements = element_conversion.cartesian_to_keplerian(
            controlled_state,
            self.central_gravitational_parameter,
        )

        controlled_semi_major_axis = controlled_orbital_elements[0]
        controlled_eccentricity = controlled_orbital_elements[1]
        controlled_true_anomaly = controlled_orbital_elements[5]

        true_anomaly_shift = self._arc_length_to_true_anomaly_shift(
            semi_major_axis=controlled_semi_major_axis,
            eccentricity=controlled_eccentricity,
            initial_true_anomaly=controlled_true_anomaly,
            arc_length=-range_error,
        )

        semi_major_axis_variation = (
            -true_anomaly_shift * controlled_semi_major_axis
        ) / (3.0 * math.pi * self.n_revolutions)

        if abs(semi_major_axis_variation) < 1e-6:
            return None

        phasing_orbit_semi_major_axis = controlled_semi_major_axis + semi_major_axis_variation

        initial_position_norm = float(np.linalg.norm(controlled_position))
        initial_velocity_norm = float(np.linalg.norm(controlled_velocity))

        vis_viva_argument = self.central_gravitational_parameter * (
            2.0 / initial_position_norm - 1.0 / phasing_orbit_semi_major_axis
        )
        if vis_viva_argument <= 0.0:
            return None

        phasing_orbit_initial_velocity_norm = math.sqrt(vis_viva_argument)
        entry_delta_v = phasing_orbit_initial_velocity_norm - initial_velocity_norm

        if abs(entry_delta_v) < 1e-9:
            return None

        phasing_orbit_period = 2.0 * math.pi * math.sqrt(
            phasing_orbit_semi_major_axis**3 / self.central_gravitational_parameter
        )
        phasing_duration = self.n_revolutions * phasing_orbit_period

        return OrbitPhaseShiftingManeuver(
            trigger_epoch=float(current_time),
            current_range=current_range,
            range_error=range_error,
            phasing_orbit_semi_major_axis=float(phasing_orbit_semi_major_axis),
            phasing_orbit_period=float(phasing_orbit_period),
            phasing_duration=float(phasing_duration),
            entry_delta_v=float(entry_delta_v),
            exit_delta_v=float(-entry_delta_v),
        )

    @staticmethod
    def get_impulsive_delta_v_vector(
        state: np.ndarray,
        delta_v: float,
    ) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        velocity = state[3:]
        velocity_norm = float(np.linalg.norm(velocity))
        return delta_v * velocity / velocity_norm

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
        true-anomaly increment or decrement such that the orbital prograde or
        retrograde arc length from the initial true anomaly to the shifted true
        anomaly equals arc_length.
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

        initial_radius = semi_latus_rectum / (
            1.0 + eccentricity * math.cos(initial_true_anomaly)
        )
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
            raise RuntimeError(
                "Could not bracket delta-nu for the requested arc length within one revolution."
            )

        delta_nu = brentq(
            root_function,
            lower_bound,
            upper_bound,
            xtol=1e-13,
            rtol=1e-13,
            maxiter=200,
        )

        return float(delta_nu)
