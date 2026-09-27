from __future__ import annotations

from dataclasses import dataclass

from scipy.integrate import quad
from scipy.optimize import brentq
from tudatpy.astro import element_conversion

import math
import numpy as np


@dataclass(frozen=True)
class OrbitPhaseShiftingManeuver:
    """Describes a planned two-burn orbit-phasing maneuver.

    Attributes:
        trigger_epoch (float): Epoch at which the range threshold was crossed
            and this maneuver was planned.
        current_range (float): Intersatellite range at trigger_epoch [m].
        range_error (float): Deviation of current_range from the target range
            [m], positive if too far, negative if too close.
        phasing_orbit_semi_major_axis (float): Semi-major axis of the
            temporary phasing orbit [m].
        phasing_orbit_period (float): Orbital period of the phasing orbit [s].
        phasing_duration (float): Total time spent in the phasing orbit before
            the exit burn, equal to n_revolutions * phasing_orbit_period [s].
        entry_delta_v (float): Along-track velocity change applied at
            trigger_epoch to enter the phasing orbit [m/s].
        exit_delta_v (float): Along-track velocity change applied at
            final_phasing_epoch to exit the phasing orbit and return to the
            nominal orbit [m/s]; equal in magnitude and opposite in sign to
            entry_delta_v.
    """
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
        """Epoch at which the exit burn should be applied.

        Returns:
            float: trigger epoch augmented by phasing duration.
        """
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
        """Initialize the guidance model.

        Args:
            bodies: SystemOfBodies environment, used to read the central
                body's gravitational parameter.
            controlled_satellite (str): Name of the satellite whose orbit is
                adjusted by planned maneuvers.
            reference_satellite (str): Name of the other satellite, used as the
                range reference.
            target_range (float): Nominal intersatellite range to maintain [m].
            n_revolutions (int): Number of orbital revolutions to spend in the
                temporary phasing orbit for each maneuver.
            distance_threshold (float): Maximum allowed deviation from
                target_range before a phasing maneuver is triggered [m].
            cooldown_duration (float): Minimum time to wait after completing a
                maneuver before another can be triggered, in seconds. Defaults
                to 3600.0 (1 hour).
        """
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
        """Compute the current intersatellite range and its deviation from target_range.

        Args:
            controlled_state (np.ndarray): 6-element Cartesian state (position +
                velocity) of the controlled satellite.
            reference_state (np.ndarray): 6-element Cartesian state of the
                reference satellite.

        Returns:
            tuple[float, float]: (current_range, range_error), where
                current_range is the Euclidean distance between the two
                satellites' positions [m], and the range error is
                the current range deviation from the target range [m].
        """
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
        """Plan a two-burn orbit-phasing maneuver to correct the intersatellite range.

        If the current range is within distance_threshold of target_range, no
        maneuver is needed. Otherwise, computes the semi-major-axis offset for a
        temporary phasing orbit that accumulates the required along-track
        correction over n_revolutions, then derives the tangential delta-v
        needed to enter that orbit (the exit burn is the exact opposite).

        Args:
            current_time (float): Epoch at which this plan is being evaluated.
            controlled_state (np.ndarray): 6-element Cartesian state of the
                controlled satellite.
            reference_state (np.ndarray): 6-element Cartesian state of the
                reference satellite.

        Returns:
            OrbitPhaseShiftingManeuver | None: The planned maneuver, or None if
                no maneuver is needed (range within threshold) or none could be
                computed (negligible correction or
                negligible resulting delta-v).
        """
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

        # Negative range_error (too close) requires falling *behind* along-track
        # to open the gap back up, hence the sign flip here.
        true_anomaly_shift = self._arc_length_to_true_anomaly_shift(
            semi_major_axis=controlled_semi_major_axis,
            eccentricity=controlled_eccentricity,
            initial_true_anomaly=controlled_true_anomaly,
            arc_length=-range_error,
        )

        # Linearized relationship between a semi-major-axis offset and the
        # cumulative along-track drift it produces over n_revolutions in the
        # phasing orbit.
        semi_major_axis_variation = (
            -true_anomaly_shift * controlled_semi_major_axis
        ) / (3.0 * math.pi * self.n_revolutions)

        if abs(semi_major_axis_variation) < 1e-6:
            return None

        phasing_orbit_semi_major_axis = controlled_semi_major_axis + semi_major_axis_variation

        initial_position_norm = float(np.linalg.norm(controlled_position))
        initial_velocity_norm = float(np.linalg.norm(controlled_velocity))

        # Vis-viva equation: required orbital speed at the current position for
        # an orbit with the phasing semi-major axis.
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
        """Convert a scalar tangential delta-v into a 3D velocity-change vector.

        The delta-v is applied purely tangentially, i.e. along the current
        velocity direction.

        Args:
            state (np.ndarray): 6-element Cartesian state (position + velocity)
                at the maneuver epoch.
            delta_v (float): Signed magnitude of the velocity change to apply
                along the current velocity direction [m/s].

        Returns:
            np.ndarray: 3-element velocity-change vector, in the same frame as
                state's velocity components.
        """
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
        """Solve for the true-anomaly shift producing a requested along-track arc length.

        Given an osculating ellipse and a requested arc length, solves for the
        true-anomaly increment or decrement such that the orbital prograde or
        retrograde arc length from the initial true anomaly to the shifted true
        anomaly equals arc_length. Uses numerical integration for the arc-length
        function and Brent's method to find its root.

        Args:
            semi_major_axis (float): Semi-major axis of the osculating ellipse [m].
            eccentricity (float): Eccentricity of the osculating ellipse [-];
                must be < 1 (elliptic orbits only).
            initial_true_anomaly (float): Starting true anomaly [rad].
            arc_length (float): Requested signed along-track arc length [m];
                positive shifts the true anomaly forward, negative backward.
            tol_abs (float): Absolute tolerance for both the arc-length
                integration and the root-finding step. Defaults to 1e-13.
            tol_rel (float): Relative tolerance for both the arc-length
                integration and the root-finding step. Defaults to 1e-13.

        Returns:
            float: The true-anomaly shift [rad] such that the arc length from
                initial true anomaly to initial true anomaly + shift equals
                the arc length. Returns 0.0 if the arc length is 0.

        Raises:
            ValueError: If eccentricity is >= 1 (non-elliptic orbit).
            RuntimeError: If a valid root bracket could not be found within one
                revolution.
        """

        if arc_length == 0:
            return 0.0

        if eccentricity >= 1.0:
            raise ValueError("Only elliptic orbits are supported.")

        semi_latus_rectum = semi_major_axis * (1.0 - eccentricity**2)

        def integrand(nu: float) -> float:
            """Return ds/dnu (arc-length element per unit true anomaly) at true anomaly nu."""
            denom = 1.0 + eccentricity * math.cos(nu)
            radius = semi_latus_rectum / denom
            dr_dnu = (semi_latus_rectum * eccentricity * math.sin(nu)) / (denom**2)
            return math.sqrt(radius**2 + dr_dnu**2)

        def get_arc_length(delta_nu: float) -> float:
            """Return the arc length from the initial true anomaly to initial_true_anomaly + the anomaly shift."""
            arc_length, _ = quad(
                integrand,
                initial_true_anomaly,
                initial_true_anomaly + delta_nu,
                epsabs=tol_abs,
                epsrel=tol_rel,
            )
            return arc_length

        def root_function(delta_nu: float) -> float:
            """Return get_arc_length(delta_nu) - arc_length, zero at the desired true anomaly shift."""
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
