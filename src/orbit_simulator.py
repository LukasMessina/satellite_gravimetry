from dataclasses import dataclass
from helpers import get_mean_orbital_period, wrap_rad
from scipy.integrate import quad
from scipy.optimize import brentq
from tudatpy.astro import element_conversion
from tudatpy import dynamics
from tudatpy.util import result2array

from environment_customizer import EnvironmentCustomizer
from guidance import Guidance

import math
import numpy as np

@dataclass
class OrbitalElements:
    """Holds the classical (Keplerian) orbital elements.

    Attributes:
        a_km (float): Semi-major axis [km].
        e (float): Eccentricity [-].
        i_deg (float): Inclination [deg].
        raan_deg (float): Right ascension of the ascending node (RAAN), Ω [deg].
        argp_deg (float): Argument of periapsis, ω [deg].
        M_deg (float): Mean anomaly at epoch [deg].
    """
    a_km: float                 # semi-major axis            [km]
    e: float                    # eccentricity               [-]
    i_deg: float                # inclination                [deg]
    raan_deg: float             # RAAN Ω                     [deg]
    argp_deg: float             # argument of periapsis ω    [deg]
    M_deg: float                # mean anomaly at epoch      [deg]

    def get_along_track_shift(
        self,
        separation_km: float,
        tol_abs: float = 1e-13,
        tol_rel: float = 1e-13,
        ) -> 'OrbitalElements':
        """Compute the orbital elements of a satellite shifted along-track from this one.

        Given this orbit as the chaser, finds the mean anomaly of a targeter
        satellite on the same orbit (same a/e/i/RAAN/argp) offset by a fixed
        along-track arc length, by numerically solving for the true-anomaly
        change that produces the requested arc length, then converting back to
        mean anomaly.

        Args:
            separation_km (float): Along-track separation between chaser and
                targeter [km]. Positive values place the targeter ahead in true
                anomaly.
            tol_abs (float): Absolute tolerance used both for the arc-length
                integration and the root-finding step. Defaults to 1e-13.
            tol_rel (float): Relative tolerance used both for the arc-length
                integration and the root-finding step. Defaults to 1e-13.

        Returns:
            OrbitalElements: A new instance with the same a/e/i/RAAN/argp as
                self, and mean anomaly adjusted for the requested separation.
                Returns self unchanged if separation_km is 0.0.

        Raises:
            ValueError: If self.a_km is not positive.
            RuntimeError: If a valid true-anomaly bracket for the root solver
                could not be found within one revolution.
        """
        if separation_km == 0.0:
            return self
        if self.a_km <= 0:
            raise ValueError("Semi-major axis must be positive.")


        separation_m = separation_km * 1e3
        M_rad = math.radians(self.M_deg)
        nu0_rad = element_conversion.mean_to_true_anomaly(mean_anomaly=M_rad, eccentricity=self.e)
        p_m = self.a_km * 1e3 * (1.0 - self.e**2)

        # Integrand to compute arc length in true anomaly
        def integrand(nu: float) -> float:
            """Return ds/dnu (arc-length element per unit true anomaly) at true anomaly nu."""
            denom = 1.0 + self.e * math.cos(nu)
            radius_m = p_m / denom
            dr_dnu = (p_m * self.e * math.sin(nu)) / (denom**2)
            return math.sqrt(radius_m**2 + dr_dnu**2)
        
        def arc_length(nu1: float) -> float:
            """Compute the arc length between true anomalies."""
            arc_length, _ = quad(integrand, nu0_rad, nu1, epsabs=tol_abs, epsrel=tol_rel)
            return arc_length
        
        # Root function
        def root_function(dnu_rad: float) -> float:
            """Return arc_length(nu0 + dnu_rad) - separation_m, zero at the desired true-anomaly shift."""
            return arc_length(nu1 = nu0_rad + dnu_rad) - separation_m

        # Root bracketing strategy (single-revolution, smallest-magnitude solution)
        # Use local radius as initial guess; expand until bracket.
        initial_radius = p_m / (1.0 + self.e * math.cos(nu0_rad))
        dnu_guess = separation_m / initial_radius # good approximation for near-circular, small separations
        # starting interval around inital guess
        lower_bound = dnu_guess * 0.5
        upper_bound = dnu_guess * 1.5
        
        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound

        # Expand bracket until sign change or until we hit +/-2π (avoid multi-revolution ambiguity)
        func_value_lower_bound = root_function(lower_bound)
        func_value_upper_bound = root_function(upper_bound)
        max_abs = 2.0 * math.pi - 1e-9
        expand = 1.5
        iter = 0
        while func_value_lower_bound * func_value_upper_bound > 0 and (abs(lower_bound) < max_abs and abs(upper_bound) < max_abs) and iter < 50:
            lower_bound *= expand
            upper_bound *= expand
            # clamp
            lower_bound = max(-max_abs, min(max_abs, lower_bound))
            upper_bound = max(-max_abs, min(max_abs, upper_bound))
            if lower_bound > upper_bound:
                lower_bound, upper_bound = upper_bound, lower_bound
            func_value_lower_bound = root_function(lower_bound)
            func_value_upper_bound = root_function(upper_bound)
            iter += 1

        if func_value_lower_bound * func_value_upper_bound > 0:
            raise RuntimeError("Could not bracket Δν for the requested separation within one revolution.")

        dnu = brentq(root_function, lower_bound, upper_bound, xtol=1e-13, rtol=1e-13, maxiter=200)

        # Update true anomaly, then convert back to mean anomaly using Tudat
        nu1 = wrap_rad(nu0_rad + dnu)
        final_eccentric_anomaly = element_conversion.true_to_eccentric_anomaly(true_anomaly=nu1, eccentricity=self.e)
        final_mean_anomaly = element_conversion.eccentric_to_mean_anomaly(eccentric_anomaly=final_eccentric_anomaly, eccentricity=self.e)

        return OrbitalElements(
            a_km=self.a_km,
            e=self.e,
            i_deg=self.i_deg,
            raan_deg=self.raan_deg,
            argp_deg=self.argp_deg,
            M_deg=math.degrees(final_mean_anomaly),
        )


def create_integrator_settings(time_step: float):
    """Build a fixed-step Runge-Kutta-Fehlberg 14(12) integrator configuration.

    Args:
        time_step (float): Fixed integration step size, in seconds.

    Returns:
        tudatpy integrator settings object, ready to pass into a propagator
        settings constructor.
    """
    return dynamics.propagation_setup.integrator.runge_kutta_fixed_step(
        time_step=time_step,
        coefficient_set=dynamics.propagation_setup.integrator.rkf_1412,
    )


def get_final_state_vectors(states_array: np.ndarray) -> np.ndarray:
    """Extract the final Cartesian state vector(s) from a propagated state history.

    Args:
        states_array (np.ndarray): 2D state history array where column 0 is
            epoch and the remaining columns are the stacked satellite state
            vectors, as returned by the translational arc propagator.

    Returns:
        np.ndarray: A copy of the last row's state columns (epoch excluded),
            suitable for use as the initial state of a subsequent arc.
    """
    states_array = np.asarray(states_array, dtype=float)
    return states_array[-1, 1:].copy()


def restructure_vector_history(history_array: np.ndarray) -> np.ndarray:
    """Deduplicate a stacked, multi-arc history array by epoch.

    When several propagation arcs are stacked together (e.g. nominal + cooldown
    arcs), the boundary epoch between two arcs is typically saved twice: once as
    the last row of the earlier arc and once as the first row of the next arc.
    This keeps only the last occurrence of each unique epoch, preserving overall
    time ordering.

    Args:
        history_array (np.ndarray): 2D array where column 0 is epoch and the
            remaining columns are state or dependent-variable values, potentially
            containing duplicate epochs across arc boundaries.

    Returns:
        np.ndarray: The input array filtered down to one row per unique epoch,
            in ascending time order.
    """
    history_array = np.asarray(history_array, dtype=float)
    last_index_for_epoch: dict[float, int] = {}
    for index, epoch in enumerate(history_array[:, 0]):
        last_index_for_epoch[float(epoch)] = index
    unique_indices = np.array(sorted(last_index_for_epoch.values()), dtype=int)
    return history_array[unique_indices]


def create_time_termination_settings(
    final_time: float,
    terminate_exactly_on_final_condition: bool = False,
):
    """Build propagator termination settings that stop at a fixed epoch.

    Args:
        final_time (float): Epoch at which propagation should terminate.
        terminate_exactly_on_final_condition (bool): If True, the propagator
            interpolates to stop exactly at the final time rather than at the next
            integrator step past it. Defaults to False.

    Returns:
        tudatpy termination settings object.
    """
    return dynamics.propagation_setup.propagator.time_termination(
        final_time,
        terminate_exactly_on_final_condition=terminate_exactly_on_final_condition,
    )


def create_nominal_termination_settings(
    bodies,
    guidance_model: Guidance,
    simulation_end_epoch: float,
):
    """Build termination settings for a nominal propagation arc.

    The arc stops at whichever comes first: reaching the simulation end epoch, or
    the intersatellite range error exceeding the guidance model's distance
    threshold (signaling that an orbit-phasing maneuver is needed).

    Args:
        bodies: tudat SystemOfBodies environment, used to read the controlled
            and reference satellites' current states during propagation.
        guidance_model (Guidance): Guidance model providing the controlled/
            reference satellite names and the range-error distance threshold.
        simulation_end_epoch (float): Epoch at which the overall simulation
            should end regardless of range-error conditions.

    Returns:
        tudatpy hybrid termination settings object combining the time and
        custom range-threshold conditions with "stop on first condition met"
        semantics.
    """
    def range_threshold_violation_termination(_: float) -> bool:
        """Return True once the intersatellite range error exceeds the guidance threshold.

        Args:
            _ (float): Current propagation time; unused, but required by
                tudat's custom-termination callback signature.

        Returns:
            bool: True if the current range error magnitude exceeds
                guidance_model.distance_threshold, signaling termination.
        """
        controlled_state = np.asarray(
            bodies.get(guidance_model.controlled_satellite).state,
            dtype=float,
        ).copy()
        reference_state = np.asarray(
            bodies.get(guidance_model.reference_satellite).state,
            dtype=float,
        ).copy()
        _, range_error = guidance_model.get_intersatellite_range_error(
            controlled_state,
            reference_state,
        )
        return abs(range_error) > guidance_model.distance_threshold

    termination_settings_list = [
        create_time_termination_settings(
            simulation_end_epoch,
            terminate_exactly_on_final_condition=False,
        ),
        dynamics.propagation_setup.propagator.custom_termination(
            range_threshold_violation_termination,
        ),
    ]

    return dynamics.propagation_setup.propagator.hybrid_termination(
        [
            *termination_settings_list,
        ],
        fulfill_single_condition=True,
    )


def propagate_translational_arc(
    bodies,
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_states,
    initial_time,
    time_step,
    termination_settings,
    propagator_type,
    output_variables,
):
    """Run a single translational-dynamics propagation arc.

    Wraps tudat's propagator settings construction and simulator invocation for
    one contiguous arc (i.e. no maneuvers or termination-condition changes
    within the arc itself).

    Args:
        bodies: tudat SystemOfBodies environment.
        central_bodies: List of central body names, one per propagated body.
        acceleration_models: tudat acceleration models for the propagated bodies.
        bodies_to_propagate: List of body names to propagate.
        initial_states: Stacked initial Cartesian state vector(s) for the
            propagated bodies at initial_time.
        initial_time (float): Epoch at which this arc begins.
        time_step (float): Fixed integrator step size, in seconds.
        termination_settings: tudatpy termination settings for this arc.
        propagator_type: tudatpy propagator type (e.g. Cowell).
        output_variables: List of tudatpy dependent-variable settings to save
            alongside the state history.

    Returns:
        tuple: (dynamics_simulator, states_array, dependent_variables_array),
            where dynamics_simulator is the tudat simulator object (used for
            diagnostics such as CPU time and function evaluations), and the two
            arrays are the raw (non-deduplicated) state and dependent-variable
            histories for this arc.
    """
    propagator_settings = dynamics.propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_states,
        initial_time,
        create_integrator_settings(time_step),
        termination_settings,
        propagator=propagator_type,
        output_variables=output_variables,
    )

    dynamics_simulator = dynamics.simulator.create_dynamics_simulator(
        bodies,
        propagator_settings,
    )

    states_array = result2array(dynamics_simulator.propagation_results.state_history)
    dependent_variables_array = result2array(
        dynamics_simulator.propagation_results.dependent_variable_history
    )

    return dynamics_simulator, states_array, dependent_variables_array


def propagate_orbits(
    bodies,
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_states,
    simulation_start_epoch: float,
    simulation_end_epoch: float,
    time_step: float,
    propagator_type,
    dependent_variables_to_save,
    guidance_model: Guidance,
    earth_gravitational_parameter: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | bool]], float, float, int]:
    """Propagate the GRACE-FO pair end-to-end, applying orbit-phasing maneuvers as needed.

    Repeatedly propagates translational-dynamics arcs from simulation_start_epoch
    to simulation_end_epoch. Each "nominal" arc runs until either the simulation
    ends or the intersatellite range error exceeds the guidance model's distance
    threshold; when the latter happens, an impulsive entry maneuver is applied,
    a "phasing" arc runs for the planned duration, an impulsive exit maneuver is
    applied, and a "cooldown" arc runs before the next nominal arc begins. All
    arcs are stitched together into single combined state and dependent-variable
    histories.

    Args:
        bodies: tudat SystemOfBodies environment.
        central_bodies: List of central body names, one per propagated body.
        acceleration_models: tudat acceleration models for the propagated bodies.
        bodies_to_propagate: List of body names to propagate (GRACE C, GRACE D).
        initial_states: Stacked initial Cartesian state vectors for both
            satellites at the simulation start epoch.
        simulation_start_epoch (float): Epoch at which propagation begins.
        simulation_end_epoch (float): Epoch at which propagation ends.
        time_step (float): Fixed integrator step size, in seconds.
        propagator_type: tudatpy propagator type (e.g. Cowell).
        dependent_variables_to_save: List of tudatpy dependent-variable settings
            to save alongside the state history.
        guidance_model (Guidance): Guidance model that plans and triggers
            orbit-phasing maneuvers based on intersatellite range error.
        earth_gravitational_parameter (float): Earth's gravitational parameter,
            used to compute each satellite's mean orbital period.

    Returns:
        tuple: (
            states_array: Combined, deduplicated state history (time + stacked
                Cartesian states for both satellites) across all arcs.
            dependent_variables_array: Combined, deduplicated dependent-variable
                history across all arcs.
            guidance_log: List of per-maneuver records (trigger epoch, range
                error, delta-v, phasing duration, completion status, etc.).
            mean_grace_fo_orbital_period: Mean of both satellites' mean orbital
                periods, in seconds.
            total_cpu_time: Total propagation CPU time across all arcs, in
                seconds.
            total_function_evaluations: Total number of dynamics function
                evaluations across all arcs.
        )

    Raises:
        RuntimeError: If the intersatellite range threshold is crossed but no
            valid orbit-phasing maneuver strategy could be computed.
    """

    state_history_propagation_segments: list[np.ndarray] = []
    dependent_variable_history_propagation_segments: list[np.ndarray] = []
    guidance_log: list[dict[str, float | bool]] = []

    current_initial_states = np.asarray(initial_states, dtype=float).copy()
    current_initial_time = float(simulation_start_epoch)
    cooldown_end_time = -np.inf

    total_cpu_time = 0.0
    total_function_evaluations = 0

    while current_initial_time < simulation_end_epoch:
        if current_initial_time < cooldown_end_time:
            current_propagation_arc_label = "cooldown"
            arc_end_time = min(cooldown_end_time, simulation_end_epoch)
            termination_settings = create_time_termination_settings(
                arc_end_time,
                terminate_exactly_on_final_condition=False,
            )
        else:
            current_propagation_arc_label = "nominal"
            termination_settings = create_nominal_termination_settings(
                bodies=bodies,
                guidance_model=guidance_model,
                simulation_end_epoch=simulation_end_epoch,
            )

        dynamics_simulator, propagation_arc_states_array, propagation_arc_dependent_variables_array = propagate_translational_arc(
            bodies=bodies,
            central_bodies=central_bodies,
            acceleration_models=acceleration_models,
            bodies_to_propagate=bodies_to_propagate,
            initial_states=current_initial_states,
            initial_time=current_initial_time,
            time_step=time_step,
            termination_settings=termination_settings,
            propagator_type=propagator_type,
            output_variables=dependent_variables_to_save,
        )

        state_history_propagation_segments.append(propagation_arc_states_array)
        dependent_variable_history_propagation_segments.append(propagation_arc_dependent_variables_array)

        cpu_time_history = dynamics_simulator.cumulative_computation_time_history
        total_cpu_time += list(cpu_time_history.values())[-1]
        function_evaluation_history = dynamics_simulator.cumulative_number_of_function_evaluations
        total_function_evaluations += list(function_evaluation_history.values())[-1]

        current_initial_time = float(propagation_arc_states_array[-1, 0])
        current_initial_states = get_final_state_vectors(propagation_arc_states_array)

        if current_initial_time >= simulation_end_epoch:
            break

        if current_propagation_arc_label == "cooldown":
            # Cooldown arcs always run to their fixed end time; maneuver
            # planning below only applies to nominal arcs.
            continue

        termination_details = dynamics_simulator.propagation_results.termination_details
        termination_flags = list(
            getattr(termination_details, "was_condition_met_when_stopping", [])
        )

        # index 0 of termination_flags corresponds to the time-termination condition
        # from create_nominal_termination_settings; index 1 is the custom
        # range-threshold condition. Only the latter should trigger maneuver
        # planning, so the time condition is excluded here.
        threshold_condition_met = any(termination_flags[1:]) if termination_flags else False
        if not threshold_condition_met:
            break

        controlled_state = current_initial_states[:6]
        reference_state = current_initial_states[6:12]
        planned_orbit_phasing_maneuver_strategy = guidance_model.plan_orbit_phasing_maneuver_strategy(
            current_time=current_initial_time,
            controlled_state=controlled_state,
            reference_state=reference_state,
        )

        if planned_orbit_phasing_maneuver_strategy is None:
            raise RuntimeError(
                "The intersatellite range threshold was crossed, but no valid orbit-phasing maneuver could be computed."
            )

        entry_delta_v_vector = guidance_model.get_impulsive_delta_v_vector(
            controlled_state,
            planned_orbit_phasing_maneuver_strategy.entry_delta_v,
        )
        current_initial_states = EnvironmentCustomizer.apply_impulsive_velocity_deviation_to_state_vector(
            current_initial_states,
            body_index=0,                               # Index referring to GRACE C body object
            delta_v_vector=entry_delta_v_vector,
        )

        maneuver_record: dict[str, float | bool] = {
            "trigger_epoch": planned_orbit_phasing_maneuver_strategy.trigger_epoch,
            "current_range": planned_orbit_phasing_maneuver_strategy.current_range,
            "range_error": planned_orbit_phasing_maneuver_strategy.range_error,
            "entry_delta_v": planned_orbit_phasing_maneuver_strategy.entry_delta_v,
            "final_phasing_epoch": planned_orbit_phasing_maneuver_strategy.final_phasing_epoch,
            "phasing_duration": planned_orbit_phasing_maneuver_strategy.phasing_duration,
            "completed": False,
        }
        guidance_log.append(maneuver_record)

        phasing_end_time = min(planned_orbit_phasing_maneuver_strategy.final_phasing_epoch, simulation_end_epoch)
        dynamics_simulator, propagation_arc_states_array, propagation_arc_dependent_variables_array = propagate_translational_arc(
            bodies=bodies,
            central_bodies=central_bodies,
            acceleration_models=acceleration_models,
            bodies_to_propagate=bodies_to_propagate,
            initial_states=current_initial_states,
            initial_time=current_initial_time,
            time_step=time_step,
            termination_settings=create_time_termination_settings(
                phasing_end_time,
                terminate_exactly_on_final_condition=True,
            ),
            propagator_type=propagator_type,
            output_variables=dependent_variables_to_save,
        )

        state_history_propagation_segments.append(propagation_arc_states_array)
        dependent_variable_history_propagation_segments.append(propagation_arc_dependent_variables_array)

        cpu_time_history = dynamics_simulator.cumulative_computation_time_history
        total_cpu_time += list(cpu_time_history.values())[-1]
        function_evaluation_history = dynamics_simulator.cumulative_number_of_function_evaluations
        total_function_evaluations += list(function_evaluation_history.values())[-1]

        current_initial_time = float(propagation_arc_states_array[-1, 0])
        current_initial_states = get_final_state_vectors(propagation_arc_states_array)
        maneuver_record["actual_phasing_end_epoch"] = current_initial_time

        maneuver_completed = planned_orbit_phasing_maneuver_strategy.final_phasing_epoch <= simulation_end_epoch
        if not maneuver_completed:
            print(
                "Stopping at the simulation end epoch before the second impulsive maneuver of the current orbital rephasing could be applied."
            )
            break

        exit_delta_v_vector = guidance_model.get_impulsive_delta_v_vector(
            current_initial_states[:6],
            planned_orbit_phasing_maneuver_strategy.exit_delta_v,
        )
        current_initial_states = EnvironmentCustomizer.apply_impulsive_velocity_deviation_to_state_vector(
            current_initial_states,
            body_index=0,
            delta_v_vector=exit_delta_v_vector,
        )

        maneuver_record["completed"] = True
        maneuver_record["exit_epoch"] = current_initial_time
        maneuver_record["exit_delta_v"] = planned_orbit_phasing_maneuver_strategy.exit_delta_v

        cooldown_end_time = min(
            current_initial_time + guidance_model.cooldown_duration,
            simulation_end_epoch,
        )

        if current_initial_time >= simulation_end_epoch:
            break

    print("\n=================================")
    print(f"Propagation CPU time : ", total_cpu_time)
    print(f"Number of function evaluations : ", total_function_evaluations)
    print(f"Number of phasing maneuvers : ", len(guidance_log))
    print(
        "Total applied delta-v [m/s] : ",
        sum(
            abs(float(maneuver_record["entry_delta_v"]))
            + (abs(float(maneuver_record["exit_delta_v"])) if maneuver_record.get("completed", False) else 0.0)
            for maneuver_record in guidance_log
        ),
    )
    print("=================================\n")

    stacked_state_history = np.vstack(state_history_propagation_segments)
    del state_history_propagation_segments
    states_array = restructure_vector_history(stacked_state_history)
    del stacked_state_history

    stacked_dependent_variable_history = np.vstack(
        dependent_variable_history_propagation_segments
    )
    del dependent_variable_history_propagation_segments
    dependent_variables_array = restructure_vector_history(
        stacked_dependent_variable_history
    )
    del stacked_dependent_variable_history

    mean_orbital_period_grace_c, mean_orbital_period_grace_d = get_mean_orbital_period(
        dependent_variables_array=dependent_variables_array,
        gravitational_parameter=earth_gravitational_parameter,
    )
    mean_grace_fo_orbital_period = 0.5 * (
        mean_orbital_period_grace_c + mean_orbital_period_grace_d
    )

    return (
        states_array,
        dependent_variables_array,
        guidance_log,
        mean_grace_fo_orbital_period,
        total_cpu_time,
        total_function_evaluations,
    )

