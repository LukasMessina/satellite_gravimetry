# NOTE: Load warning module and filter out specific deprecation warnings
# to avoid cluttering log output with irrelevant messages.
import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API"
)

# Load standard modules
from pathlib import Path
import numpy as np
import differentiator
from orbit_simulator import OrbitalElements
from findiff.coefs import coefficients, coefficients_non_uni
from typing import Any
from plotter import Plotter

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy import dynamics
from tudatpy.astro import element_conversion
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime

###################################################################
#
#                   GENERAL HELPERS FUNCTIONS
#
###################################################################

def validate_numerical_position_differentiation(
        scenario: str,
        time: np.ndarray,
        position: np.ndarray,
        reference_acceleration: np.ndarray,
        accuracy_orders: list[int],
        plotter: Plotter,
    ) -> dict[int, dict]:
    """
    Compute numerical accelerations from a sampled position time history using finite differences
    and validate them against a reference acceleration time history (e.g., Tudat output).

    Parameters
    ----------
    satellite_name : str
        Identifier used only for labeling/logging.
    time : np.ndarray, shape (N,)
        Strictly increasing time stamps (seconds). Can be uniform or non-uniform.
    position : np.ndarray, shape (N, 3)
        Cartesian position vectors (meters).
    reference_acceleration : np.ndarray, shape (N, 3)
        Reference Cartesian acceleration vectors (m/s^2).
    accuracy_orders : list[int]
        List of finite-difference accuracy orders to test (typically even integers, e.g., [2, 4, 6, 8, 10]).

    Returns
    -------
    results : dict[int, dict]
        Dictionary keyed by accuracy order. For each accuracy, the value is a dictionary with:
        - 'numerical_acceleration': np.ndarray, shape (N, 3)
                The numerically differentiated acceleration.
        - 'error': np.ndarray, shape (N, 3)
                Vector error (m/s^2).
        - 'error_norm': np.ndarray, shape (N,)
                Euclidean norm of the error at each time sample.
        - 'error_rms': float
                Root-mean-square of 'error_norm' over all samples.
        - 'stencil_points': object
                Per-sample stencil metadata, typically describing
                which indices/offsets were used and the corresponding finite-difference coefficients.

    """

    results = {}

    for accuracy in accuracy_orders:

        numerical_acceleration, metadata = differentiator.compute_acceleration(position, time, accuracy)
        error_vector = numerical_acceleration - reference_acceleration
        error_norm = np.linalg.norm(error_vector, axis=1)
        error_root_mean_square = np.sqrt(np.mean(error_norm**2))
        results[accuracy] = {
            'numerical_acceleration': numerical_acceleration,
            'error_vector': error_vector,
            'error_norm': error_norm,
            'error_rms': error_root_mean_square,
            'stencil_points': metadata['stencil_points'],
        }

        plotter.plot_acceleration_finite_difference_statistics(
            scenario=scenario,
            time=time,
            results=results,
        )

    return results

def validate_los_inter_satellite_acceleration(
    scenario: str,
    time: np.ndarray,
    target_position: np.ndarray,
    chaser_position: np.ndarray,
    target_acceleration: np.ndarray,
    chaser_acceleration: np.ndarray,
    accuracy_orders: list[int],
    plotter: Plotter,
    ) -> dict[int, dict]:
    """
    Compute numerical line-of-sight (LOS) inter-satellite accelerations from sampled position time histories
    using finite differences and validate them against reference LOS inter-satellite acceleration time histories.

    Parameters
    ----------
    scenario : str
        Identifier used only for labeling/logging.
    time : np.ndarray, shape (N,)
        Strictly increasing time stamps (seconds). Can be uniform or non-uniform.
    target_position : np.ndarray, shape (N, 3)
        Cartesian position vectors of the target satellite (meters).
    chaser_position : np.ndarray, shape (N, 3)
        Cartesian position vectors of the chaser satellite (meters).
    target_acceleration : np.ndarray, shape (N, 3)
        Reference Cartesian acceleration vectors of the target satellite (m/s^2).
    chaser_acceleration : np.ndarray, shape (N, 3)
        Reference Cartesian acceleration vectors of the chaser satellite (m/s^2).
    accuracy_orders : list[int]
        List of finite-difference accuracy orders to test.

    Returns
    -------
    Dictionary keyed by accuracy order. For each accuracy, the value is a dictionary with:
    - 'numerical_los_inter_satellite_acceleration': np.ndarray, shape (N,)
            The numerically differentiated LOS inter-satellite acceleration.
    - 'error': np.ndarray, shape (N,)
            Vector error (m/s^2).
    - 'error_abs': np.ndarray, shape (N,)
            Absolute error at each time sample.
    - 'error_rms': float
            Root-mean-square of the error over all samples.
    """

    # Inter-satellite baseline, range, LOS
    relative_position = target_position - chaser_position                       
    range_observations = np.linalg.norm(relative_position, axis=1)                   
    los_unit_vector = relative_position / range_observations[:, None]                              

    # Reference: LOS-projected relative acceleration
    reference_intersatellite_acceleration = target_acceleration - chaser_acceleration
    los_reference_intersatellite_acceleration = np.einsum("ij,ij->i", los_unit_vector, reference_intersatellite_acceleration)       

    # Uniform / non-uniform grid check 
    time_steps = np.diff(time)
    initial_time_step = time_steps[0]
    is_uniform = np.allclose(time_steps, initial_time_step, rtol=1e-10, atol=1e-12)

    results: dict[int, dict] = {}

    for accuracy in accuracy_orders:

        los_numerical_intersatellite_acceleration = np.zeros_like(los_reference_intersatellite_acceleration, dtype=float)
        stencil_points = []  # store per-epoch stencil metadata

        if is_uniform:
            coeff_info = coefficients(deriv=2, acc=accuracy)
            # Determine how many points on each side for the central scheme
            center_offsets = np.asarray(coeff_info["center"]["offsets"], dtype=int)
            offset_range = int(np.max(np.abs(center_offsets)))

            for i in range(len(time)):

                # Choose scheme based on border location 
                if i < offset_range:
                    scheme = "forward"
                elif i > len(time) - 1 - offset_range:
                    scheme = "backward"
                else:
                    scheme = "center"

                offsets = np.asarray(coeff_info[scheme]["offsets"], dtype=int)
                weights = np.asarray(coeff_info[scheme]["coefficients"], dtype=float)

                indexes = i + offsets
                cosines = los_unit_vector[indexes] @ los_unit_vector[i]                         
                los_numerical_intersatellite_acceleration[i] = np.sum(weights * cosines * range_observations[indexes]) / (initial_time_step**2)

                stencil_points.append({
                    "scheme": scheme,
                    "offsets": offsets,
                    "weights": weights,
                    "indices": indexes,
                })

        else:
            # Non-uniform: weights already embed the local spacing, so no time step division
            for i in range(len(time)):
                coeff_info = coefficients_non_uni(deriv=2, acc=accuracy, coords=time, idx=i)
                offsets = np.asarray(coeff_info["offsets"], dtype=int)
                weights = np.asarray(coeff_info["coefficients"], dtype=float)

                indexes = i + offsets
                cosines = los_unit_vector[indexes] @ los_unit_vector[i]
                los_numerical_intersatellite_acceleration[i] = np.sum(weights * cosines * range_observations[indexes])

                stencil_points.append({
                    "offsets": offsets,
                    "weights": weights,
                    "indices": indexes,
                })

        # Errors
        error = los_numerical_intersatellite_acceleration - los_reference_intersatellite_acceleration
        absolute_error = np.abs(error)
        error_rms = float(np.sqrt(np.mean(error**2)))

        results[accuracy] = {
            "los_numerical_intersatellite_acceleration": los_numerical_intersatellite_acceleration,
            "error": error,
            "absolute_error": absolute_error,
            "error_rms": error_rms,
            "stencil_points": stencil_points,
        }

    plotter.plot_los_intersatellite_acceleration_finite_difference_statistics(
        scenario=scenario,
        time=time,
        results=results,
    )

    return results


###################################################################
#
#                   FINITE DIFFERENCE METHOD TESTING
#
###################################################################

spice.load_standard_kernels()

# Create default body settings for "Earth"
bodies_to_create = ["Earth"]

# Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
global_frame_origin = "Earth"
global_frame_orientation = "J2000"
body_settings = dynamics.environment_setup.get_default_body_settings(
   bodies_to_create, global_frame_origin, global_frame_orientation)

body_settings.add_empty_settings("GRACE C_keplerian")
body_settings.add_empty_settings("GRACE D_keplerian")
body_settings.add_empty_settings("GRACE C_j0j2")
body_settings.add_empty_settings("GRACE D_j0j2")

# Create system of bodies
bodies = dynamics.environment_setup.create_system_of_bodies(body_settings)

# Define bodies that are propagated
bodies_to_propagate = ["GRACE C_keplerian", "GRACE D_keplerian", "GRACE C_j0j2", "GRACE D_j0j2"]

# Define central bodies of propagation
central_bodies = ["Earth", "Earth", "Earth", "Earth"]

# Define accelerations acting on the GRACE-FO satellites for the keplerian case
keplerian_acceleration_settings_grace_fo = dict(
   Earth=[dynamics.propagation_setup.acceleration.point_mass_gravity()]
)

# Define accelerations acting on the GRACE-FO satellites - J0 + J2 case
j0j2_acceleration_settings_grace_fo = dict(
   Earth=[dynamics.propagation_setup.acceleration.spherical_harmonic_gravity(
       maximum_degree=2,
       maximum_order=0,
   )]
)

acceleration_settings = {"GRACE C_keplerian": keplerian_acceleration_settings_grace_fo, "GRACE D_keplerian": keplerian_acceleration_settings_grace_fo,
                         "GRACE C_j0j2": j0j2_acceleration_settings_grace_fo, "GRACE D_j0j2": j0j2_acceleration_settings_grace_fo}


# Create acceleration models 
acceleration_models = dynamics.propagation_setup.create_acceleration_models(
   bodies, acceleration_settings, bodies_to_propagate, central_bodies)

# TODO: Update simulation time to match 1 month of propagation duration.
simulation_start_epoch = DateTime(2005, 3, 1, 0, 0, 0).to_epoch()
simulation_end_epoch = DateTime(2005, 3, 2, 0, 0, 0).to_epoch()

# Create numerical integrator settings
integrator_settings = dynamics.propagation_setup.integrator.runge_kutta_fixed_step(
   time_step=5.0, coefficient_set=dynamics.propagation_setup.integrator.rkf_1412
)

propagator_type = dynamics.propagation_setup.propagator.cowell

# Create termination settings
termination_settings = dynamics.propagation_setup.propagator.time_termination(simulation_end_epoch)

earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter

###################################################################
#
#                       GRACE-FO ORBIT SIMULATION
#
###################################################################

# GRACE D

grace_d_initial_altitude_km = 477.7
earth_radius_km = bodies.get("Earth").shape_model.average_radius / 1e3
grace_d_initial_orbit_semi_major_axis_km = earth_radius_km + grace_d_initial_altitude_km

grace_d_initial_orbital_elements = OrbitalElements(
                              a_km=grace_d_initial_orbit_semi_major_axis_km,
                              e=0.0019,
                              i_deg=89.0081,
                              raan_deg=0.0,
                              argp_deg=0.0,
                              M_deg=0.0,
                           )

grace_d_initial_state = element_conversion.keplerian_to_cartesian_elementwise(
   gravitational_parameter=earth_gravitational_parameter,
   semi_major_axis=grace_d_initial_orbital_elements.a_km * 1e3,
   eccentricity=grace_d_initial_orbital_elements.e,
   inclination=np.radians(grace_d_initial_orbital_elements.i_deg),
   longitude_of_ascending_node=np.radians(grace_d_initial_orbital_elements.raan_deg),
   argument_of_periapsis=np.radians(grace_d_initial_orbital_elements.argp_deg),
   true_anomaly=element_conversion.mean_to_true_anomaly(
                  mean_anomaly=np.radians(grace_d_initial_orbital_elements.M_deg),
                  eccentricity=grace_d_initial_orbital_elements.e,
               ),
)


# GRACE C

grace_c_initial_orbital_elements = grace_d_initial_orbital_elements.get_along_track_shift(
    separation_km=238.0
)
grace_c_initial_state = element_conversion.keplerian_to_cartesian_elementwise(
   gravitational_parameter=earth_gravitational_parameter,
   semi_major_axis=grace_c_initial_orbital_elements.a_km * 1e3,
   eccentricity=grace_c_initial_orbital_elements.e,
   inclination=np.radians(grace_c_initial_orbital_elements.i_deg),
   longitude_of_ascending_node=np.radians(grace_c_initial_orbital_elements.raan_deg),
   argument_of_periapsis=np.radians(grace_c_initial_orbital_elements.argp_deg),
   true_anomaly=element_conversion.mean_to_true_anomaly(
                  mean_anomaly=np.radians(grace_c_initial_orbital_elements.M_deg),
                  eccentricity=grace_c_initial_orbital_elements.e,
              ),
   )

initial_states = np.hstack((grace_c_initial_state, grace_d_initial_state, grace_c_initial_state, grace_d_initial_state))

dependent_variables_to_save = [
    dynamics.propagation_setup.dependent_variable.total_acceleration("GRACE C_keplerian"),
    dynamics.propagation_setup.dependent_variable.total_acceleration("GRACE D_keplerian"),
    dynamics.propagation_setup.dependent_variable.total_acceleration("GRACE C_j0j2"),
    dynamics.propagation_setup.dependent_variable.total_acceleration("GRACE D_j0j2"),
]

# Create propagation settings
propagator_settings = dynamics.propagation_setup.propagator.translational(
   central_bodies,
   acceleration_models,
   bodies_to_propagate,
   initial_states,
   simulation_start_epoch,
   integrator_settings,
   termination_settings,
   propagator=propagator_type,
   output_variables=dependent_variables_to_save,
)

# Create simulation object and propagate the dynamics
dynamics_simulator = dynamics.simulator.create_dynamics_simulator(
   bodies, propagator_settings
)

# Extract the resulting state history and convert it to an ndarray
states = dynamics_simulator.propagation_results.state_history
states_array = result2array(states)

# Extract the resulting dependent variable history and convert it to an ndarray
dependent_variables = dynamics_simulator.propagation_results.dependent_variable_history
dependent_variables_array = result2array(dependent_variables)

# Perform validation of position double-differentiation for both GRACE-FO satellites and both gravity cases
time = states_array[:, 0]
accuracy_orders = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# Initialize plotter instance
plotter = Plotter(output_path=Path("./GRACE-FO/plots"))

# ====================================================================
#  Satellite acceleration via numerical differentiation validation
# ====================================================================

acceleration_validation_results = {}
scenarios = [
    "GRACE C— Point-Mass Earth Gravity Model",
    "GRACE D— Point-Mass Earth Gravity Model",
    "GRACE C— Earth Gravity Model (J0 + J2)",
    "GRACE D— Earth Gravity Model (J0 + J2)",
]

for idx, satellite in enumerate(["GRACE C_keplerian", "GRACE D_keplerian", "GRACE C_j0j2", "GRACE D_j0j2"]):
    position = states_array[:, 1 + idx * 6:4 + idx * 6]
    reference_acceleration = dependent_variables_array[:, 1 +idx * 3:4 + idx * 3]
    
    results = validate_numerical_position_differentiation(
        scenario=scenarios[idx],
        time=time,
        position=position,
        reference_acceleration=reference_acceleration,
        accuracy_orders=accuracy_orders,
        plotter=plotter,
    )
    
    acceleration_validation_results[satellite] = results

# ===========================================================================
# LOS inter-satellite acceleration via numerical differentiation validation 
# ===========================================================================

scenarios = [
    "GRACE-FO LOS Relative Acceleration Error — Point-Mass Earth Gravity Model",
    "GRACE-FO LOS Relative Acceleration Error — Earth Gravity Model (J0 + J2)",
]

cases = [ "keplerian", "j0j2" ]
los_intersatellite_acceleration_results = {}

for idx, scenario in enumerate(scenarios):

    grace_c_position = states_array[:, 1 + idx * 12:4 + idx * 12]
    grace_d_position = states_array[:, 7 + idx * 12:10 + idx * 12]

    grace_c_acceleration = dependent_variables_array[:, 1 + idx * 6:4 + idx * 6]
    grace_d_acceleration = dependent_variables_array[:, 4 + idx * 6:7 + idx * 6]

    los_intersatellite_acceleration_results[cases[idx]] = validate_los_inter_satellite_acceleration(
        scenario=scenario,
        time=time,
        target_position=grace_c_position,
        chaser_position=grace_d_position,
        target_acceleration=grace_c_acceleration,
        chaser_acceleration=grace_d_acceleration,
        accuracy_orders=accuracy_orders,
        plotter=plotter,
    )

# Check and retrieve finite-difference coefficients for second derivative at various accuracy orders

time_step = np.diff(time)
initial_time_step = time_step[0]
is_uniform_grid = np.allclose(time_step, initial_time_step, rtol=1e-10, atol=1e-12)

if is_uniform_grid:
    print("Uniform time grid detected. \n")

    for accuracy in accuracy_orders: 
        print("="*120) 
        print(f"\nACCURACY ORDER: {accuracy}\n") 
        coeff = coefficients(deriv=2, acc=accuracy, symbolic=True) 
        print(coeff) 
        print("\n")

else:
    print("Non-uniform grid detected. Retrieving coefficients for each time index and accuracy order.\n" \
          "Results stored in 'coefficients_non_uni_grid' dictionary.")

    coefficients_non_uni_grid: dict[int, dict[int, dict[str, Any]]] = {}

    for accuracy in accuracy_orders:
        coefficients_per_idx = {}
        for idx in range(len(time)):  
            coefficients_per_idx[idx] = coefficients_non_uni(
                deriv=2,
                acc=accuracy,
                coords=time,
                idx=idx
            )
        coefficients_non_uni_grid[accuracy] = coefficients_per_idx

