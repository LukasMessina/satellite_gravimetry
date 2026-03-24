# NOTE: Load warning module and filter out specific deprecation warnings
# to avoid cluttering log outpuyt with irrelevant messages.
import warnings


warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API"
)

# Load standard modules
from pathlib import Path
import numpy as np
from orbit_simulator import (
    OrbitalElements,
    get_final_state_vectors,
    restructure_vector_history,
    create_time_termination_settings,
    propagate_translational_arc,
    create_nominal_termination_settings,
)
from plotter import Plotter
from noise_generator import NoiseGenerator
from environment_customizer import EnvironmentCustomizer
from guidance import Guidance
from helpers import get_noise_model_version

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy import dynamics
from tudatpy.astro import element_conversion
from tudatpy.astro import gravitation
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime
from tudatpy.dynamics.propagation_setup import dependent_variable
from tudatpy.dynamics import propagation_setup

# =====================================
# NOISE MODEL SELECTION
# =====================================

noise_model_version = get_noise_model_version()

###################################################################
#
#            GRACE-FO ORBIT SIMULATION AND INITIALIZATION
#
###################################################################

simulation_start_epoch = DateTime(2019, 1, 1, 0, 0, 0).to_epoch()
simulation_end_epoch = DateTime(2019, 2, 1, 0, 0, 0).to_epoch() 
time_step = 5.0  # seconds
number_epochs = int(np.floor((simulation_end_epoch - simulation_start_epoch) / time_step)) + 1

# =====================================
# ASD NOISE GENERATION 
# =====================================

plotter = Plotter(output_path=Path("./GRACE-FO/plots"))

pitch_history_json_path=Path("data/pitch_angles_asd_data.json")
yaw_history_json_path=Path("data/yaw_angles_asd_data.json")
roll_history_json_path=Path("data/roll_angles_asd_data.json")


Plotter.plot_pointing_angles_asd(
   plotter,
   file_name="pointing_angles_asd.png",
   pitch_history_json_path=pitch_history_json_path,
   yaw_history_json_path=yaw_history_json_path,
   roll_history_json_path=roll_history_json_path,
)

error_free_pointing_angles_time_series = dict()
noisy_attitude_time_series = dict()

# NOTE: These values are taken from the source code associated with the paper
# "Instrument data simulations for GRACE Follow-on: observation and noise models"
# by Neda Darbeheshti et al. The code is available at:
# https://github.com/Darbeheshti/GRACE-Follow-On-simulator

white_noise_asd_values = {
    "GRACE C": {
        "roll": 20e-6,
        "pitch": 20e-6,
        "yaw": 20e-6,
    },
    "GRACE D": {
        "roll": 20e-6,
        "pitch": 20e-6,
        "yaw": 20e-6,
    }
}

bias_noise_values = {
    "GRACE C": {
        "roll": 1.2e-3,
        "pitch": -2.2e-3,
        "yaw": 1.8e-3,
    },
    "GRACE D": {
        "roll": -2.9e-3,
        "pitch": -1.8e-3,
        "yaw": 2.1e-3,
    }
}

for satellite, seed in zip(["GRACE C", "GRACE D"], [42, 43]):
    
   # Generate pointing angles noise time series for each satellite

    error_free_pointing_angles_time_series[satellite], noisy_attitude_time_series[satellite] = NoiseGenerator.generate_pointing_angles_noise(
        plotter,
        pitch_history_json_path,
        yaw_history_json_path,
        roll_history_json_path,
        number_epochs,
        satellite_label=satellite,
        seed=seed,
        white_noise_asd_values=white_noise_asd_values[satellite],
        bias_noise_values=bias_noise_values[satellite]
    )

rotation_model_context: dict[str, object] = {"bodies": None}
attitude_noise_sample_times = simulation_start_epoch + np.arange(number_epochs, dtype=float) * time_step

grace_c_custom_rotation_matrix_callable = EnvironmentCustomizer.create_custom_spacecraft_rotation_function(
    spacecraft_name="GRACE C",
    counterpart_name="GRACE D",
    attitude_noise_history=error_free_pointing_angles_time_series["GRACE C"],
    sample_times=attitude_noise_sample_times,
    rotation_model_context=rotation_model_context,
)

grace_d_custom_rotation_matrix_callable = EnvironmentCustomizer.create_custom_spacecraft_rotation_function(
    spacecraft_name="GRACE D",
    counterpart_name="GRACE C",
    attitude_noise_history=error_free_pointing_angles_time_series["GRACE D"],
    sample_times=attitude_noise_sample_times,
    rotation_model_context=rotation_model_context,
)

# NOTE: This commented section shows how the initial states of the
# GRACE-FO satellites can be defined using Keplerian orbital elements.
# This method is currently not used in the simulation, as the initial
# states are now obtained through a different approach.

# # GRACE D

# grace_d_initial_altitude_km = 477.7
# earth_radius_km = bodies.get("Earth").shape_model.average_radius / 1e3
# grace_d_initial_orbit_semi_major_axis_km = earth_radius_km + grace_d_initial_altitude_km

# grace_d_initial_orbital_elements = OrbitalElements(
#                               a_km=grace_d_initial_orbit_semi_major_axis_km,
#                               e=0.0019,
#                               i_deg=89.0081,
#                               raan_deg=0.0,
#                               argp_deg=0.0,
#                               M_deg=0.0,
#                            )

# grace_d_initial_state = element_conversion.keplerian_to_cartesian_elementwise(
#    gravitational_parameter=earth_gravitational_parameter,
#    semi_major_axis=grace_d_initial_orbital_elements.a_km * 1e3,
#    eccentricity=grace_d_initial_orbital_elements.e,
#    inclination=np.radians(grace_d_initial_orbital_elements.i_deg),
#    longitude_of_ascending_node=np.radians(grace_d_initial_orbital_elements.raan_deg),
#    argument_of_periapsis=np.radians(grace_d_initial_orbital_elements.argp_deg),
#    true_anomaly=element_conversion.mean_to_true_anomaly(
#                   mean_anomaly=np.radians(grace_d_initial_orbital_elements.M_deg),
#                   eccentricity=grace_d_initial_orbital_elements.e,
#                ),
# )


# # GRACE C

# grace_c_initial_orbital_elements = grace_d_initial_orbital_elements.get_along_track_shift(
#     separation_km=238.0
# )
# grace_c_initial_state = element_conversion.keplerian_to_cartesian_elementwise(
#    gravitational_parameter=earth_gravitational_parameter,
#    semi_major_axis=grace_c_initial_orbital_elements.a_km * 1e3,
#    eccentricity=grace_c_initial_orbital_elements.e,
#    inclination=np.radians(grace_c_initial_orbital_elements.i_deg),
#    longitude_of_ascending_node=np.radians(grace_c_initial_orbital_elements.raan_deg),
#    argument_of_periapsis=np.radians(grace_c_initial_orbital_elements.argp_deg),
#    true_anomaly=element_conversion.mean_to_true_anomaly(
#                   mean_anomaly=np.radians(grace_c_initial_orbital_elements.M_deg),
#                   eccentricity=grace_c_initial_orbital_elements.e,
#               ),
#    )

# The initial state vectors of the GRACE-FO satellites are defined based
# on its TLEs at the simulation epoch, which are obtained from space-track.org
grace_c_tle = dynamics.environment_setup.ephemeris.sgp4(
    "1 43476U 18047A   18365.89673225 +.00000266 +00000-0 +96631-5 0  9993",
    "2 43476 088.9970 209.2025 0019272 138.9449 221.3254 15.23792965033965"
)
grace_d_tle = dynamics.environment_setup.ephemeris.sgp4(
    "1 43477U 18047B   18365.89702718 +.00000263 +00000-0 +95755-5 0  9997",
    "2 43477 088.9971 209.2044 0019074 138.5411 221.7288 15.23793300033968"
)

grace_c_ephemeris = dynamics.environment_setup.create_body_ephemeris(grace_c_tle, "GRACE C")
grace_d_ephemeris = dynamics.environment_setup.create_body_ephemeris(grace_d_tle, "GRACE D")

grace_c_initial_state = grace_c_ephemeris.cartesian_state(simulation_start_epoch)  # [m, m/s]
grace_d_initial_state = grace_d_ephemeris.cartesian_state(simulation_start_epoch)  # [m, m/s]

# Initial range between the two satellites, which is used as a reference value for the guidance model.
target_range = np.linalg.norm(grace_c_initial_state[0:3] - grace_d_initial_state[0:3]) # [m] 

initial_states = np.hstack((grace_c_initial_state, grace_d_initial_state))


###################################################################
#
#                   PROPAGATOR & ENVIRONMENT SETUP
#
###################################################################

spice.load_standard_kernels()
spice.load_kernel("./kernels/nep105.bsp")
spice.load_kernel("./kernels/plu060.bsp")
spice.load_kernel("./kernels/ura182.bsp")
spice.load_kernel("./kernels/gm_de440.pck")

# Vehicle configuration parameters
# TODO: Improve the drag and solar radiation pressure modelling to match GRACE-FO specifications.
# Ref: https://isdc-data.gfz.de/grace-fo/DOCUMENTS/Level-1/GRACE-FO_L1_Data_Product_User_Handbook_20190911.pdf
# Ref: https://agupubs-onlinelibrary-wiley-com.tudelft.idm.oclc.org/doi/epdf/10.1029/2020JB021297
drag_coefficient = 0.2
# Ref: https://essd.copernicus.org/articles/9/833/2017/
grace_fo_mass = 655.0   # [kg]

grace_c_per_source_occulting_bodies = {
   "Sun": ["Earth", "Moon"],
   }

grace_d_per_source_occulting_bodies = {
   "Sun": ["Earth", "Moon"],
   }

# Gravitational field parameters for the Sun
# Ref: https://iopscience.iop.org/article/10.3847/1538-4357/aca8a4/pdf
J2_sun = 2.07e-7
unormalized_cosine_coefficients_sun = np.zeros((3, 3))
unormalized_sine_coefficients_sun = np.zeros((3, 3))
unormalized_cosine_coefficients_sun[0, 0] = 1.0
unormalized_cosine_coefficients_sun[2, 0] = -J2_sun
# Normalize the Sun's gravitational field coefficients
normalized_cosine_coefficients_sun, normalized_sine_coefficients_sun = gravitation.normalize_spherical_harmonic_coefficients(
   unormalized_cosine_coefficients_sun,
   unormalized_sine_coefficients_sun,
)
sun_gravitational_parameter = spice.get_body_gravitational_parameter("Sun")
sun_astrometric_mean_radius = 695508000  # [m], astrometric (mean) radius of the Sun

# Create body settings 
bodies_to_create = [
               "Earth",
               "Sun",
               "Mercury",
               "Venus",
               "Mars",
               "Moon",
               "Jupiter",
               "Io",
               "Europa",
               "Ganymede",
               "Callisto",
               "Amalthea",
               "Saturn",
               "Titan",
               "Rhea",
               "Iapetus",
               "Dione",
               "Tethys",
               "Enceladus",
               "Hyperion",
               "Mimas",
               "Uranus",
               "Neptune",
               "Pluto",
               "Ceres",
               "Vesta",
               "Phobos",
               "Deimos",
               ]

# Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
global_frame_origin = "Earth"
global_frame_orientation = "J2000"
body_settings = dynamics.environment_setup.get_default_body_settings(
   bodies_to_create, global_frame_origin, global_frame_orientation)

body_settings.get("Sun").gravity_field_settings = dynamics.environment_setup.gravity_field.spherical_harmonic(
      gravitational_parameter = sun_gravitational_parameter,
      reference_radius = sun_astrometric_mean_radius,
      normalized_cosine_coefficients = normalized_cosine_coefficients_sun,
      normalized_sine_coefficients = normalized_sine_coefficients_sun,
      associated_reference_frame = "IAU_Sun"
)

body_settings.add_empty_settings("GRACE C")
body_settings.add_empty_settings("GRACE D")

body_settings.get("GRACE C").rotation_model_settings = dynamics.environment_setup.rotation_model.custom_rotation_model(
    base_frame="J2000",
    target_frame="GRACE C_SF",
    custom_rotation_matrix_function=grace_c_custom_rotation_matrix_callable,
    finite_difference_time_step=time_step,
)

body_settings.get("GRACE D").rotation_model_settings = dynamics.environment_setup.rotation_model.custom_rotation_model(
    base_frame="J2000",
    target_frame="GRACE D_SF",
    custom_rotation_matrix_function=grace_d_custom_rotation_matrix_callable,
    finite_difference_time_step=time_step,
)

# Loading the macromodel
grace_fo_material_properties = {
    "SiOx_Kapton_Front_Rear": dynamics.environment_setup.vehicle_systems.material_properties(
        specular_reflectivity=0.71,
        diffuse_reflectivity=0.15,
    ),
    "SiOx_Kapton_Apron": dynamics.environment_setup.vehicle_systems.material_properties(
        specular_reflectivity=0.16,
        diffuse_reflectivity=0.79,
    ),
    "Si_Glass_Solar_Arrays": dynamics.environment_setup.vehicle_systems.material_properties(
        specular_reflectivity=0.0,
        diffuse_reflectivity=0.10,
    ),
    "Si_Glass_Zenith": dynamics.environment_setup.vehicle_systems.material_properties(
        specular_reflectivity=1.0,
        diffuse_reflectivity=0.0,
    ),
    "Teflon_Nadir": dynamics.environment_setup.vehicle_systems.material_properties(
        specular_reflectivity=0.45,
        diffuse_reflectivity=0.05,
    ),
    }   

grace_fo_reradiation_settings = {
    "SiOx_Kapton_Front_Rear": True,
    "SiOx_Kapton_Apron": True,
    "Si_Glass_Solar_Arrays": True,
    "Si_Glass_Zenith": True,
    "Teflon_Nadir": True,
}

grace_fo_frame_origin = np.array([0.0, 0.0, 0.0])  # [m], origin of the spacecraft bus frame in the SF frame

grace_bus_panels = dynamics.environment_setup.vehicle_systems.body_panel_settings_list_from_dae(
    file_path=str(Path("./data/grace_fo_low_fidelity_v1.dae")),
    frame_origin=grace_fo_frame_origin,
    material_properties=grace_fo_material_properties,
    reradiation_settings=grace_fo_reradiation_settings,
    input_unit="mm",
)

grace_fo_full_panelled_body = dynamics.environment_setup.vehicle_systems.full_panelled_body_settings(
    grace_bus_panels,
)

body_settings.get("GRACE C").vehicle_shape_settings = grace_fo_full_panelled_body
body_settings.get("GRACE D").vehicle_shape_settings = grace_fo_full_panelled_body

grace_c_target_settings = dynamics.environment_setup.radiation_pressure.panelled_radiation_target(
    grace_c_per_source_occulting_bodies
)

grace_d_target_settings = dynamics.environment_setup.radiation_pressure.panelled_radiation_target(
    grace_d_per_source_occulting_bodies
)

body_settings.get("GRACE C").radiation_pressure_target_settings = (
    grace_c_target_settings
)

body_settings.get("GRACE D").radiation_pressure_target_settings = (
    grace_d_target_settings
)

aero_coefficients_settings = dynamics.environment_setup.aerodynamic_coefficients.constant_variable_cross_section(
    [drag_coefficient, 0.0, 0.0])

# Add the aerodynamic interface to the body settings
body_settings.get("GRACE C").aerodynamic_coefficient_settings = aero_coefficients_settings
body_settings.get("GRACE D").aerodynamic_coefficient_settings = aero_coefficients_settings

# create atmosphere settings and add to body settings of body "Earth"
body_settings.get( "Earth" ).atmosphere_settings = dynamics.environment_setup.atmosphere.nrlmsise00()

# Group all Solar System bodies into a single effective barycentric mass
# and model their combined gravitational effect as one equivalent source.

# Martian System
mars_gravitational_parameter = spice.get_body_gravitational_parameter("Mars")
phobos_gravitational_parameter = spice.get_body_gravitational_parameter("Phobos")
deimos_gravitational_parameter = spice.get_body_gravitational_parameter("Deimos")
mars_system_gravitational_parameter = mars_gravitational_parameter + phobos_gravitational_parameter + deimos_gravitational_parameter
body_settings.get("Mars").gravity_field_settings = dynamics.environment_setup.gravity_field.central(
   gravitational_parameter = mars_system_gravitational_parameter
)
body_settings.get( "Mars" ).ephemeris_settings = dynamics.environment_setup.ephemeris.direct_spice(
            global_frame_origin, global_frame_orientation, "Mars Barycenter" )

# Jovian System
jupiter_gravitational_parameter = spice.get_body_gravitational_parameter("Jupiter")
io_gravitational_parameter = spice.get_body_gravitational_parameter("Io")
europa_gravitational_parameter = spice.get_body_gravitational_parameter("Europa")
ganymede_gravitational_parameter = spice.get_body_gravitational_parameter("Ganymede")
callisto_gravitational_parameter = spice.get_body_gravitational_parameter("Callisto")
amalthea_gravitational_parameter = spice.get_body_gravitational_parameter("Amalthea")
jupiter_system_gravitational_parameter = (jupiter_gravitational_parameter + io_gravitational_parameter +\
                                           europa_gravitational_parameter + ganymede_gravitational_parameter +\
                                               callisto_gravitational_parameter + amalthea_gravitational_parameter)
body_settings.get("Jupiter").gravity_field_settings = dynamics.environment_setup.gravity_field.central(
   gravitational_parameter = jupiter_system_gravitational_parameter
)
body_settings.get( "Jupiter" ).ephemeris_settings = dynamics.environment_setup.ephemeris.direct_spice(
            global_frame_origin, global_frame_orientation, "Jupiter Barycenter" )

# Saturnian System
saturn_gravitational_parameter = spice.get_body_gravitational_parameter("Saturn")
titan_gravitational_parameter = spice.get_body_gravitational_parameter("Titan")
rhea_gravitational_parameter = spice.get_body_gravitational_parameter("Rhea")
iapetus_gravitational_parameter = spice.get_body_gravitational_parameter("Iapetus")
dione_gravitational_parameter = spice.get_body_gravitational_parameter("Dione")
tethys_gravitational_parameter = spice.get_body_gravitational_parameter("Tethys")
enceladus_gravitational_parameter = spice.get_body_gravitational_parameter("Enceladus")
hyperion_gravitational_parameter = spice.get_body_gravitational_parameter("Hyperion")
mimas_gravitational_parameter = spice.get_body_gravitational_parameter("Mimas")
saturn_system_gravitational_parameter = (saturn_gravitational_parameter + titan_gravitational_parameter +\
                                          rhea_gravitational_parameter + iapetus_gravitational_parameter +\
                                               dione_gravitational_parameter + tethys_gravitational_parameter +\
                                                  enceladus_gravitational_parameter + hyperion_gravitational_parameter +\
                                                     mimas_gravitational_parameter)

body_settings.get("Saturn").gravity_field_settings = dynamics.environment_setup.gravity_field.central(
   gravitational_parameter = saturn_system_gravitational_parameter
)
body_settings.get( "Saturn" ).ephemeris_settings = dynamics.environment_setup.ephemeris.direct_spice(
            global_frame_origin, global_frame_orientation, "Saturn Barycenter" )

# Create system of bodies
bodies = dynamics.environment_setup.create_system_of_bodies(body_settings)
rotation_model_context["bodies"] = bodies

bodies.get("GRACE C").mass = grace_fo_mass
bodies.get("GRACE D").mass = grace_fo_mass

# Define bodies that are propagated
bodies_to_propagate = ["GRACE C", "GRACE D"]

# Define central bodies of propagation
central_bodies = ["Earth", "Earth"]

# Define relativistic correction settings for the Earth
# Select terms to be used
use_schwarzschild = True
use_lense_thirring = True
use_de_sitter = True
# Ref: G. Petit and B. Luzum. IERS Conventions (2010), IERS Technical Note No. 36. Verlag
# des Bundesamts für Kartographie und Geodäsie, Frankfurt am Main, Germany, 2010.
# ISBN 3-89888-989-6, 2010. 44, 50, 51, 55
earth_lense_thirring_angular_momentum = np.array([0.0, 0.0, 9.80e8])  # [m^2/s], in the global frame

guidance_model = Guidance(
    bodies=bodies,
    controlled_satellite="GRACE C",
    reference_satellite="GRACE D",
    target_range=target_range,
    n_revolutions=1,
    distance_threshold=5e3,
    cooldown_duration=3600.0,
)


# Define accelerations acting on the GRACE-FO satellites
acceleration_settings_grace_d = {
    "Earth": [dynamics.propagation_setup.acceleration.relativistic_correction(
                use_schwarzschild,
                use_lense_thirring,
                use_de_sitter,
                de_sitter_central_body="Sun",
                lense_thirring_angular_momentum=np.array([0.0, 0.0, 9.80e8]), # 
           ),
           dynamics.propagation_setup.acceleration.spherical_harmonic_gravity(20, 20),  # Default Max Degree: 200, Max Order: 200
           dynamics.propagation_setup.acceleration.aerodynamic(),
           ],
    "Sun": [dynamics.propagation_setup.acceleration.spherical_harmonic_gravity(2, 0),
          dynamics.propagation_setup.acceleration.radiation_pressure(),
         ],
    "Moon": [dynamics.propagation_setup.acceleration.spherical_harmonic_gravity(2, 0)],    # Default Max Degree: 200, Max Order: 200
    "Mars": [dynamics.propagation_setup.acceleration.point_mass_gravity()],                # Default Max Degree: 120, Max Order: 120
    "Venus": [dynamics.propagation_setup.acceleration.point_mass_gravity()],               # Default Max Degree: 180, Max Order: 180   
    "Mercury": [dynamics.propagation_setup.acceleration.point_mass_gravity()],             # Default Max Degree: 160, Max Order: 160
    "Jupiter": [dynamics.propagation_setup.acceleration.point_mass_gravity()],             # Zonal coefficients up to degree 8
    "Saturn": [dynamics.propagation_setup.acceleration.point_mass_gravity()],                           
    "Uranus": [dynamics.propagation_setup.acceleration.point_mass_gravity()],
    "Neptune": [dynamics.propagation_setup.acceleration.point_mass_gravity()],
    "Ceres": [dynamics.propagation_setup.acceleration.point_mass_gravity()],
    "Vesta": [dynamics.propagation_setup.acceleration.point_mass_gravity()],
    "Pluto": [dynamics.propagation_setup.acceleration.point_mass_gravity()],
}

# Define accelerations acting on the GRACE-FO satellites
acceleration_settings_grace_c = {
    "Earth": [dynamics.propagation_setup.acceleration.relativistic_correction(
            use_schwarzschild,
            use_lense_thirring,
            use_de_sitter,
            de_sitter_central_body="Sun",
        ),
        dynamics.propagation_setup.acceleration.spherical_harmonic_gravity(20, 20),  # Default Max Degree: 200, Max Order: 200
        dynamics.propagation_setup.acceleration.aerodynamic(),
        ],
    "Sun": [dynamics.propagation_setup.acceleration.spherical_harmonic_gravity(2, 0),
        dynamics.propagation_setup.acceleration.radiation_pressure(),
        ],
    "Moon": [dynamics.propagation_setup.acceleration.spherical_harmonic_gravity(2, 0)],    # Default Max Degree: 200, Max Order: 200
    "Mars": [dynamics.propagation_setup.acceleration.point_mass_gravity()],                # Default Max Degree: 120, Max Order: 120
    "Venus": [dynamics.propagation_setup.acceleration.point_mass_gravity()],               # Default Max Degree: 180, Max Order: 180   
    "Mercury": [dynamics.propagation_setup.acceleration.point_mass_gravity()],             # Default Max Degree: 160, Max Order: 160
    "Jupiter": [dynamics.propagation_setup.acceleration.point_mass_gravity()],             # Zonal coefficients up to degree 8
    "Saturn": [dynamics.propagation_setup.acceleration.point_mass_gravity()],                           
    "Uranus": [dynamics.propagation_setup.acceleration.point_mass_gravity()],
    "Neptune": [dynamics.propagation_setup.acceleration.point_mass_gravity()],
    "Ceres": [dynamics.propagation_setup.acceleration.point_mass_gravity()],
    "Vesta": [dynamics.propagation_setup.acceleration.point_mass_gravity()],
    "Pluto": [dynamics.propagation_setup.acceleration.point_mass_gravity()],
}


acceleration_settings = {"GRACE C": acceleration_settings_grace_c, "GRACE D": acceleration_settings_grace_d}

# Create acceleration models
acceleration_models = dynamics.propagation_setup.create_acceleration_models(
   bodies, acceleration_settings, bodies_to_propagate, central_bodies)

propagator_type = dynamics.propagation_setup.propagator.cowell

earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter

dependent_variables_to_save = [
    dependent_variable.single_acceleration_norm(
        dynamics.propagation_setup.acceleration.radiation_pressure_type,  "GRACE C", "Sun",
    ),
    dependent_variable.single_acceleration_norm(
        dynamics.propagation_setup.acceleration.radiation_pressure_type,  "GRACE D", "Sun",
    ),
    dependent_variable.single_acceleration(
        dynamics.propagation_setup.acceleration.radiation_pressure_type,  "GRACE C", "Sun",
    ),
    dependent_variable.single_acceleration(
        dynamics.propagation_setup.acceleration.radiation_pressure_type,  "GRACE D", "Sun",
    ),
    dependent_variable.single_acceleration_norm(
        dynamics.propagation_setup.acceleration.aerodynamic_type, "GRACE C", "Earth"
    ),
    dependent_variable.single_acceleration_norm(
        dynamics.propagation_setup.acceleration.aerodynamic_type, "GRACE D", "Earth"
    ),
    dependent_variable.single_acceleration(
        dynamics.propagation_setup.acceleration.aerodynamic_type, "GRACE C", "Earth"
    ),
    dependent_variable.single_acceleration(
        dynamics.propagation_setup.acceleration.aerodynamic_type, "GRACE D", "Earth"
    ),
    dependent_variable.inertial_to_body_fixed_rotation_frame("GRACE C"),
    dependent_variable.inertial_to_body_fixed_rotation_frame("GRACE D"),
    dependent_variable.keplerian_state("GRACE C", "Earth"),  
    dependent_variable.keplerian_state("GRACE D", "Earth"),
    ]

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
        continue

    termination_details = dynamics_simulator.propagation_results.termination_details
    termination_flags = list(
        getattr(termination_details, "was_condition_met_when_stopping", [])
    )

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

states_array = restructure_vector_history(np.vstack(state_history_propagation_segments))
dependent_variables_array = restructure_vector_history(
    np.vstack(dependent_variable_history_propagation_segments)
)

# =====================================
# PLOTTING GRACE-FO RELATED DATA
# =====================================

time_data = states_array[:, 0]
grace_fo_position_data = [
    states_array[:, 1:4],   # GRACE C position
    states_array[:, 7:10],  # GRACE D position
]
grace_fo_velocity_data = [
    states_array[:, 4:7],    # GRACE C velocity
    states_array[:, 10:13],  # GRACE D velocity
]


plotter.plot_keplerian_elements_difference_time_evolution(
    time_data,
    dependent_variables_array
)

plotter.plot_orbits(
    no_satellites=2,
    position_data=grace_fo_position_data,
    title="GRACE C and GRACE D orbits",
    sat_labels=["GRACE C", "GRACE D"],
    file_name="grace_fo_nominal_orbits.png"
)

plotter.plot_relative_position(
   time_data=time_data,
   position_data=grace_fo_position_data,
   velocity_data=grace_fo_velocity_data,
   first_figure_title="RTN relative position components time evolution - GRACE-FO",
   first_file_name="grace_fo_rtn_relative_position_components.png",
   second_figure_title="RTN Relative position time evolution - GRACE-FO",
   second_file_name="grace_fo_3d_rtn_relative_position.png",
)

plotter.plot_srp_acceleration_time_series(
    dependent_variables_array=dependent_variables_array,
)

plotter.plot_aerodynamic_acceleration_time_series(
    dependent_variables_array=dependent_variables_array,
)

plotter.plot_impulsive_delta_v_time_series(
    maneuver_log=guidance_log,
    simulation_start_epoch=simulation_start_epoch,
)

plotter.plot_attitude_triads_orientation(
    dependent_variables_array=dependent_variables_array,
    grace_fo_position_data=grace_fo_position_data,
    epoch_idx=100,
    file_name="grace_attitude_triads_epoch_100.png"
)


# =====================================
# GPS POSITION MEASUREMENT SIMULATION 
# =====================================

if noise_model_version == 1:
    sigma_gps_position_rtn = np.array([0.01, 0.01, 0.01])*np.sqrt(1/(2*time_step))  # [R, T, N] in meters
    relative_position_error_asd_json_path=None
else:
    sigma_gps_position_rtn = None
    relative_position_error_asd_json_path=Path("data/relative_position_error_asd_data.json")

    
eci_gps_position_noise_grace_c, rtn_gps_position_noise_grace_c = NoiseGenerator.generate_gps_position_noise(
    plotter=plotter,
    num_epochs=number_epochs,
    state_vector=states_array[:, 1:7],  # GRACE C state
    sigma_rtn=sigma_gps_position_rtn,
    seed=80,
    satellite_name="grace_c",
    relative_position_error_asd_json_path=relative_position_error_asd_json_path,
    noise_model_version=noise_model_version,
)

eci_gps_position_noise_grace_d, rtn_gps_position_noise_grace_d = NoiseGenerator.generate_gps_position_noise(
    plotter=plotter,
    num_epochs=number_epochs,
    state_vector=states_array[:, 7:13],  # GRACE D state
    sigma_rtn=sigma_gps_position_rtn,
    seed=90,
    satellite_name="grace_d",
    relative_position_error_asd_json_path=relative_position_error_asd_json_path,
    noise_model_version=noise_model_version,
)
    
eci_gps_position_noise = {
    "GRACE C": eci_gps_position_noise_grace_c,
    "GRACE D": eci_gps_position_noise_grace_d,
}


# =====================================
# KBR RANGE MEASUREMENT SIMULATION 
# =====================================

# Generate KBR system and oscillator noise time series for each satellite
kbr_system_and_oscillator_noise_timeseries = NoiseGenerator.generate_kbr_system_and_oscillator_noise(
   plotter,
   states_array.shape[0],
   seed=42,
   noise_model_version=noise_model_version,
)

antenna_phase_center_offset_vector_sf = {       
      "GRACE C": np.array([1.4437, -370.6e-6, 145e-6]),  # [m]
      "GRACE D": np.array([1.4817, 183e-6, 1393.1e-6]),  # [m]
}


standard_deviation_guessed_antenna_phase_center_offset_vector_sf = {
    "GRACE C": np.array([1.2e-3, 64.7e-6, 65.4e-6]),
    "GRACE D": np.array([1.3e-3, 65.9e-6, 71.6e-6])
}  


bias_value = 2e-2  # KBR range bias in meters

kbr_range_noise = NoiseGenerator.generate_kbr_range_noise(
    error_free_pointing_angles_time_series=error_free_pointing_angles_time_series,
    noisy_attitude_time_series=noisy_attitude_time_series,
    kbr_system_and_oscillator_noise_timeseries=kbr_system_and_oscillator_noise_timeseries,
    position_data=grace_fo_position_data,
    eci_gps_position_noise=eci_gps_position_noise,
    antenna_phase_center_offset_vector_sf=antenna_phase_center_offset_vector_sf,
    standard_deviation_guessed_antenna_phase_center_offset_vector_sf=standard_deviation_guessed_antenna_phase_center_offset_vector_sf,
    bias_value=bias_value,
    plotter=plotter,
    )
