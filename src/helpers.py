import numpy as np

def wrap_deg(x: float) -> float:
    """Wrap an angle into the range [0, 360) degrees.

    Args:
        x (float): Angle in degrees, any range.

    Returns:
        float: x wrapped into [0, 360).
    """
    return x % 360.0

def wrap_rad(x: float) -> float:
    """Wrap an angle into the range [0, 2*pi) radians.

    Args:
        x (float): Angle in radians, any range.

    Returns:
        float: x wrapped into [0, 2*pi).
    """
    return x % (2.0 * np.pi)

def rtn_basis(r, v):
    """Build a single-epoch Radial-Transverse-Normal (RTN) basis from position/velocity.

    Args:
        r: 3-element position vector.
        v: 3-element velocity vector.

    Returns:
        np.ndarray: 3x3 matrix whose columns are the unit radial, transverse
            (along-track), and normal (orbit-plane-normal) basis vectors.
    """
    Rhat = r / np.linalg.norm(r)
    h = np.cross(r, v); Nhat = h / np.linalg.norm(h)
    That = np.cross(Nhat, Rhat)
    return np.column_stack((Rhat, That, Nhat))

def transform_vector_history_inertial_to_satellite_frame(
    vectors_inertial: np.ndarray,
    rot_inertial_to_sf_flat: np.ndarray,
) -> np.ndarray:
    """Rotate a history of inertial-frame vectors into the satellite frame.

    Args:
        vectors_inertial (np.ndarray): Array reshapeable to (N, 3), one inertial
            3-vector per epoch.
        rot_inertial_to_sf_flat (np.ndarray): Array reshapeable to (N, 3, 3), one
            inertial-to-satellite-frame rotation matrix per epoch, matching
            the inertial vectors in ordering.

    Returns:
        np.ndarray: Array of shape (N, 3), the input vectors expressed in the
            satellite frame at each epoch.
    """
    rot_inertial_to_sf = np.asarray(rot_inertial_to_sf_flat, dtype=float).reshape(-1, 3, 3)
    vectors_inertial = np.asarray(vectors_inertial, dtype=float).reshape(-1, 3)
    return np.einsum("nij,nj->ni", rot_inertial_to_sf, vectors_inertial)

def compute_rtn_basis_history(
    reference_position_history: np.ndarray,
    reference_velocity_history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the RTN basis history from a reference Cartesian state history.

    Args:
        reference_position_history (np.ndarray): (N, 3) array of inertial-frame
            reference positions, one per epoch.
        reference_velocity_history (np.ndarray): (N, 3) array of inertial-frame
            reference velocities, one per epoch, matching
            the reference position history in ordering.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (radial_unit_vector,
            along_track_unit_vector, normal_unit_vector), each an (N, 3) array
            of unit vectors for the corresponding RTN axis at each epoch.

    Raises:
        ValueError: If the inputs are not 2D, differently shaped, or do not
            have exactly 3 columns.
    """

    reference_position_history = np.asarray(reference_position_history, dtype=float)
    reference_velocity_history = np.asarray(reference_velocity_history, dtype=float)

    if reference_position_history.ndim != 2 or reference_velocity_history.ndim != 2:
        raise ValueError("Reference position and velocity histories must be 2D arrays.")
    if reference_position_history.shape != reference_velocity_history.shape:
        raise ValueError("Reference position and velocity histories must have the same shape.")
    if reference_position_history.shape[1] != 3:
        raise ValueError("Reference state histories must have 3 columns.")

    radial_unit_vector = reference_position_history / np.linalg.norm(
        reference_position_history, axis=1, keepdims=True
    )
    angular_momentum = np.cross(reference_position_history, reference_velocity_history)
    normal_unit_vector = angular_momentum / np.linalg.norm(
        angular_momentum, axis=1, keepdims=True
    )
    along_track_unit_vector = np.cross(normal_unit_vector, radial_unit_vector)
    along_track_unit_vector = along_track_unit_vector / np.linalg.norm(
        along_track_unit_vector, axis=1, keepdims=True
    )

    return radial_unit_vector, along_track_unit_vector, normal_unit_vector

def transform_vector_history_inertial_to_rtn(
    vector_history: np.ndarray,
    reference_position_history: np.ndarray,
    reference_velocity_history: np.ndarray,
) -> np.ndarray:
    """Project an inertial vector history onto the RTN frame built from a reference orbit.

    Args:
        vector_history (np.ndarray): (N, 3) array of inertial-frame vectors, one per epoch.
        reference_position_history (np.ndarray): (N, 3) array of inertial-frame
            reference positions defining the RTN frame at each epoch.
        reference_velocity_history (np.ndarray): (N, 3) array of inertial-frame
            reference velocities defining the RTN frame at each epoch.

    Returns:
        np.ndarray: (N, 3) array of vector history expressed in the RTN frame
            (radial, along-track, normal components) at each epoch.

    Raises:
        ValueError: If the vector history is not a 2D array with 3 columns.
    """

    reference_position_history = np.asarray(reference_position_history, dtype=float)
    reference_velocity_history = np.asarray(reference_velocity_history, dtype=float)

    radial_unit_vector = reference_position_history / np.linalg.norm(
        reference_position_history, axis=1, keepdims=True
    )
    angular_momentum = np.cross(reference_position_history, reference_velocity_history)
    normal_unit_vector = angular_momentum / np.linalg.norm(
        angular_momentum, axis=1, keepdims=True
    )
    along_track_unit_vector = np.cross(normal_unit_vector, radial_unit_vector)
    along_track_unit_vector = along_track_unit_vector / np.linalg.norm(
        along_track_unit_vector, axis=1, keepdims=True
    )

    vector_history = np.asarray(vector_history, dtype=float)
    if vector_history.ndim != 2 or vector_history.shape[1] != 3:
        raise ValueError("vector_history must be a 2D array with shape (N, 3).")

    radial_component = np.einsum("ij,ij->i", vector_history, radial_unit_vector)
    along_track_component = np.einsum("ij,ij->i", vector_history, along_track_unit_vector)
    normal_component = np.einsum("ij,ij->i", vector_history, normal_unit_vector)

    return np.column_stack((radial_component, along_track_component, normal_component))


def get_mean_orbital_period(
    dependent_variables_array: np.ndarray,
    gravitational_parameter: float,
) -> tuple[float, float]:
    """Return the mean orbital periods of GRACE C and GRACE D from saved Keplerian histories.

    Args:
        dependent_variables_array (np.ndarray): 2D dependent-variable history
            array, with GRACE C's Keplerian elements in columns 35:41 and
            GRACE D's in columns 41:47 (column 0 is semi-major axis for each).
        gravitational_parameter (float): Central body's (Earth's) gravitational
            parameter.

    Returns:
        tuple[float, float]: (mean_orbital_period_grace_c,
            mean_orbital_period_grace_d), each the time-averaged orbital period
            in seconds, computed from Kepler's third law applied to each
            epoch's semi-major axis.

    Raises:
        ValueError: If the dependent variables array is not 2D, the gravitational parameter
            is not positive, or either satellite's semi-major axis history
            contains non-positive values.
    """

    dependent_variables_array = np.asarray(dependent_variables_array, dtype=float)

    if dependent_variables_array.ndim != 2:
        raise ValueError("dependent_variables_array must be a 2D array.")

    if gravitational_parameter <= 0.0:
        raise ValueError("gravitational_parameter must be positive.")

    # NOTE: Column layout matches the fixed order of dependent_variables_to_save.
    grace_c_keplerian_history = dependent_variables_array[:, 35:41]
    grace_d_keplerian_history = dependent_variables_array[:, 41:47]

    semi_major_axis_history_grace_c = grace_c_keplerian_history[:, 0]
    semi_major_axis_history_grace_d = grace_d_keplerian_history[:, 0]

    if np.any(semi_major_axis_history_grace_c <= 0.0) or np.any(semi_major_axis_history_grace_d <= 0.0):
        raise ValueError("Semi-major axis history must contain strictly positive values.")

    orbital_period_history_grace_c = 2.0 * np.pi * np.sqrt(
        semi_major_axis_history_grace_c**3 / gravitational_parameter
    )
    orbital_period_history_grace_d = 2.0 * np.pi * np.sqrt(
        semi_major_axis_history_grace_d**3 / gravitational_parameter
    )

    mean_orbital_period_grace_c = float(np.mean(orbital_period_history_grace_c))
    mean_orbital_period_grace_d = float(np.mean(orbital_period_history_grace_d))

    return mean_orbital_period_grace_c, mean_orbital_period_grace_d


def get_noise_model_version():
    """Prompt the user interactively to select a noise model version.

    Blocks on standard input until the user enters a valid choice.

    Returns:
        int: The selected noise model version, either 1 or 2.
    """
    while True:
        try:
            user_input = int(input(
                "\nSelect Noise Model Version:\n"
                "Enter choice (1 or 2): "
            ))
            if user_input in [1, 2]:
                return user_input
            else:
                print("Invalid input. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter an integer (1 or 2).")


def get_optimal_amplitude_spectral_density_combination(*amplitude_spectral_density_arrays: np.ndarray) -> np.ndarray:
    """Combine multiple amplitude spectral densities into their inverse-variance-optimal estimate.

    Treats each input ASD as an independent noise estimate and combines them
    via inverse-variance weighting, which minimizes the combined estimate's variance
    under the assumption of independent, zero-mean noise sources.

    Args:
        *amplitude_spectral_density_arrays (np.ndarray): One or more ASD arrays
            of matching shape, each representing an independent noise
            contribution at the same frequency bins.

    Returns:
        np.ndarray: The combined ASD array, same shape as each input.

    Raises:
        ValueError: If no arrays are provided.
    """
    if not amplitude_spectral_density_arrays:
        raise ValueError("At least one amplitude spectral density array must be provided.")

    # Converting ASD -> PSD lets independent noise sources combine
    # by simple inverse-variance weighting; converting back to ASD (sqrt) at
    # the end yields the combined amplitude spectral density.
    inverse_power_spectral_densities = [1.0 / (asd_array**2) for asd_array in amplitude_spectral_density_arrays]
    optimal_asd = 1.0 / np.sqrt(np.sum(inverse_power_spectral_densities, axis=0))

    return optimal_asd