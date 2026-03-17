import numpy as np

def wrap_deg(x: float) -> float:
    """Wrap angle to [0, 360)."""
    return x % 360.0

def wrap_rad(x: float) -> float:
    """Wrap angle to [0, 2pi)."""
    return x % (2.0 * np.pi)

def rtn_basis(r, v):
    Rhat = r / np.linalg.norm(r)
    h = np.cross(r, v); Nhat = h / np.linalg.norm(h)
    That = np.cross(Nhat, Rhat)
    return np.column_stack((Rhat, That, Nhat))

def transform_vector_history_inertial_to_satellite_frame(
    vectors_inertial: np.ndarray,
    rot_inertial_to_sf_flat: np.ndarray,
) -> np.ndarray:
    rot_inertial_to_sf = np.asarray(rot_inertial_to_sf_flat, dtype=float).reshape(-1, 3, 3)
    vectors_inertial = np.asarray(vectors_inertial, dtype=float).reshape(-1, 3)
    return np.einsum("nij,nj->ni", rot_inertial_to_sf, vectors_inertial)

def compute_rtn_basis_history(
    reference_position_history: np.ndarray,
    reference_velocity_history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the RTN basis history from a reference Cartesian state history."""

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
    """Project an inertial vector history onto the RTN frame built from a reference orbit."""

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


def get_noise_model_version():
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
