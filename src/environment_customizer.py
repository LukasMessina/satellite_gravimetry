""" Environment Customizer Module """

import numpy as np
from scipy.spatial.transform import Rotation
from tudatpy.dynamics import environment

class EnvironmentCustomizer:
    """Provides custom environment models used to build the simulation.

    Currently supplies a custom attitude/rotation model driven by pre-generated
    pointing-angle time series, and a helper to apply impulsive delta-v maneuvers
    to a stacked multi-body state vector.
    """

    @staticmethod
    def create_custom_spacecraft_rotation_function(
        spacecraft_name: str,
        counterpart_name: str,
        error_free_attitude_history: dict[str, np.ndarray],
        sample_times: np.ndarray,
        rotation_model_context: dict[str, environment.SystemOfBodies],
    ):
        """Build a custom rotation-matrix callback for one spacecraft.

        The returned callback computes, at any propagation time, the rotation
        from the spacecraft frame (SF) to the global J2000 frame, as the product
        of two rotations: (1) SF-to-Line-of-Sight-Frame (LOSF), driven by
        interpolating the given yaw/pitch/roll pointing-angle history, and
        (2) LOSF-to-J2000, derived geometrically from both satellites' current
        positions (the LOSF x-axis points from this spacecraft toward its
        counterpart).

        Args:
            spacecraft_name (str): Name of the body this rotation model is for.
            counterpart_name (str): Name of the other satellite in the pair,
                used to define the line-of-sight direction.
            error_free_attitude_history (dict[str, np.ndarray]): Dict with keys
                "yaw", "pitch", "roll", each an array of angles [rad] sampled at
                sample_times.
            sample_times (np.ndarray): Epochs at which the yaw/pitch/roll arrays
                are sampled, used for interpolation.
            rotation_model_context (dict[str, environment.SystemOfBodies]): A
                mutable dict expected to hold a "bodies" key once the
                SystemOfBodies environment is created; deferred population is necessary
                because the environment does not exist yet when this rotation
                model is first attached to the body settings.

        Returns:
            Callable[[float], np.ndarray]: A function mapping a propagation
                time to a 3x3 rotation matrix from the spacecraft frame to
                J2000, suitable for the custom_rotation_model.
        """

        yaw_history = np.asarray(error_free_attitude_history["yaw"], dtype=float)
        pitch_history = np.asarray(error_free_attitude_history["pitch"], dtype=float)
        roll_history = np.asarray(error_free_attitude_history["roll"], dtype=float)

        def custom_rotation_matrix_function(current_time: float) -> np.ndarray:
            """Compute the spacecraft-frame-to-J2000 rotation matrix at current_time.

            Args:
                current_time (float): Propagation epoch at which to evaluate the
                    rotation.

            Returns:
                np.ndarray: 3x3 rotation matrix from the spacecraft frame to
                    J2000.
            """

            yaw   = float(np.interp(current_time, sample_times, yaw_history))
            pitch = float(np.interp(current_time, sample_times, pitch_history))
            roll  = float(np.interp(current_time, sample_times, roll_history))

            rot_matrix_sf_to_losf = Rotation.from_euler("ZYX", [yaw, pitch, roll], degrees=False).as_matrix()

            # NOTE: "bodies" is only populated after create_system_of_bodies runs, which
            # happens after this rotation model is attached to the body settings;
            # rotation_model_context is the indirection that lets it be filled in later.
            bodies = rotation_model_context.get("bodies")

            primary_position = np.asarray(bodies.get(spacecraft_name).state[:3], dtype=float)
            secondary_position = np.asarray(bodies.get(counterpart_name).state[:3], dtype=float)

            # Compute LOSF to J2000matrix for both satellites
            x_losf = (secondary_position - primary_position) / np.linalg.norm(secondary_position \
                                                                              - primary_position, axis=-1, keepdims=True)


            y_losf = np.cross(x_losf, primary_position) / np.linalg.norm(np.cross(x_losf, primary_position), axis=-1, keepdims=True)


            z_losf = np.cross(x_losf, y_losf) / np.linalg.norm(np.cross(x_losf, y_losf), axis=-1, keepdims=True)
            rot_matrix_losf_to_j2000 = np.stack([x_losf, y_losf, z_losf], axis=-1)


            return rot_matrix_losf_to_j2000 @ rot_matrix_sf_to_losf

        return custom_rotation_matrix_function

    @staticmethod
    def apply_impulsive_velocity_deviation_to_state_vector(
            state_vector: np.ndarray,
            body_index: int,
            delta_v_vector: np.ndarray,
        ) -> np.ndarray:
            """Apply an instantaneous velocity change to one body in a stacked state vector.

            Args:
                state_vector (np.ndarray): Stacked Cartesian state vector for
                    multiple bodies, laid out as 6 values (position + velocity)
                    per body, in body order.
                body_index (int): Zero-based index of the body within
                    state_vector to apply the delta-v to.
                delta_v_vector (np.ndarray): 3-element velocity change to add,
                    in the same frame as state_vector's velocity components.

            Returns:
                np.ndarray: A new state vector, identical to state_vector except
                    with delta_v_vector added to the targeted body's velocity
                    components.
            """
            updated_state_vector = np.asarray(state_vector, dtype=float).copy()
            velocity_start_index = 6 * body_index + 3
            updated_state_vector[velocity_start_index:velocity_start_index + 3] += np.asarray(
                delta_v_vector,
                dtype=float,
            )
            return updated_state_vector