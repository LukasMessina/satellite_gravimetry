""" Environment Customizer Module """

import numpy as np
from scipy.spatial.transform import Rotation
from tudatpy.dynamics import environment

class EnvironmentCustomizer:
    """Customizes the environment for simulations."""

    @staticmethod
    def create_custom_spacecraft_rotation_function(
        spacecraft_name: str,
        counterpart_name: str,
        attitude_noise_history: dict[str, np.ndarray],
        sample_times: np.ndarray,
        rotation_model_context: dict[str, environment.SystemOfBodies],
    ):
        """
        Create satellite frame to J2000 Earth centered reference frame rotation callback
         from Line-of-Sight reference frame to satellite frame noise angles."""

        yaw_history = np.asarray(attitude_noise_history["yaw"], dtype=float)
        pitch_history = np.asarray(attitude_noise_history["pitch"], dtype=float)
        roll_history = np.asarray(attitude_noise_history["roll"], dtype=float)

        def custom_rotation_matrix_function(current_time: float) -> np.ndarray:

            yaw   = float(np.interp(current_time, sample_times, yaw_history))
            pitch = float(np.interp(current_time, sample_times, pitch_history))
            roll  = float(np.interp(current_time, sample_times, roll_history))            

            rot_matrix_sf_to_losf = Rotation.from_euler("ZYX", [yaw, pitch, roll], degrees=False).as_matrix()

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
            updated_state_vector = np.asarray(state_vector, dtype=float).copy()
            velocity_start_index = 6 * body_index + 3
            updated_state_vector[velocity_start_index:velocity_start_index + 3] += np.asarray(
                delta_v_vector,
                dtype=float,
            )
            return updated_state_vector