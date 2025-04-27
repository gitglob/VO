import numpy as np
from scipy.linalg import expm, logm
import src.utils as utils


def constant_velocity_model(t: float, frames: list):
    """Predicts the next pose assuming constant velocity between the last 3 frames"""
    frames_list = list(frames.values())

    # Find how much time has passed
    dt_c = t - frames_list[-1].time

    # Find the previous dr
    dt_tq = frames_list[-1].time - frames_list[-2].time

    # Find the previous relative transformation
    T_wq = np.linalg.inv(frames_list[-2].pose)
    T_tw = frames_list[-1].pose
    T_tq = T_wq @ T_tw

    # Use the matrix logarithm to obtain the twist (in se(3)) corresponding to T_rel.
    twist_matrix = logm(T_tq)
    
    # Scale the twist for the prediction time interval.
    scaled_twist = twist_matrix * (dt_c / dt_tq)
    
    # Obtain the predicted incremental transformation via the matrix exponential.
    T_ct = expm(scaled_twist)
    
    # The predicted current pose is T_last followed by the predicted incremental transformation.
    T_cw = T_tw @ T_ct
    T_wc = utils.invert_transform(T_cw)

    return T_wc