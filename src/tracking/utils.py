import numpy as np
from scipy.linalg import expm, logm
import src.utils as utils
import src.globals as ctx


def constant_velocity_model(t_frame: utils.Frame) -> np.ndarray:
    """Predicts the next pose assuming constant velocity between the last 3 frames"""
    # Extract previous frames from map
    prev_frames = ctx.map.keyframes
    prev_frames_list = list(prev_frames.values())
    preprev_frame = prev_frames_list[-2]
    prev_frame = prev_frames_list[-1]

    # Extract the timestamp of the current frame
    t = t_frame.time

    # Find how much time has passed
    dt_c = t - prev_frame.time

    # Find the previous dr
    dt_t2q = prev_frame.time - preprev_frame.time

    # Find the previous relative transformation
    T_w2q = np.linalg.inv(preprev_frame.pose)
    T_t2w = prev_frame.pose
    T_t2q = T_w2q @ T_t2w

    # Use the matrix logarithm to obtain the twist (in se(3)) corresponding to T_rel.
    twist_matrix = logm(T_t2q)
    
    # Scale the twist for the prediction time interval.
    scaled_twist = twist_matrix * (dt_c / dt_t2q)
    
    # Obtain the predicted incremental transformation via the matrix exponential.
    T_c2t = expm(scaled_twist)
    
    # The predicted current pose is T_last followed by the predicted incremental transformation.
    T_c2w = T_t2w @ T_c2t

    return T_c2w
