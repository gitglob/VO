import numpy as np

from src.utils import rotation_matrix_to_euler_angles

def validate_scale(scaled_poses: list[np.ndarray], gt_poses: list[np.ndarray]):
    """
    Validate that the estimated scaled poses are in the same scale as the ground truth poses.
    """
    pose_diff = np.linalg.norm(scaled_poses[-1][:3, 3] - scaled_poses[-2][:3, 3])
    gt_pose_diff = np.linalg.norm(gt_poses[-1][:3, 3] - gt_poses[-2][:3, 3])
    
    if np.diff([pose_diff, gt_pose_diff]) > 1e-3:
        raise ValueError("Estimated and ground truth poses are not in the same scale.")

def estimate_depth_scale(estimated_poses: list[np.ndarray], gt_poses: list[np.ndarray]):
    """
    Estimate the depth scale factor from two estimated poses using their ground truth counterparts.

    Args:
        estimated_poses (list of np.ndarray): List of two estimated poses (4x4).
        gt_poses (list of np.ndarray): List of two ground truth poses (4x4).

    Returns:
        float: Estimated scale factor.
    """
    print(f"\tEstimating depth scale ...")

    if len(estimated_poses) != 2 or len(gt_poses) != 2:
        raise ValueError("Both input lists must contain exactly two poses.")

    # Extract translations
    est_translation_diff = np.linalg.norm(estimated_poses[1][:3, 3] - estimated_poses[0][:3, 3])
    gt_translation_diff = np.linalg.norm(gt_poses[1][:3, 3] - gt_poses[0][:3, 3])

    if est_translation_diff == 0:
        raise ValueError("Estimated translation difference is zero, cannot estimate scale.")

    # Compute scale
    scale_factor = gt_translation_diff / est_translation_diff
    print(f"\t\tScale factor: {scale_factor:.2f}")

    return scale_factor