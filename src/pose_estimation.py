import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from src.utils import invert_transform, get_yaw
from src.frame import Frame
from src.visualize import plot_icp

# Function to estimate the relative pose using solvePnP
def estimate_relative_pose(q_frame: Frame, t_frame: Frame, threshold=0.1, debug=False):
    """
    Estimate the relative pose between two frames using matched keypoints and depth information.

    solvePnP: Estimates the pose of a 3D object given a set of 3D points in the object coordinate space and their corresponding 2D projections in the image plane. 
    solvePnP Parameters:
        - prev_points (np.ndarray): Depth map of the previous frame
        - prev_points (np.ndarray): Depth map of the previous frame
        - K (np.ndarray): Camera intrinsic matrix 
    solvePnP Returns:
        success: A boolean flag indicating if the function successfully found a solution.
        rvec: The is a Rodrigues rotation vector representing the rotation of the object in the camera coordinate system.
        tvec: The vector that represents the translation of the object in the camera coordinate system.

    Returns:
        - pose or None: The new pose as a 4x4 transformation matrix
    """
    print(f"Estimating relative pose using ICP...")

    # Run ICP to find the best-fit transformation
    icp_result = o3d.pipelines.registration.registration_icp(
        source = q_frame.pcd_down,
        target = t_frame.pcd_down,
        max_correspondence_distance=threshold,
        # estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    T = icp_result.transformation.copy()

    if icp_result.fitness > 0.0:
        if debug:
            print(f"\tICP Fitness: {icp_result.fitness:.3f}, Inlier RMSE: {icp_result.inlier_rmse:.3f}")
            
        return T
    else:
        print("\tWarning: ICP failed to converge")
        return None

def check_velocity(T: np.ndarray, dt: float):
    """Velocity sanity check"""
    T = invert_transform(T)
    
    Rot = T[:3, :3]
    t = T[:, 3]

    # Check if the estimated velocity makes sense
    dist = np.linalg.norm(t)
    velocity = dist / dt
    if abs(velocity) > 7.5:
        print(f"\tWarning: The velocity ({velocity:.2f}) is an outlier due to bad correspondences!")
        return False

    # Check if the estimated yaw rate makes sense
    yaw = get_yaw(Rot)
    yaw_rate = yaw/dt
    if abs(yaw_rate) > 30:
        print(f"\tWarning: The yaw rate ({yaw_rate:.2f}) is an outlier due to bad correspondences!")
        return False
    
    return True

def is_keyframe(T: np.ndarray, t_threshold=0.5, angle_threshold=10, debug=False):
    """ Determine if motion expressed by t, R is significant by comparing to tresholds. """
    tx = T[0, 3] # The x component points right
    ty = T[1, 3] # The y component points down
    tz = T[2, 3] # The z component points forward

    trans = np.sqrt(tx**2 + ty**2 + tz**2)
    rvec = R.from_matrix(T[:3, :3]).as_rotvec()
    angle = np.degrees(np.linalg.norm(rvec))  # angle in degrees

    is_keyframe = trans > t_threshold or angle > angle_threshold
    if debug:
        print(f"\tDisplacement: dist: {trans:.3f}, angle: {angle:.3f}")
        if is_keyframe:
            print("\t\tKeyframe!")
        else:
            print("\t\tNot a keyframe!")

    return is_keyframe