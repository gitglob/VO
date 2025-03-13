import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from src.utils import keypoints_depth_to_3d_points, invert_transform


# Function to estimate the relative pose using solvePnP
def estimate_relative_pose(matches, q_frame, t_frame, K, dist_coeffs=None, debug=False):
    """
    Estimate the relative pose between two frames using matched keypoints and depth information.

    solvePnP: Estimates the pose of a 3D object given a set of 3D points in the object coordinate space and their corresponding 2D projections in the image plane. 
    solvePnP Parameters:
        - q_frame (list): Current frame 
        - t_frame (list): Previous frame 
        - K (np.ndarray): Camera intrinsic matrix 

    solvePnP Returns the world->camera transformation:
        success: A boolean flag indicating if the function successfully found a solution.
        rvec: The is a Rodrigues rotation vector representing the rotation of the object in the camera coordinate system.
        tvec: The vector that represents the translation of the object in the camera coordinate system.

    Returns:
        - pose or None: The new pose as a 4x4 transformation matrix
    """
    if debug:
        print(f"Estimating pose using frames {q_frame.id} & {t_frame.id}...")
    T = np.eye(4)

    # Extract matched keypoints' coordinates
    q_kpt_pixels = np.float64([q_frame.keypoints[m.queryIdx].pt for m in matches])
    t_kpt_pixels = np.float64([t_frame.keypoints[m.trainIdx].pt for m in matches])

    # Convert the keypoints to 3D coordinates using the depth map
    q_pts_3d, depth_mask = keypoints_depth_to_3d_points(q_kpt_pixels, q_frame.depth, 
                                                        cx=K[0, 2], cy=K[1, 2], 
                                                        fx=K[0, 0], fy=K[1, 1])
    if debug:
        print(f"\tDepth filtered {len(q_kpt_pixels) - np.sum(depth_mask)}/{len(q_kpt_pixels)} points.")
    
    # Keep only the pixels that have a corresponding 3D point
    t_kpt_pixels = t_kpt_pixels[depth_mask]
    
    # Check if enough 3D points exist to estimate the camera pose using the Direct Linear Transformation (DLT) algorithm
    if len(q_pts_3d) < 6:
        print(f"\tWarning: Not enough points for pose estimation. Got {len(q_pts_3d)}, expected at least 6.")
        return None, None
    
    # Use solvePnP to estimate the pose
    success, rvec, tvec, inliers = cv2.solvePnPRansac(q_pts_3d, t_kpt_pixels, 
                                                      cameraMatrix=K, distCoeffs=dist_coeffs, 
                                                      reprojectionError=0.2, confidence=0.999, 
                                                      iterationsCount=5000)
    if not success or np.linalg.norm(rvec) > 50 or np.linalg.norm(tvec) > 1:
        print("\tWarning: solvePnP failed!")
        return None, None

    inliers = inliers.flatten()
    inliers_mask = np.zeros(len(q_pts_3d), dtype=bool)
    inliers_mask[inliers] = True
    if debug:
        print(f"\tsolvePnPRansac filtered {len(q_pts_3d) - np.sum(inliers_mask)}/{len(q_pts_3d)} points.")
    
    # Keep only the pixels and points that match the estimated pose
    t_kpt_pixels = t_kpt_pixels[inliers_mask]
    q_pts_3d = q_pts_3d[inliers_mask]

    # Compute reprojection error and print it
    reprojection_mask, errors = compute_reprojection_error(q_pts_3d, t_kpt_pixels, rvec, tvec, K, dist_coeffs)
    error = np.mean(errors)
    if debug:
        print(f"\tReprojection error ({error:.2f}) filtered {len(q_pts_3d) - np.sum(reprojection_mask)}/{len(q_pts_3d)} points.")

    # Filter the points using the mask
    q_pts_3d = q_pts_3d[reprojection_mask]
    t_kpt_pixels = t_kpt_pixels[reprojection_mask]

    # Convert the refined rotation vector to a rotation matrix
    Rot, _ = cv2.Rodrigues(rvec)

    # Construct the refined pose matrix
    T[:3, :3] = Rot
    T[:3, 3] = tvec.flatten()

    # Convert the camera coordinates to the world coordinates
    T_inv = invert_transform(T)

    # Create a final mask that maps back to the original matches.
    indices = np.arange(len(matches))
    indices_after_depth = indices[depth_mask]
    indices_after_inliers = indices_after_depth[inliers_mask]
    indices_after_reproj = indices_after_inliers[reprojection_mask]

    # Filtered matches
    filtered_matches = [matches[i] for i in indices_after_reproj]

    q_frame.set_matches(t_frame.id, filtered_matches, "query")
    t_frame.set_matches(q_frame.id, filtered_matches, "train")

    return T_inv, error
    
def compute_reprojection_error(pts_3d, pts_2d, rvec, tvec, K, dist_coeffs=None, threshold=2.0):
    """Compute the reprojection error for the given 3D-2D point correspondences and pose."""
    # Project the 3D points to 2D using the estimated pose
    projected_pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist_coeffs)
    projected_pts_2d = projected_pts_2d.squeeze()
    
    # Calculate the per-point reprojection error (Euclidean distance)
    errors = np.sqrt(np.sum((pts_2d - projected_pts_2d)**2, axis=1))
    
    # Create a mask for points with error less than the threshold
    mask = errors < threshold
    return mask, errors

def is_keyframe(T: np.ndarray, t_threshold=0.5, angle_threshold=15, debug=False):
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
