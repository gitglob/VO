import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from src.utils import keypoints_depth_to_3d_points, invert_transform


# Function to estimate the relative pose using solvePnP
def estimate_relative_pose(q_frame, t_frame, K, dist_coeffs=None, debug=False):
    """
    Estimate the relative pose between two frames using matched keypoints and depth information.

    solvePnP: Estimates the pose of a 3D object given a set of 3D points in the object coordinate space and their corresponding 2D projections in the image plane. 
    solvePnP Parameters:
        - cur_keypts (list): List of keypoints in the current frame 
        - prev_keypts (list): List of keypoints in the previous frame 
        - prev_depth (np.ndarray): Depth map of the previous frame
        - K (np.ndarray): Camera intrinsic matrix 
    solvePnP Returns:
        success: A boolean flag indicating if the function successfully found a solution.
        rvec: The is a Rodrigues rotation vector representing the rotation of the object in the camera coordinate system.
        tvec: The vector that represents the translation of the object in the camera coordinate system.

    Returns:
        - pose or None: The new pose as a 4x4 transformation matrix
    """
    if debug:
        print(f"Estimating pose using frames {q_frame.id} & {t_frame.id}...")

    matches = q_frame.get_matches(t_frame.id)
    T = np.eye(4)

    # Extract matched keypoints' coordinates
    prev_kpt_pixels = np.float64([q_frame.keypoints[m.queryIdx].pt for m in matches])
    cur_kpt_pixels = np.float64([t_frame.keypoints[m.trainIdx].pt for m in matches])

    # Convert the keypoints to 3D coordinates using the depth map
    prev_pts_3d, indices = keypoints_depth_to_3d_points(prev_kpt_pixels, q_frame.depth, 
                                                        cx=K[0, 2], cy=K[1, 2], 
                                                        fx=K[0, 0], fy=K[1, 1])
    
    # Keep only the pixels that have a corresponding 3D point
    cur_kpt_pixels = cur_kpt_pixels[indices]
    
    # Check if enough 3D points exist to estimate the camera pose using the Direct Linear Transformation (DLT) algorithm
    if len(prev_pts_3d) < 6:
        print(f"\tWarning: Not enough points for pose estimation. Got {len(prev_pts_3d)}, expected at least 6.")
        return None, None
    
    # Use solvePnP to estimate the pose
    success, rvec, tvec, inliers = cv2.solvePnPRansac(prev_pts_3d, cur_kpt_pixels, 
                                                      cameraMatrix=K, distCoeffs=dist_coeffs, 
                                                      reprojectionError=0.2, confidence=0.999, 
                                                      iterationsCount=5000)
    inliers = inliers.flatten()
    inliers_mask = np.zeros(len(prev_pts_3d), dtype=bool)
    inliers_mask[inliers] = True
    if debug:
        print(f"\tsolvePnPRansac filtered {len(prev_pts_3d) - np.sum(inliers_mask)}/{len(prev_pts_3d)} points.")
    
    # Keep only the pixels and points that match the estimated pose
    cur_kpt_pixels = cur_kpt_pixels[inliers]
    prev_pts_3d = prev_pts_3d[inliers]

    # Compute reprojection error and print it
    reprojection_mask, errors = compute_reprojection_error(prev_pts_3d, cur_kpt_pixels, rvec, tvec, K, dist_coeffs)
    if debug:
        print(f"\tReprojection error filtered {len(prev_pts_3d) - np.sum(reprojection_mask)}/{len(prev_pts_3d)} points.")
        print(f"\tReprojection error: {np.mean(errors[reprojection_mask]):.2f} pixels")

    # Filter the points using the mask
    prev_pts_3d = prev_pts_3d[reprojection_mask]
    cur_kpt_pixels = cur_kpt_pixels[reprojection_mask]

    if success:
        # Refine pose using inliers by calling solvePnP again without RANSAC
        success_refine, rvec_refined, tvec_refined = cv2.solvePnP(prev_pts_3d, cur_kpt_pixels, 
                                                                  K, dist_coeffs, rvec, tvec, useExtrinsicGuess=True)

        if success_refine:
            # Compute the refined reprojection error
            _, errors = compute_reprojection_error(prev_pts_3d, cur_kpt_pixels, rvec_refined, tvec_refined, K, dist_coeffs)
            error_refined = np.mean(errors)
            if debug:
                print(f"\tRefined reprojection error: {error_refined:.2f} pixels")

            # Convert the refined rotation vector to a rotation matrix
            R_refined, _ = cv2.Rodrigues(rvec_refined)

            # Convert the camera coordinates to the world coordinates
            tvec_refined = tvec_refined.flatten()
            tx = tvec_refined[0]
            ty = tvec_refined[2]

            # Construct the refined pose matrix
            T[:3, :3] = R_refined
            T[:3, 3] = tvec_refined.flatten()
            T_inv = invert_transform(T)

            return T_inv, error_refined
        else:
            return None, None
    else:
        return None, None
    
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

def is_keyframe(T: np.ndarray, t_threshold=1, angle_threshold=3, debug=False):
    """ Determine if motion expressed by t, R is significant by comparing to tresholds. """
    tx = T[0, 3]
    ty = T[1, 3]

    trans = np.sqrt(tx**2 + ty**2)
    rpy = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
    yaw = abs(rpy[2])

    is_keyframe = trans > t_threshold or yaw > angle_threshold
    if True:#debug:
        print(f"\tDisplacement: dist: {trans:.3f}, angle: {yaw:.3f}")
        if is_keyframe:
            print("\t\tKeyframe!")
        else:
            print("\t\tNot a keyframe!")

    return is_keyframe
