import numpy as np
import cv2
from src.others.visualize import plot_reprojection

from config import results_dir, debug, SETTINGS



############################### Keypoints ###############################

def enforce_epipolar_constraint(q_kpt_pixels, t_kpt_pixels, K):
    # Compute Essential & Homography matrices

    ## Compute the Essential Matrix
    E, mask_E = cv2.findEssentialMat(q_kpt_pixels, t_kpt_pixels, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    mask_E = mask_E.ravel().astype(bool)

    ## Compute the Homography Matrix
    H, mask_H = cv2.findHomography(q_kpt_pixels, t_kpt_pixels, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    mask_H = mask_H.ravel().astype(bool)

    # Compute symmetric transfer errors & decide which model to use

    ## Compute symmetric transfer error for Essential Matrix
    error_E, num_inliers_E = compute_symmetric_transfer_error(E, q_kpt_pixels, t_kpt_pixels, 'E', K=K)

    ## Compute symmetric transfer error for Homography Matrix
    error_H, num_inliers_H = compute_symmetric_transfer_error(H, q_kpt_pixels, t_kpt_pixels, 'H', K=K)

    ## Decide which matrix to use based on the ratio of inliers
    if debug:
        print(f"\t Inliers E: {num_inliers_E}, Inliers H: {num_inliers_H}")
    if num_inliers_E == 0 and num_inliers_H == 0:
        print("All keypoint pairs yield errors > threshold..")
        return None, None, None
    
    ratio = num_inliers_H / (num_inliers_E + num_inliers_H)
    if debug:
        print(f"\t Ratio H/(E+H): {ratio}")

    use_homography = (ratio > 0.45)
    if debug:
        print(f"\t Using {'Homography' if use_homography else 'Essential'} Matrix...")

    epipolar_constraint_mask = mask_H if use_homography else mask_E
    if debug:
        print(f"\t\t E/H inliers filtered {len(q_kpt_pixels) - np.sum(epipolar_constraint_mask)}/{len(q_kpt_pixels)} matches!")

    M = H if use_homography else E

    return epipolar_constraint_mask, M, use_homography

def compute_symmetric_transfer_error(E_or_H, q_kpt_pixels, t_kpt_pixels, matrix_type='E', K=None, threshold=4.0):
    """
    Computes the symmetric transfer error for a set of corresponding keypoints using
    either an Essential matrix or a Homography matrix. This function is used to evaluate 
    how well the estimated transformation aligns the corresponding keypoints between two images.

    For each keypoint pair (one from the target image and one from the query image), 
    the function performs the following steps:
      - Computes the Fundamental matrix F from the provided Essential/Homography matrix and
        the camera intrinsic matrix K. For an Essential matrix, F is computed as:
            F = inv(K.T) @ E_or_H @ inv(K)
        For a Homography, it is computed as:
            F = inv(K) @ E_or_H @ K
      - Converts the keypoints to homogeneous coordinates.
      - Computes the corresponding epipolar lines (l1 in the target image and l2 in the query image).
      - Normalizes these lines using the Euclidean norm of their first two coefficients.
      - Calculates the perpendicular distances from each keypoint to its respective epipolar line.
      - Sums these distances to obtain a symmetric transfer error for the correspondence.

    The function then returns the mean error across all valid keypoint pairs and counts how many
    of these pairs have an error below the specified threshold (i.e., inliers).
    """
    errors = []
    num_inliers = 0

    if matrix_type == 'E':
        F = np.linalg.inv(K.T) @ E_or_H @ np.linalg.inv(K)
    else:
        F = np.linalg.inv(K) @ E_or_H @ K

    # Loop over paired keypoints
    for pt_target, pt_query in zip(t_kpt_pixels, q_kpt_pixels):
        # Convert to homogeneous coordinates
        p1 = np.array([pt_target[0], pt_target[1], 1.0])
        p2 = np.array([pt_query[0], pt_query[1], 1.0])

        # Compute the corresponding epipolar lines
        l2 = F @ p1    # Epipolar line in the query image corresponding to p1
        l1 = F.T @ p2  # Epipolar line in the target image corresponding to p2

        # Normalize the lines using the Euclidean norm of the first two coefficients
        norm_l1 = np.hypot(l1[0], l1[1])
        norm_l2 = np.hypot(l2[0], l2[1])
        if norm_l1 == 0 or norm_l2 == 0:
            continue
        l1_normalized = l1 / norm_l1
        l2_normalized = l2 / norm_l2

        # Compute perpendicular distances (absolute value of point-line dot product)
        d1 = abs(np.dot(p1, l1_normalized))
        d2 = abs(np.dot(p2, l2_normalized))
        error = d1 + d2

        errors.append(error)
        if error < threshold:
            num_inliers += 1

    mean_error = np.mean(errors) if errors else float('inf')
    return mean_error, num_inliers

############################### Triangulation ###############################

def filter_triangulation_points(q_points: np.ndarray, t_points: np.ndarray, 
                                R: np.ndarray, t: np.ndarray):
    """
    Filter out 3D points that:
     1. Lie behind the camera planes
     2. Have a small triangulation angle

    t_points : (N, 3)
    R, t      : the rotation and translation from q_cam to t_cam
    Returns:   valid_angles_mask (bool array of shape (N,)),
               filtered_points_3d (N_filtered, 3) or (None, None)
    """
    print("\t\tFiltering triangulation points...")
    # -----------------------------------------------------
    # (1) Positive-depth check in both cameras
    # -----------------------------------------------------
    num_points = len(t_points)
    triang_mask = np.ones(num_points, dtype=bool)

    # t_points is in the query and train camera => check Z > 0
    Z1 = q_points[:, 2]
    Z2 = t_points[:, 2]

    # We'll mark True if both Z1, Z2 > 0
    cheirality_mask = (Z1 > 0) & (Z2 > 0)
    triang_mask[triang_mask==True] = cheirality_mask

    # If no point remains, triangulation failed
    t_points = t_points[cheirality_mask]
    q_points = q_points[cheirality_mask]
    print(f"\t\tCheirality check filtered {num_points - cheirality_mask.sum()}/{num_points} points!")
    if cheirality_mask.sum() == 0:
        return None

    # -----------------------------------------------------
    # (2) Triangulation angle check
    # -----------------------------------------------------

    # Camera centers in the coordinate system where camera1 is at the origin.
    # For convenience, flatten them to shape (3,).
    C1 = np.zeros(3)                 # First camera at origin
    C2 = (-R.T @ t).reshape(3)       # Second camera center in the same coords

    # Vectors from camera centers to each 3D point
    # t_points is (N, 3), so the result is (N, 3)
    vec1 = q_points - C1[None, :]   # shape: (N, 3)
    vec2 = q_points - C2[None, :]   # shape: (N, 3)

    # Compute norms along axis=1 (per row) - distance of each point
    norms1 = np.linalg.norm(vec1, axis=1)  # shape: (N,)
    norms2 = np.linalg.norm(vec2, axis=1)  # shape: (N,)

    # Normalize vectors (element-wise division) - unit vectors
    vec1_norm = vec1 / (norms1[:, None] + 1e-8) # shape: (N, 3)
    vec2_norm = vec2 / (norms2[:, None] + 1e-8) # shape: (N, 3)

    # Compute dot product along axis=1 to get cos(theta)
    cos_angles = np.sum(vec1_norm * vec2_norm, axis=1)  # shape: (N,)

    # Clip to avoid numerical issues slightly outside [-1, 1]
    cos_angles = np.clip(cos_angles, -1.0, 1.0)

    # Convert to angles in degrees
    angles = np.degrees(np.arccos(cos_angles))  # shape: (N,)

    # Calculate median angle
    median_angle = np.median(angles)
    print(f"\t\t The median angle is {median_angle:.3f} deg.")

    # Filter out points with too small triangulation angle
    valid_angles_mask = angles >= SETTINGS["triangulation"]["min_angle"]
    filtered_angles = angles[valid_angles_mask]
    triang_mask[triang_mask==True] = valid_angles_mask

    # Check conditions to decide whether to discard
    print(f"\t\t Low Angles check filtered {sum(~valid_angles_mask)}/{cheirality_mask.sum()} points!")

    # Filter out points with very high angle compared to the median
    max_med_angles_mask = filtered_angles / median_angle < SETTINGS["triangulation"]["max_ratio_between_max_and_med_angle"]
    filtered_angles = filtered_angles[max_med_angles_mask]
    triang_mask[triang_mask==True] = max_med_angles_mask

    # Check conditions to decide whether to discard
    print(f"\t\t Max/Med Angles check filtered {sum(~max_med_angles_mask)}/{valid_angles_mask.sum()} points!")

    
    print(f"\t\tTotal filtering: {num_points-triang_mask.sum()}/{num_points}")
    print(f"\t\t {max_med_angles_mask.sum()} points left!")

    return triang_mask # (N,)

def filter_by_reprojection(matches, q_frame, t_frame, R, t, K, epipolar_constraint_mask, reproj_threshold=1.0):
    """
    Triangulate inlier correspondences, reproject them into the current frame, and filter matches by reprojection error.

    Args:
        matches (list): list of cv2.DMatch objects.
        frame, q_frame: Frame objects.
        R, t: relative pose from q_frame to frame.
        K: camera intrinsic matrix.
        epipolar_constraint_mask (np.array): initial boolean mask for inlier matches.
        reproj_threshold (float): reprojection error threshold in pixels.
        debug (bool): flag for visual debugging.
        results_dir (Path): path to save debug visualizations.

    Returns:
        np.array: Updated boolean mask with matches having large reprojection errors filtered out.
    """
    if debug:
        print("\tReprojecting correspondances...")
    # Extract matched keypoints
    q_frame_pts = np.float32([q_frame.keypoints[m.queryIdx].pt for m in matches])
    t_frame_pts = np.float32([t_frame.keypoints[m.trainIdx].pt for m in matches])

    # Select inlier matches
    q_pts = q_frame_pts[epipolar_constraint_mask]
    t_pts = t_frame_pts[epipolar_constraint_mask]

    # Projection matrices
    q_M = K @ np.eye(3,4)        # Reference frame (identity)
    t_M = K @ np.hstack((R, t))  # Current frame

    # Triangulate points
    q_points_4d = cv2.triangulatePoints(q_M, t_M, q_pts.T, t_pts.T)
    q_points_3d = (q_points_4d[:3] / q_points_4d[3]).T

    # Reproject points into the second (current) camera
    t_points = (R @ q_points_3d.T + t).T
    points_proj2, _ = cv2.projectPoints(t_points, np.zeros(3), np.zeros(3), K, None)
    points_proj_px = points_proj2.reshape(-1, 2)

    # Compute reprojection errors
    errors = np.linalg.norm(points_proj_px - t_pts, axis=1)

    # Update the inlier mask
    valid_mask = errors < reproj_threshold
    updated_mask = epipolar_constraint_mask.copy()
    original_indices = np.flatnonzero(epipolar_constraint_mask)
    updated_mask[original_indices[~valid_mask]] = False

    num_removed_matches = len(q_pts) - np.sum(valid_mask)
    if debug:
        print(f"\t\tPrior mean reprojection error: {np.mean(errors):.3f} px")
        print(f"\t\tReprojection filtered: {num_removed_matches}/{len(q_pts)}")
        print(f"\t\tFinal mean reprojection error: {np.mean(errors[valid_mask]):.3f} px")

    # Debugging visualization
    if debug:
        save_path = results_dir / f"matches/2-EH_reprojection/{q_frame.id}_{t_frame.id}.png"
        plot_reprojection(t_frame.img, t_pts, points_proj_px, path=save_path)

    return updated_mask
