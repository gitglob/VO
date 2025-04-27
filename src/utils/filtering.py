import numpy as np
import cv2
import src.utils as utils
import src.visualization as vis

from config import results_dir, SETTINGS, K, log

debug = SETTINGS["generic"]["debug"]
SCALE_FACTOR = SETTINGS["orb"]["scale_factor"]
N_LEVELS = SETTINGS["orb"]["level_pyramid"]
min_observations = SETTINGS["orb"]["level_pyramid"]

############################### Matches ###############################

def filterMatches(matches, lowe_ratio):
    """Filter out matches using Lowe's Ratio Test"""
    good_matches = []
    for m, n in matches:
        if m.distance < lowe_ratio * n.distance:
            good_matches.append(m)

    if debug:
        log.info(f"\t Lowe's Test filtered {len(matches) - len(good_matches)}/{len(matches)} matches!")

    # Next, ensure uniqueness by keeping only the best match per train descriptor.
    unique_matches = {}
    for m in good_matches:
        # If this train descriptor is not seen yet, or if the current match is better, update.
        if m.trainIdx not in unique_matches or m.distance < unique_matches[m.trainIdx].distance:
            unique_matches[m.trainIdx] = m

    # Convert the dictionary values to a list of unique matches
    unique_matches = list(unique_matches.values())

    if debug:
        log.info(f"\t Uniqueness filtered {len(good_matches) - len(unique_matches)}/{len(good_matches)} matches!")

    return unique_matches

############################### Keypoints ###############################

def enforce_epipolar_constraint(q_kpt_pixels, t_kpt_pixels):
    # Compute Essential & Homography matrices

    ## Compute the Essential Matrix
    E, mask_E = cv2.findEssentialMat(q_kpt_pixels, t_kpt_pixels, 
                                     cameraMatrix=K, method=cv2.RANSAC, 
                                     prob=0.999, threshold=1.0)
    mask_E = mask_E.ravel().astype(bool)

    ## Compute the Homography Matrix
    H, mask_H = cv2.findHomography(q_kpt_pixels, t_kpt_pixels,
                                   method=cv2.RANSAC, 
                                   ransacReprojThreshold=1.0)
    mask_H = mask_H.ravel().astype(bool)

    # Compute symmetric transfer errors & decide which model to use

    ## Compute symmetric transfer error for Essential Matrix
    score_F = compute_symmetric_transfer_error(E, q_kpt_pixels, t_kpt_pixels, 'E')

    ## Compute symmetric transfer error for Homography Matrix
    score_H = compute_symmetric_transfer_error(H, q_kpt_pixels, t_kpt_pixels, 'H')
    
    ## Decide which matrix to use based on the ratio of inliers
    ratio_H = score_H / (score_H + score_F)
    use_homography = (ratio_H > 0.45)
    epipolar_constraint_mask = mask_H if use_homography else mask_E
    M = H if use_homography else E
    if debug:
        log.info(f"\t\t Ratio: {ratio_H:.2f}. Using {'Homography' if use_homography else 'Essential'} Matrix...")
        log.info(f"\t\t Epipolar Constraint filtered {sum(~epipolar_constraint_mask)}/{len(q_kpt_pixels)} matches!")

    return epipolar_constraint_mask, M, use_homography

def compute_symmetric_transfer_error(E_or_H, q_kpt_pixels, t_kpt_pixels, matrix_type='E', T_H = 5.99, T_F = 3.84):
    """
    Computes the symmetric transfer error (in pixels) for a set of corresponding keypoints using
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
    score = 0

    if matrix_type == 'E':
        if K is None:
            raise ValueError("Camera intrinsic matrix K must be provided when using an essential matrix.")
        try:
            K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            raise ValueError("Provided homography is non-invertible.")
        
        # Compute fundamental matrix F from the essential matrix
        # The fundamental matrix maps pixels in one image to an epiline in the other image
        F = K_inv.T @ E_or_H @ K_inv
        
        for q_px, t_px in zip(q_kpt_pixels, t_kpt_pixels):
            # Convert pixel coordinates to homogeneous coordinates
            qp = np.array([q_px[0], q_px[1], 1.0])
            tp = np.array([t_px[0], t_px[1], 1.0])
            # Compute the epipolar lines in the other image
            # That means that the q_pixels will be found in the t)image on the t_line and the t_pixels on the q_image on the q_line
            # A line is in the form of ax + by + c, where [x,y] are the pixel coordinates
            q_line = F @ tp   # Line in query image corresponding to tp
            t_line = F.T @ qp # Line in train image corresponding to qp
            
            # Normalize the lines using the norm of the first two components
            q_line_norm = np.hypot(q_line[0], q_line[1])
            t_line_norm = np.hypot(t_line[0], t_line[1])
            if q_line_norm == 0 or t_line_norm == 0:
                continue
            q_line_normalized = q_line / q_line_norm
            t_line_normalized = t_line / t_line_norm

            # Compute the perpendicular distances from the points to the lines
            dq = abs(np.dot(qp, q_line_normalized))
            q_error = dq**2
            dt = abs(np.dot(tp, t_line_normalized))
            t_error = dt**2

            # Compute the score
            sq = T_F - q_error if q_error < T_F else 0
            st = T_F - t_error if t_error < T_F else 0
            score += sq + st
    elif matrix_type == 'H':
        # For homography, compute the symmetric reprojection error.
        # The Homography matrix in general maps points from the train image to the query image: p1 = H @ p2
        H = E_or_H  # H: mapping from target to query image
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            raise ValueError("Provided homography is non-invertible.")
        
        for q_px, t_px in zip(q_kpt_pixels, t_kpt_pixels):
            # Convert pixel coordinates to homogeneous coordinates
            qp = np.array([q_px[0], q_px[1], 1.0])
            tp = np.array([t_px[0], t_px[1], 1.0])
            
            # Map train points to the query image
            qp_est = H @ tp
            if np.isclose(qp_est[2], 0):
                continue
            qp_est = qp_est / qp_est[2]
            
            # Map query points to the train image
            tp_est = H_inv @ qp
            if np.isclose(tp_est[2], 0):
                continue
            tp_est = tp_est / tp_est[2]
            
            # Compute reprojection errors in both directions (Euclidean distances)
            dq = np.linalg.norm(tp - tp_est)
            q_error = dq**2
            dt = np.linalg.norm(qp - qp_est)
            t_error = dt**2

            # Compute the score
            sq = T_H - q_error if q_error < T_H else 0
            st = T_H - t_error if t_error < T_H else 0
            score += sq + st
    else:
        raise ValueError("matrix_type must be either 'E' (essential matrix) or 'H' (homography)")

    return score

############################### Triangulation ###############################

def filter_cheirality(q_points: np.ndarray, t_points: np.ndarray):
    """Filter out 3D points that lie behind the camera planes"""
    num_points = len(t_points)

    # t_points is in the query and train camera => check Z > 0
    Z1 = q_points[:, 2]
    Z2 = t_points[:, 2]

    # We'll mark True if both Z1, Z2 > 0
    cheirality_mask = (Z1 > 0) & (Z2 > 0)

    if debug:
        log.info(f"\t\t Cheirality check filtered {sum(~cheirality_mask)}/{num_points} points!")
    if cheirality_mask.sum() == 0:
        return None
    
    return cheirality_mask
    
def filter_parallax(q_points: np.ndarray, t_points: np.ndarray, T: np.ndarray, min_angle: float):
    """
    Filter out 3D points that have a small triangulation angle

    t_points : (N, 3)
    R, t      : the rotation and translation from q_cam to t_cam
    Returns:   valid_angles_mask (bool array of shape (N,)),
               filtered_points_3d (N_filtered, 3) or (None, None)
    """
    num_points = len(t_points)
    R = T[:3, :3]
    t = T[:3, 3]

    # Camera centers in the coordinate system where camera1 is at the origin.
    # For convenience, flatten them to shape (3,).
    C1 = np.zeros(3)                 # First camera at origin
    C2 = (-R.T @ t).reshape(3)       # Second camera center in the same coords

    # Vectors from camera centers to each 3D point
    # t_points is (N, 3), so the result is (N, 3)
    vec1 = q_points - C1[None, :]   # shape: (N, 3)
    vec2 = t_points - C2[None, :]   # shape: (N, 3)

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

    # Filter out points with too small triangulation angle
    valid_angles_mask = angles >= min_angle

    # Check conditions to decide whether to discard
    if debug:
        log.info(f"\t\t Median Angle: {np.median(angles):.2f}")
        log.info(f"\t\t Low Angles check filtered {sum(~valid_angles_mask)}/{num_points} points!")

    return valid_angles_mask # (N,)

def filter_by_reprojection(matches, q_frame, t_frame, R, t, threshold, save_path):
    """
    Triangulate inlier correspondences, reproject them into the current frame, and filter matches by reprojection error.

    Args:
        matches (list): list of cv2.DMatch objects.
        frame, q_frame: Frame objects.
        R, t: relative pose from q_frame to frame.
        K: camera intrinsic matrix.
        epipolar_constraint_mask (np.array): initial boolean mask for inlier matches.

    Returns:
        np.array: Updated boolean mask with matches having large reprojection errors filtered out.
    """
    # Extract matched keypoints
    q_pxs = np.float64([q_frame.keypoints[m.queryIdx].pt for m in matches])
    t_pxs = np.float64([t_frame.keypoints[m.trainIdx].pt for m in matches])

    # Projection matrices
    q_M = K @ np.eye(3,4)        # Reference frame (identity)
    t_M = K @ np.hstack((R, t))  # Current frame

    # Triangulate points
    q_points_4d = cv2.triangulatePoints(q_M, t_M, q_pxs.T, t_pxs.T)
    q_points_3d = (q_points_4d[:3] / q_points_4d[3]).T

    # Reproject points into the second (current) camera
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3,1) 
    points_proj2, _ = cv2.projectPoints(q_points_3d, rvec, tvec, K, None)
    points_proj_px = points_proj2.reshape(-1, 2)

    # Compute reprojection errors
    errors = np.linalg.norm(points_proj_px - t_pxs, axis=1)
    reproj_mask = errors < threshold

    num_removed_matches = len(q_pxs) - np.sum(reproj_mask)
    if debug:
        log.info(f"\t\t Reprojection filtered: {num_removed_matches}/{len(q_pxs)}. E: {np.mean(errors):.3f} -> {np.mean(errors[reproj_mask]):.3f}")

    # Debugging visualization
    if debug:
        vis.plot_reprojection(t_frame.img, t_pxs[~reproj_mask], points_proj_px[~reproj_mask], path=save_path / f"{q_frame.id}_{t_frame.id}a.png")
        vis.plot_reprojection(t_frame.img, t_pxs[reproj_mask], points_proj_px[reproj_mask], path=save_path / f"{q_frame.id}_{t_frame.id}b.png")

    return reproj_mask

def filter_scale(points: np.ndarray, kpts: np.ndarray, T_cw: np.ndarray):
    """
    Filters a set of 3D points based on scale invariance thresholds derived from the corresponding keypoints.
    """
    num_points = len(points)
    cam_center = T_cw[:3, 3]

    # Iterate over all points
    scale_mask = np.ones(num_points, dtype=bool)
    for i in range(num_points):
        pos = points[i]
        kpt = kpts[i]
    
        # Get map point distance
        dist = np.linalg.norm(pos - cam_center)
        dmin, dmax = utils.get_scale_invariance_limits(dist, kpt.octave)

        # Check if the map_point distance is in the scale invariance region
        if dist < dmin or dist > dmax:
            scale_mask[i] = False

    # Check conditions to decide whether to discard
    if debug:
        log.info(f"\t\t Scale check filtered {num_points - scale_mask.sum()}/{num_points} points!")

    return scale_mask
