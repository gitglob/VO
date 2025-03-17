import numpy as np
import cv2
from src.frame import Frame
from src.utils import invert_transform, transform_points
from src.visualize import plot_matches

from config import results_dir, debug, SETTINGS


def initialize_pose(q_frame: Frame, t_frame: Frame, K: np.ndarray):
    """
    Initializes the camera pose by estimating the relative rotation and translation 
    between two consecutive frames using feature matches.

    This function computes the Essential and Homography matrices to determine 
    the best motion model. It then recovers the relative pose (rotation and 
    translation) using the Essential matrix if the motion is mostly translational 
    or the Homography matrix if the scene is planar. Finally, the pose is used 
    to initialize the frames and triangulate 3D points.

    Args:
        q_frame (Frame): The previous frame.
        t_frame (Frame): The current frame.
        K (np.ndarray): The camera intrinsic matrix (3x3).
        debug (bool, optional): If True, saves and visualizes matches.

    Returns:
        Tuple[np.ndarray or None, bool]: 
            - The inverse 4x4 transformation matrix (q_frame -> frame) if successful, otherwise None.
            - A boolean indicating whether the initialization was successful.
    """
    if debug:
        print(f"Initializing the camera pose using frames {q_frame.id} & {t_frame.id}...")
    
    # ------------------------------------------------------------------------
    # 1. Get keypoint matches
    # ------------------------------------------------------------------------

    # Extract the matches between the previous and current frame
    matches = q_frame.get_matches(t_frame.id)
    if len(matches) < 5:
        print("Not enough matches to compute the Essential Matrix!")
        return None, False

    # Extract keypoint pixel coordinates and indices for both frames from the feature match
    q_kpt_pixels = np.float32([q_frame.keypoints[m.queryIdx].pt for m in matches])
    t_kpt_pixels = np.float32([t_frame.keypoints[m.trainIdx].pt for m in matches])

    # ------------------------------------------------------------------------
    # 2. Compute Essential & Homography matrices
    # ------------------------------------------------------------------------

    # Compute the Essential Matrix
    E, mask_E = cv2.findEssentialMat(q_kpt_pixels, t_kpt_pixels, K, method=cv2.RANSAC, 
                                     prob=0.999, threshold=1.0)
    mask_E = mask_E.ravel().astype(bool)

    # Compute the Homography Matrix
    H, mask_H = cv2.findHomography(q_kpt_pixels, t_kpt_pixels, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    mask_H = mask_H.ravel().astype(bool)

    # ------------------------------------------------------------------------
    # 3. Compute symmetric transfer errors & decide which model to use
    # ------------------------------------------------------------------------

    # Compute symmetric transfer error for Essential Matrix
    error_E, num_inliers_E = compute_symmetric_transfer_error(E, q_kpt_pixels, t_kpt_pixels, 'E', K=K)

    # Compute symmetric transfer error for Homography Matrix
    error_H, num_inliers_H = compute_symmetric_transfer_error(H, q_kpt_pixels, t_kpt_pixels, 'H', K=K)

    # Decide which matrix to use based on the ratio of inliers
    if debug:
        print(f"\tInliers E: {num_inliers_E}, Inliers H: {num_inliers_H}")
    if num_inliers_E == 0 and num_inliers_H == 0:
        print("[initialize] All keypoint pairs yield errors > threshold..")
        return None, False
    ratio = num_inliers_H / (num_inliers_E + num_inliers_H)
    if debug:
        print(f"\tRatio H/(E+H): {ratio}")
    use_homography = (ratio > 0.45)

    # ------------------------------------------------------------------------
    # 4. Recover pose (R, t) from Essential or Homography
    # ------------------------------------------------------------------------

    # Filter keypoints based on the chosen mask
    inlier_match_mask = mask_H if use_homography else mask_E
    inlier_q_pixels = q_kpt_pixels[inlier_match_mask]
    inlier_t_pixels = t_kpt_pixels[inlier_match_mask]
    if debug:
        print(f"\tUsing {'Homography' if use_homography else 'Essential'} Matrix...")
        print(f"\t\tE/H inliers filtered {len(matches) - np.sum(inlier_match_mask)}/{len(matches)} matches!")

    # Check if we will use homography
    R, t, final_match_mask = None, None, None
    if not use_homography:
        # Decompose Essential Matrix
        _, R_est, t_est, mask_pose = cv2.recoverPose(E, inlier_q_pixels, inlier_t_pixels, K)

        # mask_pose indicates inliers used in cv2.recoverPose (1 for inliers, 0 for outliers)
        mask_pose = mask_pose.ravel().astype(bool)
        if debug:
            print(f"\t\tPose Recovery filtered {np.sum(inlier_match_mask) - np.sum(mask_pose)}/{np.sum(inlier_match_mask)} matches!")

        if R_est is not None and t_est is not None and np.any(mask_pose):
            R, t = R_est, t_est

            # Create a final_match_mask by combining the epipolar constraint and transformation fitting checks
            final_match_mask = np.zeros_like(inlier_match_mask, dtype=bool)
            final_match_mask[inlier_match_mask] = mask_pose

        # Reprojection filter
        final_match_mask = filter_by_reprojection(
            matches, q_frame, t_frame,
            R, t, K,
            final_match_mask
        )
        if debug:
            print(f"\t\tReprojection filtered {np.sum(mask_pose) - np.sum(final_match_mask)}/{np.sum(mask_pose)} matches!")
    else:
        # Decompose Homography Matrix
        num_solutions, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)

        # Select the best solution based on criteria
        best_solution = None
        max_front_points = 0
        best_alignment = -1
        desired_normal = np.array([0, 0, 1])

        for i in range(num_solutions):
            R_candidate = Rs[i]
            t_candidate = Ts[i]
            n_candidate = Ns[i]

            # Check if the normal aligns with the 'upward' direction (optional criterion)
            alignment = np.dot(n_candidate, desired_normal)

            # Check if points are in front of camera
            front_points = 0
            invK = np.linalg.inv(K)
            for j in range(len(inlier_q_pixels)):
                # Current frame pixel in camera coords
                p_curr_cam = invK @ np.array([*inlier_q_pixels[j], 1.0])  
                # Previous frame pixel in camera coords
                p_prev_cam = invK @ np.array([*inlier_t_pixels[j], 1.0])

                # Depth for current pixel after transformation
                denom = np.dot(n_candidate, R_candidate @ p_curr_cam + t_candidate)
                depth_curr = np.dot(n_candidate, p_curr_cam) / (denom + 1e-12)  # small eps for safety
                
                # Depth for previous pixel (just dot product since it's reference plane)
                depth_prev = np.dot(n_candidate, p_prev_cam)

                if depth_prev > 0 and depth_curr > 0:
                    front_points += 1

            # Update best solution if it meets criteria
            if front_points > max_front_points and alignment > best_alignment:
                max_front_points = front_points
                best_alignment = alignment
                best_solution = i

        # Use the best solution
        R = Rs[best_solution]
        t = Ts[best_solution]

        # The final mask doesn't change if we use the Homography Matrix
        final_match_mask = inlier_match_mask

        # Reprojection filter
        final_match_mask = filter_by_reprojection(
            q_kpt_pixels, t_kpt_pixels,
            R, t, K,
            final_match_mask,
            reproj_threshold=2.0
        )

    # If we failed to recover R and t
    if R is None or t is None:
        print("[initialize] Failed to recover a valid pose from either E or H.")
        return None, False

    remaining_points = np.sum(final_match_mask)
    num_removed_matches = len(matches) - remaining_points
    if debug:
        print(f"\tTotal filtering: {num_removed_matches}/{len(matches)}. {remaining_points} matches left!")

    # ------------------------------------------------------------------------
    # 5. Build the 4x4 Pose matrix
    # ------------------------------------------------------------------------
    # Extract the c1 to c2 pose
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t.flatten()
    # Extract the c2 to c1 pose (this is the new robot's pose in the old coordinate system)
    inv_pose = invert_transform(pose)
    
    # Initialize the frames
    q_frame.initialize(t_frame.id, use_homography, final_match_mask, pose)
    t_frame.initialize(q_frame.id, use_homography, final_match_mask, inv_pose)
            
    # Save the matches
    if debug:
        match_save_path = results_dir / "matches/1-initialization" / f"{q_frame.id}_{t_frame.id}.png"
        plot_matches(q_frame, t_frame, mask=final_match_mask, save_path=match_save_path)

    return pose, True

def filter_by_reprojection(matches, q_frame, t_frame, R, t, K, inlier_match_mask, reproj_threshold=1.0):
    """
    Triangulate inlier correspondences, reproject them into the current frame, and filter matches by reprojection error.

    Args:
        matches (list): list of cv2.DMatch objects.
        frame, q_frame: Frame objects.
        R, t: relative pose from q_frame to frame.
        K: camera intrinsic matrix.
        inlier_match_mask (np.array): initial boolean mask for inlier matches.
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
    q_pts = q_frame_pts[inlier_match_mask]
    t_pts = t_frame_pts[inlier_match_mask]

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
    updated_mask = inlier_match_mask.copy()
    original_indices = np.flatnonzero(inlier_match_mask)
    updated_mask[original_indices[~valid_mask]] = False

    num_removed_matches = len(q_pts) - np.sum(valid_mask)
    if debug:
        print(f"\t\tPrior mean reprojection error: {np.mean(errors):.3f} px")
        print(f"\t\tReprojection filtered: {num_removed_matches}/{len(q_pts)}")
        print(f"\t\tFinal mean reprojection error: {np.mean(errors[valid_mask]):.3f} px")

    # Debugging visualization
    if debug:
        reproj_img = t_frame.img.copy()
        for i in range(len(t_pts)):
            obs = tuple(np.int32(t_pts[i]))
            reproj = tuple(np.int32(points_proj_px[i]))
            cv2.circle(reproj_img, obs, 3, (0, 0, 255), -1)      # Observed points (red)
            cv2.circle(reproj_img, reproj, 2, (0, 255, 0), -1)   # Projected points (green)
            cv2.line(reproj_img, obs, reproj, (255, 0, 0), 1)    # Error line (blue)

        debug_img_path = results_dir / f"matches/2-reprojection/{q_frame.id}_{t_frame.id}.png"
        debug_img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_img_path), reproj_img)

    return updated_mask
      
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

def triangulate_points(q_frame: Frame, t_frame: Frame, K: np.ndarray, scale: int):
    if debug:
        print(f"Triangulating points between frames {q_frame.id} & {t_frame.id}...")
    # Extract the Rotation and Translation arrays between the 2 frames
    T_qt = q_frame.match[t_frame.id]["T"] # [q->t]
    R_qt = T_qt[:3, :3]
    t_qt = T_qt[:3, 3].reshape(3,1)

    # ------------------------------------------------------------------------
    # 6. Triangulate 3D points
    # ------------------------------------------------------------------------

    # Extract inlier matches
    matches = q_frame.get_matches(t_frame.id)
    inlier_match_mask = q_frame.match[t_frame.id]["inlier_match_mask"]

    # Initialize the triangulation match mask
    triang_match_mask = inlier_match_mask.copy()

    # Extract keypoint pixel coordinates and indices for both frames from the feature match
    q_kpt_pixels = np.float32([q_frame.keypoints[m.queryIdx].pt for m in matches[inlier_match_mask]])
    t_kpt_pixels = np.float32([t_frame.keypoints[m.trainIdx].pt for m in matches[inlier_match_mask]])

    # Triangulate
    q_points = triangulate(q_kpt_pixels, t_kpt_pixels, R_qt, t_qt, K) # (N, 3)
    if q_points is None or len(q_points) == 0:
        print("[initialize] Triangulation returned no 3D points.")
        return None, None, False

    # Transfer the points to the current coordinate frame [t->q]
    t_points = transform_points(q_points, T_qt) # (N, 3)

    # Scale the points
    q_points = scale * q_points
    t_points = scale * t_points

    # ------------------------------------------------------------------------
    # 7. Filter out points with small triangulation angles (cheirality check)
    # ------------------------------------------------------------------------

    filters_mask = filter_triangulation_points(q_points, t_points, R_qt, t_qt)
    # If too few points or too small median angle, return None
    if filters_mask is None or filters_mask.sum() < SETTINGS["matches"]["min"]:
        print("Discarding frame due to insufficient triangulation quality.")
        return None, None, False
    q_points = q_points[filters_mask]
    t_points = t_points[filters_mask]

    # Combine the Homography/Essential Matrix mask with the valid angles mask
    triang_match_mask[triang_match_mask==True] = filters_mask

    # ------------------------------------------------------------------------
    # 8. Save the triangulated points and masks to the t_frame
    # ------------------------------------------------------------------------

    q_frame.match[t_frame.id]["triangulation_match_mask"] = triang_match_mask
    q_frame.match[t_frame.id]["points"] = q_points

    t_frame.match[q_frame.id]["triangulation_match_mask"] = triang_match_mask
    t_frame.match[q_frame.id]["points"] = t_points

    # Also save the triangulated points keypoint identifiers
    q_kpt_ids = np.float32([q_frame.keypoints[m.queryIdx].class_id for m in matches[triang_match_mask]])
    t_frame.match[q_frame.id]["point_ids"] = q_kpt_ids

    t_kpt_ids = np.float32([t_frame.keypoints[m.trainIdx].class_id for m in matches[triang_match_mask]])
    q_frame.match[t_frame.id]["point_ids"] = t_kpt_ids
            
    # Save the matches
    if debug:
        match_save_path = results_dir / "matches/3-triangulation" / f"{q_frame.id}_{t_frame.id}.png"
        plot_matches(q_frame, t_frame, triang_match_mask, match_save_path)

    # Return the initial pose and filtered points
    return t_points, t_kpt_ids, True
      
def triangulate(q_frame_pixels, t_frame_pixels, R, t, K):
    # Compute projection matrices for triangulation
    q_M = K @ np.eye(3,4)  # First camera at origin
    t_M = K @ np.hstack((R, t))  # Second camera at R, t

    # Triangulate points
    q_frame_points_4d_hom = cv2.triangulatePoints(q_M, t_M, q_frame_pixels.T, t_frame_pixels.T)

    # Convert homogeneous coordinates to 3D
    q_points_3d = q_frame_points_4d_hom[:3] / q_frame_points_4d_hom[3]

    return q_points_3d.T # (N, 3)

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
    print("\t\tFiltering points with small triangulation angles...")
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
    max_med_angles_mask = filtered_angles / median_angle < SETTINGS["triangulation"]["max_ratio_between_max_and_min_angle"]
    filtered_angles = filtered_angles[max_med_angles_mask]
    triang_mask[triang_mask==True] = max_med_angles_mask

    # Check conditions to decide whether to discard
    print(f"\t\t Max/Med Angles check filtered {sum(~max_med_angles_mask)}/{valid_angles_mask.sum()} points!")

    
    print(f"\t\tTotal filtering: {num_points-triang_mask.sum()}/{num_points}")
    print(f"\t\t {max_med_angles_mask.sum()} points left!")

    return triang_mask # (N,)
