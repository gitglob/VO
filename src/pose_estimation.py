import numpy as np
import cv2
from src.frame import Frame
from src.utils import invert_transform
from src.visualize import plot_matches
from src.utils import rotation_matrix_to_euler_angles

from config import results_dir


def estimate_pose(q_frame: Frame, t_frame: Frame, K: np.ndarray, debug=False):
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
    num_matches = len(matches)
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
    E, mask_E = cv2.findEssentialMat(q_kpt_pixels, t_kpt_pixels, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
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
        print("All keypoint pairs yield errors > threshold..")
        return None, False
    ratio = num_inliers_H / (num_inliers_E + num_inliers_H)
    if debug:
        print(f"\tRatio H/(E+H): {ratio}")
    use_homography = (ratio > 0.45)

    # ------------------------------------------------------------------------
    # 4. Recover pose (R, t) from Essential or Homography
    # ------------------------------------------------------------------------

    # Filter keypoints based on the chosen mask
    epipolar_constraint_mask = mask_H if use_homography else mask_E
    matches = matches[epipolar_constraint_mask]
    inlier_q_frame_pixels = q_kpt_pixels[epipolar_constraint_mask]
    inlier_t_frame_pixels = t_kpt_pixels[epipolar_constraint_mask]
    if debug:
        print(f"\tUsing {'Homography' if use_homography else 'Essential'} Matrix...")
        print(f"\t\tEpipolar Constraint filtered {num_matches - epipolar_constraint_mask.sum()}/{num_matches} matches!")

    # Check if we will use homography
    R, t, final_match_mask = None, None, None
    if not use_homography:
        # Decompose Essential Matrix
        _, R_est, t_est, mask_pose = cv2.recoverPose(E, inlier_q_frame_pixels, inlier_t_frame_pixels, K)

        # mask_pose indicates inliers used in cv2.recoverPose (1 for inliers, 0 for outliers)
        mask_pose = mask_pose.ravel().astype(bool)
        if debug:
            print(f"\t\tPose Recovery filtered {epipolar_constraint_mask.sum() - mask_pose.sum()}/{epipolar_constraint_mask.sum()} matches!")

        if R_est is not None and t_est is not None and np.any(mask_pose):
            R, t = R_est, t_est

        # Reprojection filter
        matches = matches[mask_pose]
        reproj_mask = filter_by_reprojection(
            matches, q_frame, t_frame,
            R, t, K,
            debug=debug
        )
        if debug:
            print(f"\t\tReprojection filtered {mask_pose.sum() - reproj_mask.sum()}/{mask_pose.sum()} matches!")
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
            for j in range(len(inlier_q_frame_pixels)):
                # Current frame pixel in camera coords
                p_curr_cam = invK @ np.array([*inlier_q_frame_pixels[j], 1.0])  
                # Previous frame pixel in camera coords
                p_prev_cam = invK @ np.array([*inlier_t_frame_pixels[j], 1.0])

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

        # Reprojection filter
        reproj_mask = filter_by_reprojection(
            q_kpt_pixels, t_kpt_pixels,
            R, t, K,
            debug=debug
        )
    matches = matches[reproj_mask]

    # If we failed to recover R and t
    if R is None or t is None:
        print("Failed to recover a valid pose from either E or H.")
        return None, False

    remaining_points = reproj_mask.sum()
    num_removed_matches = len(matches) - remaining_points
    if debug:
        print(f"\tTotal filtering: {num_removed_matches}/{len(matches)}. {remaining_points} matches left!")

    # ------------------------------------------------------------------------
    # 5. Build the 4x4 Pose matrix
    # ------------------------------------------------------------------------
    # Extract the c1 to c2 pose (this is the transformation that you need to transform a point from the old to the new coordinate system)
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t.flatten()
    # Extract the c2 to c1 pose (this is the new robot's pose in the old coordinate system)
    inv_pose = invert_transform(pose)
    
    # Save the matches
    if debug:
        match_save_path = results_dir / "matches/pose-estimation" / f"{q_frame.id}_{t_frame.id}.png"
        plot_matches(q_frame.img, q_frame.keypoints,
                     t_frame.img, t_frame.keypoints,
                     matches, match_save_path)

    return inv_pose, True

def filter_by_reprojection(matches, q_frame, t_frame, R, t, K, reproj_threshold=1.0, debug=False):
    """
    Triangulate inlier correspondences, reproject them into the current frame, and filter matches by reprojection error.

    Args:
        matches (list): list of cv2.DMatch objects.
        frame, q_frame: Frame objects.
        R, t: relative pose from q_frame to frame.
        K: camera intrinsic matrix.
        reproj_threshold (float): reprojection error threshold in pixels.
        debug (bool): flag for visual debugging.
        results_dir (Path): path to save debug visualizations.

    Returns:
        np.array: Updated boolean mask with matches having large reprojection errors filtered out.
    """
    if debug:
        print("\tReprojecting correspondances...")
    # Extract matched keypoints
    q_pts = np.float32([q_frame.keypoints[m.queryIdx].pt for m in matches])
    t_pts = np.float32([t_frame.keypoints[m.trainIdx].pt for m in matches])

    # Projection matrices
    q_M = K @ np.eye(3,4)        # Reference frame (identity)
    t_M = K @ np.hstack((R, t))  # Current frame

    # Triangulate points
    q_points_4d = cv2.triangulatePoints(q_M, t_M, q_pts.T, t_pts.T)
    q_points_3d = (q_points_4d[:3] / q_points_4d[3]).T

    # Reproject points into the second (current) camera
    t_points_3d = (R @ q_points_3d.T + t).T
    points_proj2, _ = cv2.projectPoints(t_points_3d, np.zeros(3), np.zeros(3), K, None)
    points_proj_px = points_proj2.reshape(-1, 2)

    # Compute reprojection errors
    errors = np.linalg.norm(points_proj_px - t_pts, axis=1)

    # Update the inlier mask
    reproj_mask = errors < reproj_threshold

    num_removed_matches = len(q_pts) - np.sum(reproj_mask)
    if debug:
        print(f"\t\tPrior mean reprojection error: {np.mean(errors):.3f} px")
        print(f"\t\tReprojection filtered: {num_removed_matches}/{len(q_pts)}")
        print(f"\t\tFinal mean reprojection error: {np.mean(errors[reproj_mask]):.3f} px")

    # Debugging visualization
    if debug:
        reproj_img = t_frame.img.copy()
        for i in range(len(t_pts)):
            obs = tuple(np.int32(t_pts[i]))
            reproj = tuple(np.int32(points_proj_px[i]))
            cv2.circle(reproj_img, obs, 3, (0, 0, 255), -1)      # Observed points (red)
            cv2.circle(reproj_img, reproj, 2, (0, 255, 0), -1)   # Projected points (green)
            cv2.line(reproj_img, obs, reproj, (255, 0, 0), 1)    # Error line (blue)

        debug_img_path = results_dir / f"matches/reprojection/{q_frame.id}_{t_frame.id}.png"
        debug_img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_img_path), reproj_img)

    return reproj_mask
      
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

def is_keyframe(P, t_threshold=3, angle_threshold=20, debug=False):
    """ Determine if motion expressed by t, R is significant by comparing to tresholds. """
    R = P[:3, :3]
    t = P[:3, 3]

    trans = np.sqrt(t[0]**2 + t[2]**2)
    rpy = rotation_matrix_to_euler_angles(R)
    pitch = abs(rpy[1])

    is_keyframe = trans > t_threshold or pitch > angle_threshold
    if debug:
        print(f"\tDisplacement: dist: {trans:.3f}, angle: {pitch:.3f}")
        if is_keyframe:
            print("\t\tKeyframe!")
        else:
            print("\t\tNot a keyframe!")

    return is_keyframe
