import numpy as np
import cv2
from src.frame import Frame
from src.utils import invert_transform
from src.visualize import plot_matches

from config import results_dir


def initialize_pose(frame: Frame, ref_frame: Frame, K: np.ndarray, debug=False):
    """
    Initializes the camera pose by estimating the relative rotation and translation 
    between two consecutive frames using feature matches.

    This function computes the Essential and Homography matrices to determine 
    the best motion model. It then recovers the relative pose (rotation and 
    translation) using the Essential matrix if the motion is mostly translational 
    or the Homography matrix if the scene is planar. Finally, the pose is used 
    to initialize the frames and triangulate 3D points.

    Args:
        frame (Frame): The current frame containing keypoints, descriptors, and matches.
        ref_frame (Frame): The previous reference frame.
        K (np.ndarray): The camera intrinsic matrix (3x3).
        debug (bool, optional): If True, saves and visualizes matches used for initialization.

    Returns:
        Tuple[np.ndarray or None, bool]: 
            - The inverse 4x4 transformation matrix (ref_frame -> frame) if successful, otherwise None.
            - A boolean indicating whether the initialization was successful.
    """
    print(f"Initializing the camera pose using frames {frame.id} & {ref_frame.id}...")
    
    # ------------------------------------------------------------------------
    # 1. Get keypoint matches
    # ------------------------------------------------------------------------

    # Extract the matches between the previous and current frame
    matches = frame.get_matches(ref_frame.id)
    if len(matches) < 5:
        print("Not enough matches to compute the Essential Matrix!")
        return frame.pose, False

    # Extract keypoint pixel coordinates and indices for both frames from the feature match
    frame_kpt_indices = np.array([m.queryIdx for m in matches])
    frame_kpt_pixels = np.float32([frame.keypoints[idx].pt for idx in frame_kpt_indices])
    ref_frame_kpt_indices = np.array([m.trainIdx for m in matches])
    ref_frame_kpt_pixels = np.float32([ref_frame.keypoints[idx].pt for idx in ref_frame_kpt_indices])

    # ------------------------------------------------------------------------
    # 2. Compute Essential & Homography matrices
    # ------------------------------------------------------------------------

    # Compute the Essential Matrix
    E, mask_E = cv2.findEssentialMat(ref_frame_kpt_pixels, frame_kpt_pixels, K, method=cv2.RANSAC, prob=0.99, threshold=1.5)
    mask_E = mask_E.ravel().astype(bool)

    # Compute the Homography Matrix
    H, mask_H = cv2.findHomography(ref_frame_kpt_pixels, frame_kpt_pixels, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    mask_H = mask_H.ravel().astype(bool)

    # ------------------------------------------------------------------------
    # 3. Compute symmetric transfer errors & decide which model to use
    # ------------------------------------------------------------------------

    # Compute symmetric transfer error for Essential Matrix
    error_E, num_inliers_E = compute_symmetric_transfer_error(E, frame_kpt_pixels, ref_frame_kpt_pixels, 'E', K=K)

    # Compute symmetric transfer error for Homography Matrix
    error_H, num_inliers_H = compute_symmetric_transfer_error(H, frame_kpt_pixels, ref_frame_kpt_pixels, 'H', K=K)

    # Decide which matrix to use based on the ratio of inliers
    ratio = num_inliers_H / (num_inliers_E + num_inliers_H)
    print(f"\tInliers E: {num_inliers_E}, Inliers H: {num_inliers_H}, Ratio H/(E+H): {ratio}")
    use_homography = (ratio > 0.45)

    # ------------------------------------------------------------------------
    # 4. Recover pose (R, t) from Essential or Homography
    # ------------------------------------------------------------------------

    # Filter keypoints based on the chosen mask
    inlier_match_mask = mask_H if use_homography else mask_E
    print(f"\tUsing {'Homography' if use_homography else 'Essential'} Matrix...")
    inlier_frame_pixels = frame_kpt_pixels[inlier_match_mask]
    inlier_ref_frame_pixels = ref_frame_kpt_pixels[inlier_match_mask]
    print(f"\t\tE/H inliers filtered {len(matches) - np.sum(inlier_match_mask)}/{len(matches)} matches!")

    # Check if we will use homography
    R, t, final_match_mask = None, None, None
    if not use_homography:
        # Decompose Essential Matrix
        points, R_est, t_est, mask_pose = cv2.recoverPose(E, inlier_ref_frame_pixels, inlier_frame_pixels, K)

        # mask_pose indicates inliers used in cv2.recoverPose (1 for inliers, 0 for outliers)
        mask_pose = mask_pose.ravel().astype(bool)
        print(f"\t\tPose Recovery filtered {np.sum(inlier_match_mask) - np.sum(mask_pose)}/{np.sum(inlier_match_mask)} matches!")

        if R_est is not None and t_est is not None and np.any(mask_pose):
            R, t = R_est, t_est

            # Create a final_match_mask by combining the epipolar constraint and transformation fitting checks
            final_match_mask = np.zeros_like(inlier_match_mask, dtype=bool)
            final_match_mask[inlier_match_mask] = mask_pose

        # Reprojection filter
        final_match_mask = filter_by_reprojection(
            matches, frame, ref_frame,
            R, t, K,
            final_match_mask,
            reproj_threshold=2.0,
            debug=debug
        )
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
            for j in range(len(inlier_ref_frame_pixels)):
                # Current frame pixel in camera coords
                p_curr_cam = invK @ np.array([*inlier_ref_frame_pixels[j], 1.0])  
                # Previous frame pixel in camera coords
                p_prev_cam = invK @ np.array([*inlier_frame_pixels[j], 1.0])

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
            frame_kpt_pixels, ref_frame_kpt_pixels,
            R, t, K,
            final_match_mask,
            reproj_threshold=2.0,
            debug=debug
        )

    # If we failed to recover R and t
    if R is None or t is None:
        print("[initialize] Failed to recover a valid pose from either E or H.")
        return None, False

    num_removed_matches = len(matches) - np.sum(final_match_mask)
    remaining_points = np.sum(final_match_mask)
    print(f"\tTotal filtering: {num_removed_matches}/{len(matches)}. {remaining_points} matches left!")

    # ------------------------------------------------------------------------
    # 5. Build the 4x4 Pose matrix
    # ------------------------------------------------------------------------
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t.flatten()
    inv_pose = invert_transform(pose)

    # Print the transformation
    yaw_deg = abs(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    print(f"\tTransformation: dx:{pose[0,3]:.3f}, dy:{pose[1,3]:.3f}, yaw: {yaw_deg:.3f}")
    
    # Initialize the frames
    frame.triangulate(ref_frame.id, use_homography, final_match_mask, pose, "initialization")
    ref_frame.triangulate(frame.id, use_homography, final_match_mask, inv_pose, "initialization")
            
    # Save the matches
    if debug:
        match_save_path = results_dir / "matches/1-initialization" / f"{frame.id}_{ref_frame.id}.png"
        plot_matches(frame.img, frame.keypoints,
                     ref_frame.img, ref_frame.keypoints,
                     matches[final_match_mask], match_save_path)

    return pose, True

def filter_by_reprojection(matches, frame, ref_frame, R, t, K, inlier_match_mask, reproj_threshold=2.0, debug=False):
    """
    Triangulate inlier correspondences, reproject them into the current frame, and filter matches by reprojection error.

    Args:
        matches (list): list of cv2.DMatch objects.
        frame, ref_frame: Frame objects.
        R, t: relative pose from ref_frame to frame.
        K: camera intrinsic matrix.
        inlier_match_mask (np.array): initial boolean mask for inlier matches.
        reproj_threshold (float): reprojection error threshold in pixels.
        debug (bool): flag for visual debugging.
        results_dir (Path): path to save debug visualizations.

    Returns:
        np.array: Updated boolean mask with matches having large reprojection errors filtered out.
    """
    print("\tReprojecting correspondances...")
    # Extract matched keypoints
    frame_pts = np.float32([frame.keypoints[m.queryIdx].pt for m in matches])
    ref_frame_pts = np.float32([ref_frame.keypoints[m.trainIdx].pt for m in matches])

    # Select inlier matches
    pts_ref = ref_frame_pts[inlier_match_mask]
    pts_frame = frame_pts[inlier_match_mask]

    # Projection matrices
    M1 = K @ np.eye(3,4)        # Reference frame (identity)
    M2 = K @ np.hstack((R, t))  # Current frame

    # Triangulate points
    points_4d = cv2.triangulatePoints(M1, M2, pts_ref.T, pts_frame.T)
    points_3d = (points_4d[:3] / points_4d[3]).T

    # Reproject points into the second (current) camera
    points_cam2 = (R @ points_3d.T + t).T
    points_proj2, _ = cv2.projectPoints(points_cam2, np.zeros(3), np.zeros(3), K, None)
    points_proj_px = points_proj2.reshape(-1, 2)

    # Compute reprojection errors
    errors = np.linalg.norm(points_proj_px - pts_frame, axis=1)
    print(f"\t\tPrior mean reprojection error: {np.mean(errors):.3f} px")

    # Update the inlier mask
    new_mask_local = errors < reproj_threshold
    full_mask = inlier_match_mask.copy()
    full_mask[np.flatnonzero(inlier_match_mask)[~new_mask_local]] = False

    num_removed_matches = len(pts_ref) - np.sum(new_mask_local)
    remaining_points = np.sum(full_mask)
    print(f"\t\tReprojection filtered: {num_removed_matches}/{len(pts_ref)}")

    # Compute reprojection errors
    errors = np.linalg.norm(points_proj_px - pts_frame, axis=1)
    print(f"\t\tFinal mean reprojection error: {np.mean(errors):.3f} px")

    # Debugging visualization
    if debug:
        reproj_img = frame.img.copy()
        for i in range(len(pts_frame)):
            obs = tuple(np.int32(pts_frame[i]))
            reproj = tuple(np.int32(points_proj_px[i]))
            cv2.circle(reproj_img, obs, 3, (0, 0, 255), -1)      # Observed points (red)
            cv2.circle(reproj_img, reproj, 2, (0, 255, 0), -1)   # Projected points (green)
            cv2.line(reproj_img, obs, reproj, (255, 0, 0), 1)    # Error line (blue)

        debug_img_path = results_dir / f"matches/2-reprojection/{ref_frame.id}_{frame.id}.png"
        debug_img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_img_path), reproj_img)

    return full_mask

def triangulate_points(frame: Frame, ref_frame: Frame, K: np.ndarray, debug: bool = False):
    print(f"Triangulating points between frames {frame.id} & {ref_frame.id}...")
    # Extract the Rotation and Translation arrays between the 2 frames
    pose = frame.match[ref_frame.id]["pose"]
    R = pose[:3, :3]
    t = pose[:3, 3].reshape(3,1)

    # ------------------------------------------------------------------------
    # 6. Triangulate 3D points
    # ------------------------------------------------------------------------

    # Extract inlier matches
    matches = frame.match[ref_frame.id]["matches"]
    inlier_match_mask = frame.match[ref_frame.id]["inlier_match_mask"]

    # Extract keypoint pixel coordinates and indices for both frames from the feature match
    frame_kpt_indices = np.array([m.queryIdx for m in matches])
    frame_kpt_pixels = np.float32([frame.keypoints[idx].pt for idx in frame_kpt_indices])
    ref_frame_kpt_indices = np.array([m.trainIdx for m in matches])
    ref_frame_kpt_pixels = np.float32([ref_frame.keypoints[idx].pt for idx in ref_frame_kpt_indices])

    # Triangulate
    frame_points_3d = triangulate(frame_kpt_pixels, ref_frame_kpt_pixels, R, t, K)
    if frame_points_3d[inlier_match_mask] is None or len(frame_points_3d[inlier_match_mask]) == 0:
        print("[initialize] Triangulation returned no 3D points.")
        return None, None, False

    # ------------------------------------------------------------------------
    # 7. Filter out points with small triangulation angles (cheirality check)
    # ------------------------------------------------------------------------

    valid_angles_mask = filter_triangulation_points(frame_points_3d, R, t)
    if valid_angles_mask is None:
        return None, None, False

    # Combine the Homography/Essential Matrix mask with the valid angles mask
    triangulation_match_mask = inlier_match_mask & valid_angles_mask

    # Combine the triangulation with the feature mask
    frame_triangulation_mask = frame.match[ref_frame.id]["match_mask"]
    ref_frame_triangulation_mask = ref_frame.match[frame.id]["match_mask"]
    for i, is_inlier in enumerate(triangulation_match_mask):
        if is_inlier:
            frame_kp_idx = frame_kpt_indices[i]
            frame_triangulation_mask[frame_kp_idx] = True
            ref_frame_kp_idx = ref_frame_kpt_indices[i]
            ref_frame_triangulation_mask[ref_frame_kp_idx] = True

    # ------------------------------------------------------------------------
    # 8. Save the triangulated points and masks to the frame
    # ------------------------------------------------------------------------

    frame.match[ref_frame.id]["triangulation_mask"] = frame_triangulation_mask
    frame.match[ref_frame.id]["triangulation_match_mask"] = triangulation_match_mask
    frame.match[ref_frame.id]["points"] = frame_points_3d[triangulation_match_mask]

    ref_frame.match[frame.id]["triangulation_mask"] = ref_frame_triangulation_mask
    ref_frame.match[frame.id]["triangulation_match_mask"] = triangulation_match_mask
    ref_frame.match[frame.id]["points"] = frame_points_3d[triangulation_match_mask]

    # Also save the triangulated points keypoint identifiers
    frame_kpt_indices = np.array([m.queryIdx for m in matches])
    frame_kpt_ids = np.float32([frame.keypoints[idx].class_id for idx in frame_kpt_indices])
    frame.match[ref_frame.id]["point_ids"] = frame_kpt_ids[triangulation_match_mask]

    ref_frame_kpt_indices = np.array([m.trainIdx for m in matches])
    ref_frame_kpt_ids = np.float32([ref_frame.keypoints[idx].class_id for idx in ref_frame_kpt_indices])
    ref_frame.match[frame.id]["point_ids"] = ref_frame_kpt_ids[triangulation_match_mask]
            
    # Save the matches
    if debug:
        match_save_path = results_dir / "matches/3-triangulation" / f"{frame.id}_{ref_frame.id}.png"
        plot_matches(frame.img, frame.keypoints,
                     ref_frame.img, ref_frame.keypoints,
                     matches[triangulation_match_mask], match_save_path)

    # Return the initial pose and filtered points
    return frame_points_3d[triangulation_match_mask], frame_kpt_ids[triangulation_match_mask], True
      
def compute_symmetric_transfer_error(E_or_H, frame_kpt_pixels, ref_frame_kpt_pixels, matrix_type='E', K=None):
    errors = []
    num_inliers = 0

    if matrix_type == 'E':
        F = np.linalg.inv(K.T) @ E_or_H @ np.linalg.inv(K)
    else:
        F = np.linalg.inv(K) @ E_or_H @ K

    for i in range(len(frame_kpt_pixels)):
        p1 = np.array([frame_kpt_pixels[i][0], frame_kpt_pixels[i][1], 1])
        p2 = np.array([ref_frame_kpt_pixels[i][0], ref_frame_kpt_pixels[i][1], 1])

        # Epipolar lines
        l2 = F @ p1
        l1 = F.T @ p2

        # Normalize the lines
        l1 /= np.sqrt(l1[0]**2 + l1[1]**2)
        l2 /= np.sqrt(l2[0]**2 + l2[1]**2)

        # Distances
        d1 = abs(p1 @ l1)
        d2 = abs(p2 @ l2)

        error = d1 + d2
        errors.append(error)

        if error < 4.0:  # Threshold for inliers (you may adjust this value)
            num_inliers += 1

    mean_error = np.mean(errors)
    return mean_error, num_inliers

def triangulate(frame_pixels, ref_frame_pixels, R, t, K):
    # Compute projection matrices for triangulation
    M1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # First camera at origin
    M2 = K @ np.hstack((R, t))  # Second camera at R, t

    # Triangulate points
    frame_points_4d_hom = cv2.triangulatePoints(M1, M2, ref_frame_pixels.T, frame_pixels.T)

    # Convert homogeneous coordinates to 3D
    frame_points_3d = frame_points_4d_hom[:3] / frame_points_4d_hom[3]

    return frame_points_3d.T # (N, 3)

def filter_triangulation_points(points_3d, R, t, angle_threshold=1, median_threshold=1):
    """
    Filter out 3D points that have a small triangulation angle between two camera centers.

    When points are triangulated from two views, points that lie almost directly on 
    the line connecting the two camera centers have a very small triangulation angle. 
    These points are often numerically unstable and can degrade the accuracy of the 
    reconstruction. This function discards such points by thresholding the angle to 
    at least 1 degree, and further checks that enough points remain and that the 
    median angle is at least 2 degrees.

    points_3d : (N, 3)
    R, t      : the rotation and translation from camera1 to camera2
    Returns:   valid_angles_mask (bool array of shape (N,)),
               filtered_points_3d (N_filtered, 3) or (None, None)
    """
    print("\t\tFiltering points with small triangulation angles...")
    # -----------------------------------------------------
    # (1) Positive-depth check in both cameras
    # -----------------------------------------------------
    num_points = len(points_3d)

    # points_3d is in the first camera coords => check Z > 0
    Z1 = points_3d[:, 2]
    # Transform into the second camera frame
    points_cam2 = (R @ points_3d.T + t).T  # shape: (N,3)
    Z2 = points_cam2[:, 2]

    # We'll mark True if both Z1, Z2 > 0
    cheirality_mask = (Z1 > 0) & (Z2 > 0)

    # If no point remains, triangulation failed
    num_remaining_points = points_3d[cheirality_mask].shape[0]
    print(f"\t\tCheirality check filtered {num_points - num_remaining_points}/{num_points} points!")
    if num_remaining_points == 0:
        return None, None

    # -----------------------------------------------------
    # (2) Triangulation angle check
    # -----------------------------------------------------

    # Camera centers in the coordinate system where camera1 is at the origin.
    # For convenience, flatten them to shape (3,).
    C1 = np.zeros(3)                 # First camera at origin
    C2 = (-R.T @ t).reshape(3)       # Second camera center in the same coords

    # Vectors from camera centers to each 3D point
    # points_3d is (N, 3), so the result is (N, 3)
    vec1 = points_3d - C1[None, :]   # shape: (N, 3)
    vec2 = points_3d - C2[None, :]   # shape: (N, 3)

    # Compute norms along axis=1 (per row)
    norms1 = np.linalg.norm(vec1, axis=1)  # shape: (N,)
    norms2 = np.linalg.norm(vec2, axis=1)  # shape: (N,)

    # Normalize vectors (element-wise division)
    vec1_norm = vec1 / norms1[:, None]     # shape: (N, 3)
    vec2_norm = vec2 / norms2[:, None]     # shape: (N, 3)

    # Compute dot product along axis=1 to get cos(theta)
    cos_angles = np.sum(vec1_norm * vec2_norm, axis=1)  # shape: (N,)

    # Clip to avoid numerical issues slightly outside [-1, 1]
    cos_angles = np.clip(cos_angles, -1.0, 1.0)

    # Convert to angles in degrees
    angles = np.degrees(np.arccos(cos_angles))  # shape: (N,)

    # Filter out points with triangulation angle < 1 degree
    valid_angles_mask = angles >= angle_threshold

    # Filter points and angles
    filtered_angles = angles[valid_angles_mask]
    num_remaining_points = points_3d[valid_angles_mask].shape[0]
    median_angle = np.median(filtered_angles) if num_remaining_points > 0 else 0

    # Check conditions to decide whether to discard
    print(f"\t\tLow Angles check filtered {num_points - num_remaining_points}/{num_points} points!")
    print(f"\t\tThe median angle is {median_angle:.3f} deg.")

    # If too few points or too small median angle, return None
    triang_mask = valid_angles_mask & cheirality_mask
    num_remaining_points = points_3d[triang_mask].shape[0]
    print(f"\t\tTotal filtering: {num_points-num_remaining_points}/{num_points}")
    print(f"\t\t{num_remaining_points} points left!")

    if num_remaining_points < 40 or median_angle < median_threshold:
        print("Discarding frame due to insufficient triangulation quality.")
        return None, None

    return triang_mask # (N,)
