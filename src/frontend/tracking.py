from typing import List
import numpy as np
import cv2
from src.frame import Frame
from src.backend.local_map import Map
from src.frontend.initialization import compute_symmetric_transfer_error, triangulate, filter_triangulation_points
from src.utils import invert_transform, transform_points


def get_new_triangulated_points(frame: Frame, ref_frame: Frame, map: Map, K: np.ndarray):
    """
    Identifies and triangulates new 3D points from feature matches between two frames.

    This function estimates the relative pose between the current and reference frames
    using either the Essential or Homography matrix. It then triangulates 3D points
    from feature correspondences, filters newly observed points, and updates the map.

    Args:
        frame (Frame): The current frame containing keypoints, descriptors, and matches.
        ref_frame (Frame): The previous reference frame.
        map (Map): The global map storing existing 3D points.
        K (np.ndarray): The camera intrinsic matrix (3x3).

    Returns:
        Tuple[np.ndarray or None, np.ndarray or None, np.ndarray or None, bool]: 
            - The inverse 4x4 transformation matrix (ref_frame -> frame) if successful, otherwise None.
            - A (N,3) array of newly triangulated 3D points if successful, otherwise None.
            - A (N,) array of associated point IDs if successful, otherwise None.
            - A boolean indicating whether triangulation was successful.
    """
    print(f"Performing tracking using keyframes {frame.id} & {ref_frame.id}...")
    
    # ------------------------------------------------------------------------
    # 1. Get keypoint matches
    # ------------------------------------------------------------------------

    # Extract the matches between the previous and current ref_frame
    matches = frame.get_matches(ref_frame.id)
    if len(matches) < 5:
        print("Not enough matches to compute the Essential Matrix!")
        return None, None, None, False

    # Extract keypoint pixel coordinates and indices for both frames from the feature match
    frame_kpt_indices = np.array([m.queryIdx for m in matches])
    frame_kpt_pixels = np.float32([frame.keypoints[idx].pt for idx in frame_kpt_indices])
    ref_frame_kpt_indices = np.array([m.trainIdx for m in matches])
    ref_frame_kpt_pixels = np.float32([ref_frame.keypoints[idx].pt for idx in ref_frame_kpt_indices])

    # Placeholder for the triangulation mask
    triangulation_mask = np.ones(len(matches), dtype=bool)

    # ------------------------------------------------------------------------
    # 2. Compute Essential & Homography matrices
    # ------------------------------------------------------------------------

    # Compute the Essential Matrix
    E, mask_E = cv2.findEssentialMat(frame_kpt_pixels, ref_frame_kpt_pixels, K, method=cv2.RANSAC, prob=0.99, threshold=1.5)
    mask_E = mask_E.ravel().astype(bool)

    # Compute the Homography Matrix
    H, mask_H = cv2.findHomography(frame_kpt_pixels, ref_frame_kpt_pixels, method=cv2.RANSAC, ransacReprojThreshold=3.0)
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
    inlier_mask = mask_H if use_homography else mask_E
    inlier_frame_pixels = frame_kpt_pixels[inlier_mask]
    inlier_ref_frame_pixels = ref_frame_kpt_pixels[inlier_mask]

    # Placeholders for R, t and the final triangulation mask
    R, t = None, None
    mask_pose = np.ones_like(inlier_mask, dtype=bool)

    # Check if we will use homography
    if not use_homography:
        # Decompose Essential Matrix
        points, R_est, t_est, mask_pose = cv2.recoverPose(E, inlier_frame_pixels, inlier_ref_frame_pixels, K)

        # mask_pose indicates inliers used in cv2.recoverPose (1 for inliers, 0 for outliers)
        mask_pose = mask_pose.ravel().astype(bool)

        if R_est is not None and t_est is not None and np.any(mask_pose):
            R, t = R_est, t_est
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
                # Current ref_frame pixel in camera coords
                p_curr_cam = invK @ np.array([*inlier_ref_frame_pixels[j], 1.0])  
                # Previous ref_frame pixel in camera coords
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

    # If we failed to recover R and t
    if R is None or t is None:
        print("[update_map] Failed to get epipolar constraint inliers.")
        return None, None, None, False

    # Create a triangulation_mask but combining the epipolar constraint and transformation fitting checks
    triangulation_mask[inlier_mask] = mask_pose

    # ------------------------------------------------------------------------
    # 5. Build the 4x4 Pose matrix
    # ------------------------------------------------------------------------
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t.flatten()
    inv_pose = invert_transform(pose)
    
    # Initialize the frames
    frame.triangulate(ref_frame.id, use_homography, triangulation_mask, pose, "tracking")
    ref_frame.triangulate(frame.id, use_homography, triangulation_mask, inv_pose, "tracking")

    # ------------------------------------------------------------------------
    # 6. Find 3D points that haven't been triangulated before
    # ------------------------------------------------------------------------

    # Extract the reference frame keypoint ids
    ref_frame_kpt_ids = np.array([ref_frame.keypoints[idx].class_id for idx in ref_frame_kpt_indices])

    # Find which of the map keypoints don't intersect with the reference inlier keypoint IDs
    new_ids = np.setdiff1d(ref_frame_kpt_ids[triangulation_mask], map.point_ids)

    print(f"{len(new_ids)} new points to triangulate...")

    if len(new_ids) == 0:
        print("[update_map] No new points to triangulate.")
        return None, None, None, False
    
    # Create a mask for the newly triangulated points
    new_3d_points_mask = np.isin(ref_frame_kpt_ids[triangulation_mask], new_ids)

    # Combine the inlier and new triangulation mask
    triangulation_mask[triangulation_mask] &= new_3d_points_mask
    
    # ------------------------------------------------------------------------
    # 7. Find the pixel coordinates of the new points
    # ------------------------------------------------------------------------

    # Placeholder for the new keypoint pixels
    ref_frame_kpt_pixels = []
    frame_kpt_pixels = []

    # Iterate over all matches
    for m in matches[triangulation_mask]:
        # Extract the reference frame index
        idx = m.trainIdx
        ref_frame_kpt_pixels.append(ref_frame.keypoints[idx].pt)
        frame_kpt_pixels.append(frame.keypoints[idx].pt)

    ref_frame_kpt_pixels = np.array(ref_frame_kpt_pixels)
    frame_kpt_pixels = np.array(frame_kpt_pixels)
    
    # ------------------------------------------------------------------------
    # 8. Triangulate these points
    # ------------------------------------------------------------------------

    # Triangulate
    frame_points_3d = triangulate(frame_kpt_pixels, ref_frame_kpt_pixels, R, t, K)
    if frame_points_3d is None or len(frame_points_3d) == 0:
        print("[tracking] Triangulation returned no 3D points.")
        return None, None, None, False

    # ------------------------------------------------------------------------
    # 9. Filter out points with small triangulation angles (cheirality check)
    # ------------------------------------------------------------------------

    valid_angles_mask = filter_triangulation_points(frame_points_3d, R, t)
    if valid_angles_mask is None:
        return None, None, None, False
    frame_points_3d = frame_points_3d[valid_angles_mask]

    # Combine the triangulation mask with the valid angles mask
    triangulation_mask[triangulation_mask] = valid_angles_mask

    # Extract the ids of the valid triangulated 3d points
    frame_points_ids = ref_frame_kpt_ids[triangulation_mask]

    # ------------------------------------------------------------------------
    # 10. Save the triangulated mask and points to the frame
    # ------------------------------------------------------------------------

    frame.match[ref_frame.id]["triangulation_match_mask"] = triangulation_mask
    frame.match[ref_frame.id]["points"] = frame_points_3d
    frame.match[ref_frame.id]["point_ids"] = frame_points_ids

    ref_frame.match[frame.id]["triangulation_match_mask"] = triangulation_mask
    ref_frame.match[frame.id]["points"] = frame_points_3d
    ref_frame.match[frame.id]["point_ids"] = frame_points_ids
   
    # Return the newly triangulated points
    return inv_pose, frame_points_3d, frame_points_ids, True

def guided_descriptor_search(
    points_in_view_2d: np.ndarray,
    descriptors_in_view: np.ndarray,
    frame: Frame,
    search_window: int = 20,
    distance_threshold: int = 100
):
    """
    For each map point (with a known 2D projection and a descriptor),
    search within a 'search_window' pixel box in the current frame.
    Compare descriptors using Hamming distance and keep the best match
    if it's below the 'distance_threshold'.

    Args:
        points_in_view_2d (np.ndarray): (N,2) array of projected 2D points for the map points.
        descriptors_in_view (np.ndarray): (N,32) array of the map descriptors (ORB).
        frame (Frame): the current frame, which has .keypoints and .descriptors
        search_window (int): half-size of the bounding box (in pixels) around (u,v).
        distance_threshold (int): max Hamming distance allowed to accept a match.

    Returns:
        matches (list of tuples): Each element is (map_idx, frame_idx, best_dist) indicating
                                  which map point index matched which frame keypoint index,
                                  and the descriptor distance. 
                                  - map_idx in [0..N-1]
                                  - frame_idx in [0..len(frame.keypoints)-1]
    """

    # Current frame data
    frame_keypoints = frame.keypoints
    frame_descriptors = frame.descriptors

    # Prepare results
    matches = []  # list of (map_idx, frame_idx, best_dist)

    # For each projected map point, find candidate keypoints in a 2D window
    for map_idx, (u, v) in enumerate(points_in_view_2d):
        desc_3d = descriptors_in_view[map_idx]  # shape (32, )

        # Collect candidate keypoint indices within the bounding box
        candidate_indices = []
        u_min, u_max = u - search_window, u + search_window
        v_min, v_max = v - search_window, v + search_window

        # Iterate over all the frame keypoints
        for f_idx, kpt in enumerate(frame_keypoints):
            (x_kp, y_kp) = kpt.pt  # Keypoint location
            # Check if the keypoint is within the search window box
            if (x_kp >= u_min and x_kp <= u_max and
                y_kp >= v_min and y_kp <= v_max):
                candidate_indices.append(f_idx)

        if len(candidate_indices) == 0:
            # No keypoints found near the projected point
            continue

        # Find the best match among candidates by Hamming distance
        best_dist = float("inf")
        best_f_idx = -1

        for f_idx in candidate_indices:
            desc_2d = frame_descriptors[f_idx]  # shape (32,)
            # Compute Hamming distance
            dist = cv2.norm(desc_3d, desc_2d, cv2.NORM_HAMMING)
            if dist < best_dist:
                best_dist = dist
                best_f_idx = f_idx

        # Accept the best match if below threshold
        if best_dist < distance_threshold:
            matches.append((map_idx, best_f_idx, best_dist))

    return matches

def predict_pose_constant_velocity(poses):
    """
    Given a list of past poses, predict the next pose assuming constant velocity.
    Velocity is calculated in SE(3) as: delta = T_{k-1}^-1 * T_k
    Then the prediction is: T_{k+1}^pred = T_k * delta
    If there are fewer than 2 poses, just return the last pose as a fallback.
    """
    if len(poses) < 2:
        return poses[-1]  # No velocity info, fallback to last pose

    prev_pose = poses[-2]
    curr_pose = poses[-1]
    delta = invert_transform(prev_pose) @ curr_pose # Relative motion T_{k-1}^-1 * T_k
    pred_pose = curr_pose @ delta                   # T_k * delta
    return pred_pose

# Function to estimate the relative pose using solvePnP
def estimate_relative_pose(
    map_points_3d: np.ndarray,
    frame: Frame,
    guided_matches: list,
    old_pose: np.ndarray,   # <-- old camera pose in world coords (4x4)
    K: np.ndarray,
    debug: bool = False,
    dist_coeffs=None
):
    """
    Estimate the relative camera displacement using a 3D-2D PnP approach.

    Args:
        map_points_3d (np.ndarray): 
            (N, 3) array of 3D map points in world coordinates
            that correspond to the 'map_idx' indices in guided_matches.
        frame (Frame): 
            The current frame containing keypoints and descriptors.
        guided_matches (list of (int, int, float)): 
            A list of (map_idx, frame_idx, best_dist) from guided_descriptor_search().
            - map_idx is the index into map_points_3d
            - frame_idx is the index of frame.keypoints
            - best_dist is the matching descriptor distance (not used here except for reference).
        old_pose (np.ndarray):
            (4,4) The previous camera pose in world coords, i.e. T_{cam_old <- world}.
        K (np.ndarray): 
            (3, 3) camera intrinsic matrix.
        debug (bool): 
            If True, prints debug info.
        dist_coeffs:
            Distortion coefficients for the camera. Default = None.

    Returns:
        (displacement, rmse):
            displacement (np.ndarray): 4×4 transformation matrix T_{cam_new <- cam_old}.
                                       i.e., the relative transform from the old camera frame
                                       to the new camera frame.
            rmse (float): root-mean-squared reprojection error over the inliers.

        If the function fails, returns (None, None).
    """

    print(f"Estimating relative pose in frame {frame.id} using {len(map_points_3d)} map points...")

    # 1) Check if enough 3D points exist
    if len(guided_matches) < 6:
        print("\tWarning: Not enough points for pose estimation. Expected at least 6.")
        return None, None

    # 2) Build 3D <-> 2D correspondences
    object_points = []
    image_pixels = []
    for (map_idx, frame_idx, best_dist) in guided_matches:
        object_points.append(map_points_3d[map_idx])        # 3D in world coords
        kp = frame.keypoints[frame_idx]
        image_pixels.append(kp.pt)                          # 2D pixel (u, v)

    object_points = np.array(object_points, dtype=np.float32)   # (M, 3)
    image_pixels = np.array(image_pixels, dtype=np.float32)     # (M, 2)

    # 3) solvePnPRansac to get rvec/tvec for world->new_cam
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,       # 3D (world coords)
        image_pixels,        # 2D
        K,
        distCoeffs=dist_coeffs,
        reprojectionError=3.0,
        confidence=0.99,
        iterationsCount=1000
    )
    if not success or inliers is None or len(inliers) < 6:
        print("\tsolvePnP failed or not enough inliers.")
        return None, None

    # 4) Compute reprojection error (before refinement)
    error = compute_reprojection_error(object_points[inliers], image_pixels[inliers],
                                       rvec, tvec, K, dist_coeffs)
    if debug:
        print(f"\tReprojection error (before refine): {error:.2f} pixels")

    # 5) Refine with inliers only (no RANSAC)
    success_refined, rvec_refined, tvec_refined = cv2.solvePnP(
        object_points[inliers], image_pixels[inliers],
        K, dist_coeffs, rvec, tvec, useExtrinsicGuess=True
    )
    if not success_refined:
        print("\tsolvePnP refinement failed.")
        return None, None

    # 6) Compute the refined reprojection error
    error_refined = compute_reprojection_error(object_points[inliers],
                                               image_pixels[inliers],
                                               rvec_refined, tvec_refined,
                                               K, dist_coeffs)
    if debug:
        print(f"\tRefined reprojection error: {error_refined:.2f} pixels")

    # 7) Construct T_{world->cam_new}
    R_wc, _ = cv2.Rodrigues(rvec_refined)  # rotation matrix
    T_wc = np.eye(4, dtype=np.float32)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = tvec_refined.ravel()

    # 8) We want T_{cam_new <- world} = T_cw_new, so invert T_wc
    T_cw_new = invert_transform(T_wc)

    # 9) The old pose is T_cw_old = (cam_old->world)
    #    The new pose is T_cw_new = (cam_new->world)
    #
    #    So the "displacement" transform we want is
    #        T_{cam_new <- cam_old} = T_cw_old^{-1} * T_cw_new
    #    because
    #        T_cw_new = T_cw_old * T_{cam_new <- cam_old}.
    #
    #    If `old_pose` was given as T_cw_old, we can compute:
    old_pose_inv = invert_transform(old_pose)  # T_wc_old = (world->cam_old)
    # Actually, if old_pose is T_cw_old, then invert it to get T_wc_old.
    T_cnew_cold = old_pose_inv @ T_cw_new

    # 10) Print or debug
    if debug:
        # The displacement is from cam_old to cam_new
        dx, dy, dz = T_cnew_cold[:3, 3]
        yaw_deg = np.degrees(np.arctan2(T_cnew_cold[1, 0], T_cnew_cold[0, 0]))
        print(f"\tDisplacement: dx:{dx:.3f}, dy:{dy:.3f}, dz:{dz:.3f}, yaw:{yaw_deg:.3f}")

    return T_cnew_cold, error_refined
    
def compute_reprojection_error(pts_3d, pts_2d, rvec, tvec, K, dist_coeffs):
    """Compute the reprojection error for the given 3D-2D point correspondences and pose."""
    # Project the 3D points to 2D using the estimated pose
    projected_pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist_coeffs)
    
    # Calculate the reprojection error
    error = np.sqrt(np.mean(np.sum(((pts_2d - projected_pts_2d).squeeze()) ** 2, axis=1)))
    
    return error

def is_keyframe(P, t_threshold=0.3, yaw_threshold=1, debug=False):
    """ Determine if motion expressed by t, R is significant by comparing to tresholds. """
    R = P[:3, :3]
    t = P[:3, 3]

    dx = abs(t[0])
    dy = abs(t[1])
    yaw_deg = abs(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

    is_keyframe = dx > t_threshold or dy > t_threshold or yaw_deg > yaw_threshold
    if debug:
        if is_keyframe:
            print("\t\tKeyframe!")
        else:
            print("\t\tNot a keyframe!")

    return is_keyframe
