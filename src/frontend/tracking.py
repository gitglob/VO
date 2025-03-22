import numpy as np
import cv2
from src.frame import Frame
from src.backend.local_map import Map
from src.frontend.initialization import triangulate, filter_triangulation_points, enforce_epipolar_constraint
from src.utils import invert_transform, get_yaw, transform_points
from src.visualize import plot_matches, plot_reprojection
from config import debug, SETTINGS, results_dir


MIN_INLIERS = SETTINGS["PnP"]["min_inliers"]


def get_new_triangulated_points(q_frame: Frame, t_frame: Frame, map: Map, K: np.ndarray):
    """
    Identifies and triangulates new 3D points from feature matches between two frames.

    This function extracts the relative pose between the current and reference frames.
    It then triangulates 3D points from feature correspondences, filters newly observed points, and updates the map.

    Args:
        q_frame (Frame): The previous reference t_frame.
        t_frame (Frame): The current t_frame containing keypoints, descriptors, and matches.
        map (Map): The global map storing existing 3D points.
        K (np.ndarray): The camera intrinsic matrix (3x3).

    Returns:
        Tuple[np.ndarray or None, np.ndarray or None, np.ndarray or None, bool]: 
            - The inverse 4x4 transformation matrix (q_frame -> t_frame) if successful, otherwise None.
            - A (N,3) array of newly triangulated 3D points if successful, otherwise None.
            - A (N,) array of associated point IDs if successful, otherwise None.
            - A boolean indicating whether triangulation was successful.
    """
    if debug:
        print(f"Performing tracking using keyframes {q_frame.id} & {t_frame.id}...")
    
    # ------------------------------------------------------------------------
    # 1. Get keypoint matches and R, t
    # ------------------------------------------------------------------------

    # Extract the matches between the previous and current frame
    matches = q_frame.get_matches(t_frame.id)
    triang_mask = np.ones(len(matches), dtype=bool)

    # Extract keypoint pixel coordinates and indices for both frames from the feature match
    q_kpt_pixels = np.float32([q_frame.keypoints[m.queryIdx].pt for m in matches])
    t_kpt_pixels = np.float32([t_frame.keypoints[m.trainIdx].pt for m in matches])

    # ------------------------------------------------------------------------
    # 2. Enforce Epipolar Constraint
    # ------------------------------------------------------------------------

    epipolar_constraint_mask, _, _ = enforce_epipolar_constraint(q_kpt_pixels, t_kpt_pixels, K)
    if epipolar_constraint_mask is None:
        print("[Tracking] Failed to apply epipolar constraint..")
        return None, None, None, False

    q_frame.match[t_frame.id]["epipolar_constraint_mask"] = epipolar_constraint_mask
    t_frame.match[q_frame.id]["epipolar_constraint_mask"] = epipolar_constraint_mask
    matches = matches[epipolar_constraint_mask]
    triang_mask[triang_mask == True] = epipolar_constraint_mask
    
    # Save the matches
    if debug:
        match_save_path = results_dir / f"matches/6-epipolar_constraint" / f"{q_frame.id}_{t_frame.id}.png"
        plot_matches(q_frame, t_frame, save_path=match_save_path)

    # Extract the q->t transformation
    # Extract the Rotation and Translation arrays between the 2 frames
    T_qt = q_frame.match[t_frame.id]["T"] # [q->t]
    R_qt = T_qt[:3, :3]
    t_qt = T_qt[:3, 3].reshape(3,1)

    # ------------------------------------------------------------------------
    # 3. Find 3D points that haven't been triangulated before
    # ------------------------------------------------------------------------

    # Extract the reference t_frame keypoint ids
    q_kpts = np.array([q_frame.keypoints[m.queryIdx] for m in matches])
    t_kpts = np.array([t_frame.keypoints[m.trainIdx] for m in matches])
    q_kpt_ids = np.array([kpt.class_id for kpt in q_kpts])
    t_kpt_ids = np.array([kpt.class_id for kpt in t_kpts])

    # Find which of the map keypoints don't intersect with the reference inlier keypoint IDs
    new_ids = np.setdiff1d(q_kpt_ids, map.point_ids)
    if len(new_ids) == 0:
        print("No new points to triangulate.")
        return None, None, None, False
    if debug:
        print(f"\t {len(new_ids)} new points to triangulate...")
    
    # Create a mask for the newly triangulated points
    new_points_mask = np.isin(q_kpt_ids, new_ids)

    # Apply the new points mask
    t_kpts = t_kpts[new_points_mask]
    t_kpt_ids = t_kpt_ids[new_points_mask]
    matches = matches[new_points_mask]
    triang_mask[triang_mask == True] = new_points_mask
    
    # ------------------------------------------------------------------------
    # 4. Find the pixel coordinates of the new points
    # ------------------------------------------------------------------------

    # Placeholder for the new keypoint pixels
    q_new_kpt_pixels = []
    t_new_kpt_pixels = []

    # Iterate over all matches
    for m in matches:
        # Extract the reference t_frame index
        q_new_kpt_pixels.append(q_frame.keypoints[m.queryIdx].pt)
        t_new_kpt_pixels.append(t_frame.keypoints[m.trainIdx].pt)

    q_new_kpt_pixels = np.array(q_new_kpt_pixels)
    t_new_kpt_pixels = np.array(t_new_kpt_pixels)
    
    # ------------------------------------------------------------------------
    # 5. Triangulate these points
    # ------------------------------------------------------------------------

    # Triangulate
    q_new_points = triangulate(q_new_kpt_pixels, t_new_kpt_pixels, R_qt, t_qt, K)
    if q_new_points is None or len(q_new_points) == 0:
        print("Triangulation returned no 3D points.")
        return None, None, False

    # Transfer the points to the current coordinate frame [t->q]
    t_new_points = transform_points(q_new_points, T_qt) # (N, 3)

    # ------------------------------------------------------------------------
    # 6. Filter out points with small triangulation angles (cheirality check)
    # ------------------------------------------------------------------------

    filters_mask = filter_triangulation_points(q_new_points, t_new_points, R_qt, t_qt)
    if filters_mask is None or filters_mask.sum() == 0:
        return None, None, False
    matches = matches[filters_mask]
    t_new_points = t_new_points[filters_mask]

    # Extract the ids of the valid triangulated 3d points
    new_points_kpts = t_kpts[filters_mask]

    # Extract the descriptors of the valid triangulated 3d points
    new_point_descriptors = np.array([t_frame.descriptors[m.trainIdx] for m in matches])

    # Combine the triangulation mask with the valid angles mask
    triang_mask[triang_mask == True] = filters_mask

    # ------------------------------------------------------------------------
    # 7. Save the triangulated mask and points to the t_frame
    # ------------------------------------------------------------------------

    t_frame.match[q_frame.id]["triangulation_mask"] = triang_mask
    t_frame.match[q_frame.id]["points"] = t_new_points

    q_frame.match[t_frame.id]["triangulation_mask"] = triang_mask
    q_frame.match[t_frame.id]["points"] = t_new_points
   
    # Return the newly triangulated points
    return t_new_points, new_points_kpts, new_point_descriptors, True

def guided_descriptor_search(map: Map, t_frame: Frame, T_wc: np.ndarray):
    """
    For each map point (with a known 2D projection and a descriptor),
    search within a 'search_window' pixel box in the current t_frame.
    Compare descriptors using Hamming distance and keep the best match
    if it's below the 'distance_threshold'.

    Args:
        map_pxs_in_view (np.ndarray): (N,2) array of projected 2D points for the map points.
        map_desc_in_view (np.ndarray): (N,32) array of the map descriptors (ORB).
        t_frame (Frame): the current t_frame, which has .keypoints and .descriptors
        search_window (int): half-size of the bounding box (in pixels) around (u,v).
        distance_threshold (int): max Hamming distance allowed to accept a match.

    Returns:
        matches (list of tuples): Each element is (map_idx, frame_idx, best_dist) indicating
                                  which map point index matched which t_frame keypoint index,
                                  and the descriptor distance. 
                                  - map_idx in [0..N-1]
                                  - frame_idx in [0..len(t_frame.keypoints)-1]
    """
    ## Extract the normalized camera center vector
    cam_center_vec = T_wc[:3, 3]
    cam_center_vec_norm = cam_center_vec / np.linalg.norm(cam_center_vec)

    # Extract config
    search_window = SETTINGS["guided_search"]["search_window"]
    max_distance = SETTINGS["guided_search"]["max_distance"]
    view_angle_threshold = SETTINGS["guided_search"]["view_angle_threshold"]
    scale_factor = SETTINGS["orb"]["scale_factor"]
    n_levels = SETTINGS["orb"]["level_pyramid"]

    # Prepare results
    matches = []  # list of (map_idx, frame_idx, best_dist)

    # 1) Keep the map points that are in the current frame
    num_points = map.num_points_in_view
    map_points = map.points_in_view

    # For each projected map point, find candidate keypoints in a 2D window
    num_view_angle_f = 0
    num_scale_f = 0
    num_unmatched_kpt = 0
    num_min_dist_f = 0
    for map_idx in range(num_points):
        # Extract the map point, its pixel location, its descriptor
        map_point = map_points[map_idx]  
        (u, v) = map_point.kpt.pt          # (2, )
        map_point_desc = map_point.desc  # (32, )

        # Extract the map point 3d position and its distance from the map origin
        map_point_pos = map_point.pos    # (3, )
        map_point_dist = np.linalg.norm(map_point_pos)

        ## Extract the angle between the 2 vectors
        map_point_pos_norm = map_point_pos / map_point_dist
        dot_product = np.clip(np.dot(cam_center_vec_norm, map_point_pos_norm), -1.0, 1.0)
        map_point_view_angle = np.degrees(np.arccos(dot_product))

        # 2) Check if the viewing angle is less than the threshold
        if map_point_view_angle < view_angle_threshold:
            num_view_angle_f += 1
            continue

        # Extract the ORB scale invariance limits for that map point
        dmin, dmax = map_point.getScaleInvarianceLimits()

        # 3) Check if the map_point distance is in the scale invariance region
        if map_point_dist < dmin or map_point_dist > dmax:
            num_scale_f += 1
            continue

        # 4) Compute the scale in the frame
        t_scale = map_point_dist/dmin
    
        # Compute the predicted pyramid level from t_scale.
        predicted_level = int(round(np.log(t_scale) / np.log(scale_factor)))
        # Clamp predicted_level to valid range
        predicted_level = max(0, min(predicted_level, n_levels - 1))

        # Collect candidate keypoint indices within the bounding box
        candidate_indices = []
        u_min, u_max = u - search_window, u + search_window
        v_min, v_max = v - search_window, v + search_window

        # Iterate over all the t_frame keypoints
        for kpt_idx, kpt in enumerate(t_frame.keypoints):
            (u_kpt, v_kpt) = kpt.pt

            # Check if the keypoint is within the search window box
            if (u_kpt < u_min or u_kpt > u_max or
                v_kpt < v_min or v_kpt > v_max):
                continue
    
            # Check if the keypoint's pyramid level matches the predicted level (or is within an acceptable tolerance)
            # For a strict match:
            if kpt.octave != predicted_level:
                continue

            # Keep that frame candidate
            candidate_indices.append(kpt_idx)

        # No keypoints found near the projected point
        if len(candidate_indices) == 0:
            num_unmatched_kpt += 1
            continue

        # Find the best match among candidates by Hamming distance
        best_dist = float("inf")
        best_f_idx = -1

        # Iterate over the frame keypoint candidates
        for f_kpt_idx in candidate_indices:
            frame_point_desc = t_frame.descriptors[f_kpt_idx]  # (32,)
            # Compute Hamming distance
            dist = cv2.norm(map_point_desc, frame_point_desc, cv2.NORM_HAMMING)
            if dist < best_dist:
                best_dist = dist
                best_f_idx = f_kpt_idx

        # Accept the best match if below threshold
        if best_dist > max_distance:
            num_min_dist_f += 1
            continue

        matches.append((map_idx, best_f_idx, best_dist))

    if debug:
        print(f"\t Guided Search Filtering: ",
              f"\n\t\t View Angle: {num_view_angle_f}",
              f"\n\t\t Scale: {num_scale_f}",
              f"\n\t\t Unmatched: {num_unmatched_kpt}",
              f"\n\t\t Max Dist: {num_min_dist_f}")
        print(f"\t Found {len(matches)} guided descriptor matches.")

    return matches

def predict_pose_constant_velocity(poses):
    """
    Given a list of past poses, predict the next pose assuming constant velocity.
    Velocity is calculated in SE(3) as: delta = T_{k-1}^-1 * T_k
    Then the prediction is: T_{k+1}^pred = T_k * delta
    If there are fewer than 2 poses, just return the last pose as a fallback.
    """
    assert(len(poses) >= 2)

    curr_pose = poses[-1] # T_{k}->{world}
    prev_pose = poses[-2] # T_{k-1}->{world}
    T_prev_curr = prev_pose @ invert_transform(curr_pose) # T_{k-1}->{k} == T_{k}->{k+1}

    pred_pose_wc = invert_transform(curr_pose) @ T_prev_curr # T_{world}->{k+1}
    pred_pose = invert_transform(pred_pose_wc) # T_{k+1}->{world}

    return pred_pose

# Function to estimate the relative pose using solvePnP
def estimate_relative_pose(
    map: Map,
    t_frame: Frame,
    guided_matches: list,
    K: np.ndarray,
    dist_coeffs=None
):
    """
    Estimate the relative camera displacement using a 3D-2D PnP approach.

    Args:
        map_points_w (np.ndarray): 
            (N, 3) array of 3D map points in world coordinates
            that correspond to the 'map_idx' indices in guided_matches.
        t_frame (Frame): 
            The current t_frame containing keypoints and descriptors.
        guided_matches (list of (int, int, float)): 
            A list of (map_idx, frame_idx, best_dist) from guided_descriptor_search().
            - map_idx is the index into map_points_w
            - frame_idx is the index of t_frame.keypoints
            - best_dist is the matching descriptor distance (not used here except for reference).
        K (np.ndarray): 
            (3, 3) camera intrinsic matrix.
        dist_coeffs:
            Distortion coefficients for the camera. Default = None.

    Returns:
        (displacement, rmse):
            displacement (np.ndarray): 4×4 transformation matrix T_{cam_new <- cam_old}.
                                       i.e., the relative transform from the old camera t_frame
                                       to the new camera t_frame.
            rmse (float): root-mean-squared reprojection error over the inliers.

        If the function fails, returns (None, None).
    """
    map_points_w = map.points_in_view
    num_points = len(guided_matches)
    print(f"Estimating Map -> Frame #{t_frame.id} pose using {num_points}/{len(map_points_w)} map points...")

    # 1) Build 3D <-> 2D correspondences
    map_points = []
    image_pxs = []
    for (map_idx, frame_idx, best_dist) in guided_matches:
        map_points.append(map_points_w[map_idx])        # 3D in world coords
        kp = t_frame.keypoints[frame_idx]
        image_pxs.append(kp.pt)                         # 2D pixel (u, v)

    map_points = np.array(map_points, dtype=object)   
    map_point_positions = np.array([p.pos for p in map_points], dtype=np.float32) # (M, 3)
    image_pxs = np.array(image_pxs, dtype=np.float32)     # (M, 2)

    # 2) solvePnPRansac to get rvec/tvec for world->new_cam
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        map_point_positions,
        image_pxs,
        cameraMatrix=K,
        distCoeffs=dist_coeffs,
        reprojectionError=SETTINGS["PnP"]["reprojection_threshold"],
        confidence=SETTINGS["PnP"]["confidence"],
        iterationsCount=SETTINGS["PnP"]["iterations"]
    )
    if not success or inliers is None or len(inliers) < MIN_INLIERS:
        print("\t solvePnP failed or not enough inliers.")
        return None
    
    t_wc = tvec.flatten()
    R_wc, _ = cv2.Rodrigues(rvec)
    inliers = inliers.flatten()
    inliers_mask = np.zeros(num_points, dtype=bool)
    inliers_mask[inliers] = True
    num_tracked_points = inliers_mask.sum()
    if debug:
        print(f"\t solvePnPRansac filtered {num_points - num_tracked_points}/{num_points} points.")
    
    # Keep only the pixels and points that match the estimated pose
    image_pxs = image_pxs[inliers_mask]
    map_points = map_points[inliers_mask]
    map_point_positions = map_point_positions[inliers_mask]

    # Invrease the match counter for all matched points
    for p in map_points:
        p.match_counter += 1

    # 3) Compute reprojection error
    ## Project the 3D points to 2D using the estimated pose
    projected_world_pxs, _ = cv2.projectPoints(map_point_positions, rvec, t_wc, K, dist_coeffs)
    projected_world_pxs = projected_world_pxs.squeeze()
    
    ## Calculate the per-point reprojection error (Euclidean distance)
    errors = np.sqrt(np.sum((image_pxs - projected_world_pxs)**2, axis=1))
    
    ## Create a mask for points with error less than the threshold
    reproj_mask = errors < SETTINGS["PnP"]["reprojection_threshold"]
    error = np.mean(errors)
    if debug:
        print(f"\t Reprojection error ({error:.2f}) filtered {len(map_points) - reproj_mask.sum()}/{len(map_points)} points.")

    ## Visualization
    if debug:
        img_path = results_dir / f"matches/4-PnP_reprojection/map_{t_frame.id}.png"
        plot_reprojection(t_frame.img, image_pxs, projected_world_pxs, path=img_path)

    # 7) Construct T_{world->cam_new}
    T_wc = np.eye(4, dtype=np.float32)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = t_wc

    return T_wc, num_tracked_points

def is_keyframe(T: np.ndarray, num_tracked_points: int):
    """ Determine if motion expressed by t, R is significant by comparing to tresholds. """
    tx = T[0, 3] # The x component points right
    ty = T[1, 3] # The y component points down
    tz = T[2, 3] # The z component points forward

    trans = np.sqrt(tx**2 + ty**2 + tz**2)
    angle = abs(get_yaw(T[:3, :3]))

    # is_keyframe = trans > SETTINGS["keyframe"]["distance"] or angle > SETTINGS["keyframe"]["angle"]
    is_keyframe = num_tracked_points > SETTINGS["keyframe"]["num_tracked_points"]
    if debug:
        print(f"\t Tracked points: {num_tracked_points}, dist: {trans:.3f}, angle: {angle:.3f}")
        if is_keyframe:
            print("\t\t Keyframe!")
        else:
            print("\t\t Not a keyframe!")

    return is_keyframe
