import numpy as np
import cv2
from src.frame import Frame
from src.backend.local_map import Map
from src.frontend.initialization import triangulate, filter_triangulation_points, enforce_epipolar_constraint
from src.utils import invert_transform, get_yaw, transform_points
from src.visualize import plot_matches
from config import debug, SETTINGS, results_dir


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
        return None, False

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
    q_kpt_ids = np.array([q_frame.keypoints[m.queryIdx].class_id for m in matches])

    # Find which of the map keypoints don't intersect with the reference inlier keypoint IDs
    new_ids = np.setdiff1d(q_kpt_ids, map.point_ids)
    if len(new_ids) == 0:
        print("No new points to triangulate.")
        return None, None, False
    if debug:
        print(f"\t {len(new_ids)} new points to triangulate...")
    
    # Create a mask for the newly triangulated points
    new_points_mask = np.isin(q_kpt_ids, new_ids)

    # Apply the new points mask
    q_kpt_ids = q_kpt_ids[new_points_mask]
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
    t_new_points = t_new_points[filters_mask]

    # Extract the ids of the valid triangulated 3d points
    new_points_ids = q_kpt_ids[filters_mask]

    # Combine the triangulation mask with the valid angles mask
    triang_mask[triang_mask == True] = filters_mask

    # ------------------------------------------------------------------------
    # 7. Save the triangulated mask and points to the t_frame
    # ------------------------------------------------------------------------

    t_frame.match[q_frame.id]["triangulation_match_mask"] = triang_mask
    t_frame.match[q_frame.id]["points"] = t_new_points
    t_frame.match[q_frame.id]["point_ids"] = new_points_ids

    q_frame.match[t_frame.id]["triangulation_match_mask"] = triang_mask
    q_frame.match[t_frame.id]["points"] = t_new_points
    q_frame.match[t_frame.id]["point_ids"] = new_points_ids
   
    # Return the newly triangulated points
    return t_new_points, new_points_ids, True

def guided_descriptor_search(
    map_pxs_in_view: np.ndarray,
    map_desc_in_view: np.ndarray,
    t_frame: Frame,
    search_window: int = 40,
    distance_threshold: int = 100
):
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

    # Current t_frame data
    frame_kpts = t_frame.keypoints
    frame_desc = t_frame.descriptors

    # Prepare results
    matches = []  # list of (map_idx, frame_idx, best_dist)

    # For each projected map point, find candidate keypoints in a 2D window
    for map_idx, (u, v) in enumerate(map_pxs_in_view):
        desc_3d = map_desc_in_view[map_idx]  # shape (32, )

        # Collect candidate keypoint indices within the bounding box
        candidate_indices = []
        u_min, u_max = u - search_window, u + search_window
        v_min, v_max = v - search_window, v + search_window

        # Iterate over all the t_frame keypoints
        for f_idx, kpt in enumerate(frame_kpts):
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
            desc_2d = frame_desc[f_idx]  # shape (32,)
            # Compute Hamming distance
            dist = cv2.norm(desc_3d, desc_2d, cv2.NORM_HAMMING)
            if dist < best_dist:
                best_dist = dist
                best_f_idx = f_idx

        # Accept the best match if below threshold
        if best_dist < distance_threshold:
            matches.append((map_idx, best_f_idx, best_dist))

    if debug:
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
    map_points_w: np.ndarray,
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

    print(f"Estimating Map -> Frame #{t_frame.id} pose using {len(map_points_w)} map points...")
    num_points = len(guided_matches)

    # 1) Check if enough 3D points exist
    if num_points < 6:
        print("\t Warning: Not enough points for pose estimation. Expected at least 6.")
        return None

    # 2) Build 3D <-> 2D correspondences
    world_points = []
    image_pxs = []
    for (map_idx, frame_idx, best_dist) in guided_matches:
        world_points.append(map_points_w[map_idx])        # 3D in world coords
        kp = t_frame.keypoints[frame_idx]
        image_pxs.append(kp.pt)                          # 2D pixel (u, v)

    world_points = np.array(world_points, dtype=np.float32)   # (M, 3)
    image_pxs = np.array(image_pxs, dtype=np.float32)     # (M, 2)

    # 3) solvePnPRansac to get rvec/tvec for world->new_cam
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        world_points,
        image_pxs,
        cameraMatrix=K,
        distCoeffs=dist_coeffs,
        reprojectionError=SETTINGS["PnP"]["reprojection_threshold"],
        confidence=SETTINGS["PnP"]["confidence"],
        iterationsCount=SETTINGS["PnP"]["iterations"]
    )
    if not success or inliers is None or len(inliers) < 6:
        print("\t solvePnP failed or not enough inliers.")
        return None
    
    t_wc = tvec.flatten()
    R_wc, _ = cv2.Rodrigues(rvec)
    inliers = inliers.flatten()
    inliers_mask = np.zeros(num_points, dtype=bool)
    inliers_mask[inliers] = True
    if debug:
        print(f"\t solvePnPRansac filtered {num_points - np.sum(inliers_mask)}/{num_points} points.")
    
    # Keep only the pixels and points that match the estimated pose
    image_pxs = image_pxs[inliers_mask]
    world_points = world_points[inliers_mask]

    # 4) Compute reprojection error
    ## Project the 3D points to 2D using the estimated pose
    projected_world_pxs, _ = cv2.projectPoints(world_points, rvec, t_wc, K, dist_coeffs)
    projected_world_pxs = projected_world_pxs.squeeze()
    
    ## Calculate the per-point reprojection error (Euclidean distance)
    errors = np.sqrt(np.sum((image_pxs - projected_world_pxs)**2, axis=1))
    
    ## Create a mask for points with error less than the threshold
    reproj_mask = errors < SETTINGS["PnP"]["reprojection_threshold"]
    error = np.mean(errors)
    if debug:
        print(f"\t Reprojection error ({error:.2f}) filtered {num_points - reproj_mask.sum()}/{num_points} points.")

    ## Visualization
    if debug:
        reproj_img = cv2.cvtColor(t_frame.img, cv2.COLOR_GRAY2BGR)
        for i in range(len(image_pxs)):
            obs = tuple(np.int32(image_pxs[i]))
            reproj = tuple(np.int32(projected_world_pxs[i]))
            cv2.circle(reproj_img, obs, 3, (0, 0, 255), -1)      # Observed points (red)
            cv2.circle(reproj_img, reproj, 2, (0, 255, 0), -1)   # Projected points (green)
            cv2.line(reproj_img, obs, reproj, (255, 0, 0), 1)    # Error line (blue)

        debug_img_path = results_dir / f"matches/4-PnP_reprojection/map_{t_frame.id}.png"
        debug_img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_img_path), reproj_img)

    # 7) Construct T_{world->cam_new}
    T_wc = np.eye(4, dtype=np.float32)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = t_wc

    return T_wc

def is_keyframe(T: np.ndarray):
    """ Determine if motion expressed by t, R is significant by comparing to tresholds. """
    tx = T[0, 3] # The x component points right
    ty = T[1, 3] # The y component points down
    tz = T[2, 3] # The z component points forward

    trans = np.sqrt(tx**2 + ty**2 + tz**2)
    angle = abs(get_yaw(T[:3, :3]))

    is_keyframe = trans > SETTINGS["keyframe"]["distance"] or angle > SETTINGS["keyframe"]["angle"]
    if debug:
        print(f"\t Displacement: dist: {trans:.3f}, angle: {angle:.3f}")
        if is_keyframe:
            print("\t\t Keyframe!")
        else:
            print("\t\t Not a keyframe!")

    return is_keyframe
