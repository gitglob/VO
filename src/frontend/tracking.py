import numpy as np
import cv2
from scipy.linalg import expm, logm
from src.others.local_map import Map, mapPoint
from src.frontend.initialization import triangulate
from src.others.frame import Frame
from src.others.utils import get_yaw, transform_points, invert_transform
from src.others.visualize import plot_matches, plot_reprojection, plot_pixels
from src.others.filtering import filter_triangulation_points, enforce_epipolar_constraint, filter_by_reprojection, filter_scale
from config import SETTINGS, results_dir


debug = SETTINGS["generic"]["debug"]
MIN_INLIERS = SETTINGS["PnP"]["min_inliers"]
W = SETTINGS["camera"]["width"]
H = SETTINGS["camera"]["height"]
MIN_NUM_MATCHES = SETTINGS["point_association"]["num_matches"]
SEARCH_WINDOW = SETTINGS["point_association"]["search_window"]
HAMMING_THRESHOLD = SETTINGS["point_association"]["hamming_threshold"]


def constant_velocity_model(t: float, times: list, poses: list):
    # Find how much time has passed
    dt_c = t - times[-1]

    # Find the previous dr
    dt_tq = times[-1] - times[-2]

    # Find the previous relative transformation
    T_wq = np.linalg.inv(poses[-2])
    T_tw = poses[-1]
    T_tq = T_wq @ T_tw

    # Use the matrix logarithm to obtain the twist (in se(3)) corresponding to T_rel.
    twist_matrix = logm(T_tq)
    
    # Scale the twist for the prediction time interval.
    scaled_twist = twist_matrix * (dt_c / dt_tq)
    
    # Obtain the predicted incremental transformation via the matrix exponential.
    T_ct = expm(scaled_twist)
    
    # The predicted current pose is T_last followed by the predicted incremental transformation.
    T_cw = T_tw @ T_ct
    T_wc = invert_transform(T_cw)

    return T_wc

def project_point(p_w: np.ndarray, T_wc: np.ndarray, K: np.ndarray):
    """
    Projects a 3D point from world coordinates into the image given a camera pose and intrinsic matrix.
    
    Args:
        p_w: A 3-element iterable representing the 3D point (e.g. np.array([x,y,z])).
        T_wc: A 4x4 homogeneous transformation matrix mapping world coordinates to camera coordinates.
        K:   The 3x3 camera intrinsic matrix.
    
    Returns:
        A tuple (u, v) representing the predicted pixel coordinate or None if the point projects behind the camera.
    """
    # Convert point to homogeneous coordinates (4-vector).
    p_w_hom = np.array([p_w[0], p_w[1], p_w[2], 1.0])
    # Transform into camera coordinate system.
    p_cam = T_wc @ p_w_hom
    # Check if the point is in front of the camera.
    if p_cam[2] <= 0:
        return None
    # Normalize to obtain (x/z, y/z, 1).
    p_cam_norm = p_cam[:3] / p_cam[2]
    # Project to pixel coordinates.
    p_proj = K @ p_cam_norm

    return (p_proj[0], p_proj[1])

def localPointAssociation(map: Map, t_frame: Frame, K: np.ndarray, T_wc: np.ndarray):
    """
    Associates map points with current frame keypoints by searching only within a local window
    around the predicted pixel location.

    For each map point (from map.points_in_view), its 3D location (assumed to be stored in p.pt) is
    projected into the current frame using the predicted camera pose T_wc and the intrinsic matrix K.
    Then, the current frame keypoints (t_frame.keypoints) are scanned to find candidates that fall
    inside a (2*search_window x 2*search_window) region around the predicted pixel.
    Among those candidates, the best match is chosen based on descriptor similarity.

    Args:
        map: The local map object. It must have:
             - map.points_in_view: a list of map points.
             - Each map point should have:
                     .pt - its 3D world coordinate (e.g. a numpy array [x,y,z])
                     .observations - a list of observation dictionaries, where each observation has a
                                    "descriptor" key.
        t_frame: The current frame, which has:
                 - t_frame.keypoints: a list of keypoints (each with a .pt attribute, e.g. (u,v)).
                 - t_frame.descriptors: an array of descriptors for those keypoints.
        K: The camera intrinsic matrix.
        T_wc: The predicted current camera pose (4x4 homogeneous transformation), mapping world
              coordinates to camera coordinates.

    Returns:
        pairs: A list of tuples (map_idx, frame_idx) indicating the association of map points
               to current frame keypoints.
    """
    pairs = []

    # Loop over all map points that are expected to be in view.
    for i, map_point in enumerate(map.points_in_view):
        # Project the map point's 3D location into the current frame.
        pred_px = project_point(map_point.pos, T_wc, K)
        if pred_px is None:
            continue  # Skip points that project behind the camera.
        
        # We choose the last descriptor to represent the map point
        map_desc = map_point.observations[-1]["descriptor"]
        
        # Collect candidate current frame keypoints whose pixel coordinates fall within 
        # a window around the predicted pixel
        candidates = []
        for j, kp in enumerate(t_frame.keypoints):
            kp_pt = kp.pt
            if (abs(kp_pt[0] - pred_px[0]) <= SEARCH_WINDOW//2 and
                abs(kp_pt[1] - pred_px[1]) <= SEARCH_WINDOW//2):
                candidates.append(j)
        
        # If no keypoints are found in the window, skip to the next map point.
        if not candidates:
            continue
        
        # For each candidate, compute the descriptor distance using the Hamming norm.
        best_dist = np.inf
        best_idx = None
        for j in candidates:
            candidate_desc = t_frame.descriptors[j]
            # Compute Hamming distance.
            d = cv2.norm(np.array(map_desc), np.array(candidate_desc), cv2.NORM_HAMMING)
            if d < best_dist:
                best_dist = d
                best_idx = j
        
        # Accept the match only if the best distance is below the threshold.
        if best_idx is not None and best_dist < HAMMING_THRESHOLD:
            pairs.append((i, best_idx))
            
    # Save the matches
    if debug:
        match_save_path = results_dir / "matches/tracking/0-point_association" / f"map_{t_frame.id}.png"
        t_pxs = np.array([t_frame.keypoints[p[1]].pt for p in pairs], dtype=np.float64)
        plot_pixels(t_frame.img, t_pxs, save_path=match_save_path)
    
    if debug:
        print(f"\t Found {len(pairs)} Point Associations!")

    return pairs

def globalPointAssociation(map: Map, t_frame: Frame, K: np.ndarray):
    """
    Matches the map points seen in previous frames with the current frame.

    Args:
        map: The local map
        t_frame: The current t_frame, which has .keypoints and .descriptors
        T_wc: The predicted camera pose
        search_window: half-size of the bounding box (in pixels) around (u,v).

    Returns:
        pairs: (map_idx, frame_idx) indicating which map point matched which t_frame keypoint
    """
    # Extract the in view descriptors
    map_points = map.points_in_view
    map_descriptors = []
    map_point_idx = []
    map_pixels = []
    for i, p in enumerate(map_points):
        # Get the descriptors from every observation of a point
        for obs in p.observations:
            map_descriptors.append(obs["descriptor"])
            map_point_idx.append(i)
            map_pixels.append(obs["keypoint"].pt)
    map_descriptors = np.array(map_descriptors)
    map_point_idx = np.array(map_point_idx)
    map_pixels = np.array(map_pixels)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # Match descriptors
    matches = bf.knnMatch(map_descriptors, t_frame.descriptors, k=2)
    if len(matches) < MIN_NUM_MATCHES:
        return []

    # Filter matches
    # Apply Lowe's ratio test to filter out false matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.95 * n.distance:
            good_matches.append(m)
    if debug:
        print(f"\t Lowe's Test filtered {len(matches) - len(good_matches)}/{len(matches)} matches!")

    if len(good_matches) < MIN_NUM_MATCHES:
        return []
    
    # Next, ensure uniqueness by keeping only the best match per train descriptor.
    unique_matches = {}
    for m in good_matches:
        # If this train descriptor is not seen yet, or if the current match is better, update.
        if m.trainIdx not in unique_matches or m.distance < unique_matches[m.trainIdx].distance:
            unique_matches[m.trainIdx] = m

    # Convert the dictionary values to a list of unique matches
    unique_matches = list(unique_matches.values())
    if debug:
        print(f"\t Uniqueness filtered {len(good_matches) - len(unique_matches)}/{len(good_matches)} matches!")

    if len(unique_matches) < MIN_NUM_MATCHES:
        return []
            
    # Save the matches
    if debug:
        match_save_path = results_dir / "matches/tracking/0-point_association" / f"map_{t_frame.id}.png"
        t_pxs = np.array([t_frame.keypoints[m.trainIdx].pt for m in unique_matches], dtype=np.float64)
        plot_pixels(t_frame.img, t_pxs, save_path=match_save_path)
    
    # Finally, filter using the epipolar constraint
    # q_pixels = np.array([map_pixels[m.queryIdx] for m in unique_matches], dtype=np.float64)
    # t_pixels = np.array([t_frame.keypoints[m.trainIdx].pt for m in unique_matches], dtype=np.float64)
    # epipolar_constraint_mask, _, _ = enforce_epipolar_constraint(q_pixels, t_pixels, K)
    # if epipolar_constraint_mask is None:
    #     print("Failed to apply epipolar constraint..")
    #     return []
    # unique_matches = np.array(unique_matches)[epipolar_constraint_mask].tolist()
            
    # # Save the matches
    # if debug:
    #     match_save_path = results_dir / "matches/tracking/1-epipolar_constraint" / f"map_{t_frame.id}.png"
    #     plot_pixels(t_frame.img, t_pixels, save_path=match_save_path)
    
    # Prepare results
    pairs = []  # list of (map_idx, frame_idx, best_dist)
    for m in unique_matches:
        pairs.append((map_point_idx[m.queryIdx], m.trainIdx))

    if debug:
        print(f"\t Found {len(pairs)} Point Associations!")

    return pairs

# Function to estimate the relative pose using solvePnP
def estimate_relative_pose(
    map: Map,
    t_frame: Frame,
    map_t_pairs: list,
    K: np.ndarray,
    dist_coeffs=None
):
    """
    Estimate the relative camera displacement using a 3D-2D PnP approach.

    Args:
        map_points_w (np.ndarray): 
            (N, 3) array of 3D map points in world coordinates
            that correspond to the 'map_idx' indices in map_t_pairs.
        t_frame (Frame): 
            The current t_frame containing keypoints and descriptors.
        map_t_pairs (list of (int, int)): 
            - map_idx is the index into map_points_w
            - frame_idx is the index of t_frame.keypoints
        K (np.ndarray): 
            (3, 3) camera intrinsic matrix.
        dist_coeffs:
            Distortion coefficients for the camera. Default = None.

    Returns:
        displacement (np.ndarray): 4×4 transformation matrix T_{cam_new <- cam_old}.
                                    i.e., the relative transform from the old camera t_frame
                                    to the new camera t_frame.

        If the function fails, returns (None, None).
    """
    map_points_w = map.points_in_view
    num_points = len(map_t_pairs)
    if debug:
        print(f"Estimating Map -> Frame #{t_frame.id} pose using {num_points}/{len(map_points_w)} map points...")

    # 1) Build 3D <-> 2D correspondences
    map_points = []
    image_pxs = []
    for (map_idx, frame_idx) in map_t_pairs:
        map_points.append(map_points_w[map_idx])  # 3D in world coords
        kp = t_frame.keypoints[frame_idx]
        image_pxs.append(kp.pt)                   # 2D pixel (u, v)

    map_points = np.array(map_points, dtype=object)   
    map_point_positions = np.array([p.pos for p in map_points], dtype=np.float64) # (M, 3)
    image_pxs = np.array(image_pxs, dtype=np.float64)     # (M, 2)

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
    if not success or inliers is None:
        print("\t solvePnP failed!")
        return None, None
    if len(inliers) < MIN_INLIERS:
        print("\t solvePnP did not find enough inliers!")
        return None, None
    inliers = inliers.flatten()

    # Build an inliers mask
    inliers_mask = np.zeros(num_points, dtype=bool)
    inliers_mask[inliers] = True
    num_tracked_points = inliers_mask.sum()
    if debug:
        print(f"\t solvePnPRansac filtered {num_points - num_tracked_points}/{num_points} points.")
    
    # 3) Refine the pose using Levenberg-Marquardt on the inlier correspondences.
    rvec, tvec = cv2.solvePnPRefineLM(
        map_point_positions[inliers],
        image_pxs[inliers],
        K,
        dist_coeffs,
        rvec,
        tvec
    )

    # 4) Convert refined pose to a 4x4 transformation matrix.
    t_wc = tvec.flatten()
    R_wc, _ = cv2.Rodrigues(rvec)
    
    # Save the PnP tracked mask to the map
    tracking_mask = np.zeros(map.num_points, dtype=bool)
    for i, (map_idx, _) in enumerate(map_t_pairs): 
        tracking_mask[map_idx] = inliers_mask[i]
    map.set_tracking_mask(tracking_mask)

    # 5) Compute reprojection error
    ## Project the 3D points to 2D using the estimated pose
    projected_world_pxs, _ = cv2.projectPoints(map_point_positions, rvec, t_wc, K, dist_coeffs)
    projected_world_pxs = projected_world_pxs.squeeze()
    
    ## Calculate the per-point reprojection error (Euclidean distance)
    errors = np.sqrt(np.sum((image_pxs[inliers_mask] - projected_world_pxs[inliers_mask])**2, axis=1))
    
    ## Create a mask for points with error less than the threshold
    reproj_mask = errors < SETTINGS["PnP"]["reprojection_threshold"]
    if debug:
        print(f"\t Reprojection:",
              f"\n\t\t Median/Mean error ({np.median(errors):.2f}, {np.mean(errors):.2f})",
              f"\n\t\t Outliers {len(errors) - reproj_mask.sum()}/{len(errors)} points.")

    ## Visualization
    if debug:
        img_path = results_dir / f"matches/tracking/1-PnP_reprojection/map_{t_frame.id}a.png"
        plot_reprojection(t_frame.img, image_pxs[~inliers_mask], projected_world_pxs[~inliers_mask], path=img_path)
        img_path = results_dir / f"matches/tracking/1-PnP_reprojection/map_{t_frame.id}b.png"
        plot_reprojection(t_frame.img, image_pxs[inliers_mask], projected_world_pxs[inliers_mask], path=img_path)

    # 6) Construct T_{world->cam_new}
    T_wc = np.eye(4, dtype=np.float64)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = t_wc

    return T_wc, num_tracked_points

def is_keyframe(T: np.ndarray, num_tracked_points: int = 9999):
    """ Determine if motion expressed by t, R is significant by comparing to tresholds. """
    tx = T[0, 3] # The x component points right
    ty = T[1, 3] # The y component points down
    tz = T[2, 3] # The z component points forward

    trans = np.sqrt(tx**2 + ty**2 + tz**2)
    angle = abs(get_yaw(T[:3, :3]))

    is_keyframe = num_tracked_points > SETTINGS["keyframe"]["num_tracked_points"]
    # is_keyframe = is_keyframe and (trans > SETTINGS["keyframe"]["distance"] or angle > SETTINGS["keyframe"]["angle"])
    
    if debug:
        print(f"\t Tracked points: {num_tracked_points}, dist: {trans:.3f}, angle: {angle:.3f}")
        if is_keyframe:
            print("\t\t Keyframe!")
        else:
            print("\t\t Not a keyframe!")

    return is_keyframe

def triangulateNewPoints(q_frame: Frame, t_frame: Frame, map: Map, K: np.ndarray):
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
        print(f"Triangulating New Points using keyframes {q_frame.id} & {t_frame.id}...")
    
    # ------------------------------------------------------------------------
    # 1. Get keypoint matches
    # ------------------------------------------------------------------------

    # Extract the matches between the previous and current frame
    matches = q_frame.get_matches(t_frame.id)

    # Extract keypoint pixel coordinates and indices for both frames from the feature match
    q_kpt_pixels = np.float64([q_frame.keypoints[m.queryIdx].pt for m in matches])
    t_kpt_pixels = np.float64([t_frame.keypoints[m.trainIdx].pt for m in matches])

    # ------------------------------------------------------------------------
    # 2. Enforce Epipolar Constraint
    # ------------------------------------------------------------------------

    epipolar_constraint_mask, _, _ = enforce_epipolar_constraint(q_kpt_pixels, t_kpt_pixels, K)
    if epipolar_constraint_mask is None:
        print("[Tracking] Failed to apply epipolar constraint..")
        return None, None, None, None, None, None, False
    
    # Save the matches
    if debug:
        match_save_path = results_dir / f"matches/mapping/1-epipolar_constraint" / f"{q_frame.id}_{t_frame.id}a.png"
        plot_matches(matches[~epipolar_constraint_mask], q_frame, t_frame, save_path=match_save_path)
        match_save_path = results_dir / f"matches/mapping/1-epipolar_constraint" / f"{q_frame.id}_{t_frame.id}b.png"
        plot_matches(matches[epipolar_constraint_mask], q_frame, t_frame, save_path=match_save_path)
    matches = matches[epipolar_constraint_mask]

    # Extract the q->t transformation
    # Extract the Rotation and Translation arrays between the 2 frames
    T_qt = q_frame.match[t_frame.id]["T"] # [q->t]
    R_qt = T_qt[:3, :3]
    t_qt = T_qt[:3, 3].reshape(3,1)
   
    # ------------------------------------------------------------------------
    # 3. Filter based on the reprojection of the PnP pose
    # ------------------------------------------------------------------------ 

    reproj_mask = filter_by_reprojection(
        matches, q_frame, t_frame,
        R_qt, t_qt, K,
        save_path= results_dir / f"matches/mapping/2-reprojection"
    )
    
    # Save the matches
    if debug:
        match_save_path = results_dir / f"matches/mapping/3-reprojection" / f"{q_frame.id}_{t_frame.id}a.png"
        plot_matches(matches[~reproj_mask], q_frame, t_frame, save_path=match_save_path)
        match_save_path = results_dir / f"matches/mapping/3-reprojection" / f"{q_frame.id}_{t_frame.id}b.png"
        plot_matches(matches[reproj_mask], q_frame, t_frame, save_path=match_save_path)
    matches = matches[reproj_mask]
    
    # ------------------------------------------------------------------------
    # 4. Triangulate the matched points using the PnP estimated pose
    # ------------------------------------------------------------------------

    # Extract the pixel coordinates of the points
    q_pxs = np.array([q_frame.keypoints[m.queryIdx].pt for m in matches])
    t_pxs = np.array([t_frame.keypoints[m.trainIdx].pt for m in matches])

    # Triangulate
    q_points = triangulate(q_pxs, t_pxs, R_qt, t_qt, K)
    if q_points is None or len(q_points) == 0:
        print("Triangulation returned no 3D points.")
        return None, None, None, None, None, None, False

    # Transfer the points to the current coordinate frame [t->q]
    t_points = transform_points(q_points, T_qt) # (N, 3)

    # ------------------------------------------------------------------------
    # 5. Filter triangulated points for Z<0 and small triang. angle
    # ------------------------------------------------------------------------

    triang_mask = filter_triangulation_points(q_points, t_points, R_qt, t_qt)
    if triang_mask is None or triang_mask.sum() == 0:
        return None, None, None, None, None, None, False
    
    # Save the matches
    if debug:
        match_save_path = results_dir / f"matches/mapping/4-triangulation" / f"{q_frame.id}_{t_frame.id}a.png"
        plot_matches(matches[~triang_mask], q_frame, t_frame, save_path=match_save_path)
        match_save_path = results_dir / f"matches/mapping/4-triangulation" / f"{q_frame.id}_{t_frame.id}b.png"
        plot_matches(matches[triang_mask], q_frame, t_frame, save_path=match_save_path)

    matches = matches[triang_mask]
    q_points = q_points[triang_mask]
    t_points = t_points[triang_mask]

    # Extract the keypoints and descriptors of the valid triangulated 3d points
    q_kpts = np.array([q_frame.keypoints[m.queryIdx] for m in matches])
    t_kpts = np.array([t_frame.keypoints[m.trainIdx] for m in matches])

    # ------------------------------------------------------------------------
    # 6. Filter triangulated points based on scale
    # ------------------------------------------------------------------------

    q_scale_mask = filter_scale(q_points, q_kpts, q_frame.pose)
    t_scale_mask = filter_scale(t_points, t_kpts, t_frame.pose)
    scale_mask = q_scale_mask & t_scale_mask
    if scale_mask is None or scale_mask.sum() == 0:
        return None, None, None, None, None, None, False

    # Apply the scale mask to points, keypoints and descriptors
    matches = matches[scale_mask]
    q_points = q_points[q_scale_mask]
    t_points = t_points[t_scale_mask]
    q_kpts = q_kpts[t_scale_mask]
    t_kpts = t_kpts[t_scale_mask]

    # ------------------------------------------------------------------------
    # 7. Find 3D points that have/haven't been triangulated before
    # ------------------------------------------------------------------------

    # Extract the reference keypoint ids
    q_kpt_ids = np.array([q_frame.keypoints[m.queryIdx].class_id for m in matches])

    # Find which of the map keypoints don't intersect with the reference inlier keypoint IDs - these are new
    new_ids = np.setdiff1d(q_kpt_ids, map.point_ids)
    if len(new_ids) == 0:
        print("No new points to triangulate.")
        return None, None, None, None, None, None, False
    if debug:
        print(f"\t {len(new_ids)}/{len(q_kpt_ids)} points are new!")
    
    # Create a mask for the old/new triangulated points
    new_points_mask = np.isin(q_kpt_ids, new_ids)
    old_points_mask = ~new_points_mask

    # Apply the masks
    new_matches = matches[new_points_mask]
    old_matches = matches[old_points_mask]
    
    # Save the matches
    if debug:
        match_save_path = results_dir / f"matches/mapping/5-old_new" / f"{q_frame.id}_{t_frame.id}_old.png"
        plot_matches(old_matches, q_frame, t_frame, save_path=match_save_path)
        match_save_path = results_dir / f"matches/mapping/5-old_new" / f"{q_frame.id}_{t_frame.id}_new.png"
        plot_matches(new_matches, q_frame, t_frame, save_path=match_save_path)

    # ------------------------------------------------------------------------
    # 8. Save the old/new keypoints and descriptors
    # ------------------------------------------------------------------------

    q_old_points = q_points[old_points_mask]
    q_new_points = q_points[new_points_mask]
    q_frame.match[t_frame.id]["points"] = q_new_points

    t_old_points = t_points[old_points_mask]
    t_new_points = t_points[new_points_mask]
    t_frame.match[q_frame.id]["points"] = t_new_points

    t_old_kpts = t_kpts[old_points_mask]
    t_new_kpts = t_kpts[new_points_mask]

    t_old_descriptors = np.uint8([t_frame.descriptors[m.trainIdx] for m in old_matches])
    t_new_descriptors = np.uint8([t_frame.descriptors[m.trainIdx] for m in new_matches])

    # Return the newly triangulated points
    return t_old_points, t_old_kpts, t_old_descriptors, t_new_points, t_new_kpts, t_new_descriptors, True
