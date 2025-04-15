import bisect
import numpy as np
import cv2
from scipy.linalg import expm, logm
from src.others.local_map import Map, mapPoint
from src.backend.convisibility_graph import ConvisibilityGraph
from src.others.frame import Frame
from src.others.utils import invert_transform
from src.others.visualize import plot_pixels
from config import SETTINGS, results_dir, K


debug = SETTINGS["generic"]["debug"]
MIN_INLIERS = SETTINGS["PnP"]["min_inliers"]
W = SETTINGS["camera"]["width"]
H = SETTINGS["camera"]["height"]
MIN_NUM_MATCHES = SETTINGS["point_association"]["num_matches"]
HAMMING_THRESHOLD = SETTINGS["point_association"]["hamming_threshold"]


def constant_velocity_model(t: float, times: list, frames: list):
    """Predicts the next pose assuming constant velocity between the last 3 frames"""
    poses = [f.pose for f in list(frames.values())]

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

def project_point(p_w: np.ndarray, T_wc: np.ndarray):
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

def bowPointAssociation(map: Map, cand_frame: Frame, t_frame: Frame, cgraph: ConvisibilityGraph):
    """
    Matches the map points seen in a candidate frame with the current frame.

    Args:
        map: The local map
        cand_frame: A candidate frame, that already contributed points to the map
        t_frame: The current frame, which has .keypoints and .descriptors
        T_wc: The predicted camera pose
        search_window: half-size of the bounding box (in pixels) around (u,v).

    Returns:
        pairs: (map_idx, frame_idx) indicating which map point matched which t_frame keypoint
    """
    # Extract candidate map points
    map_points = cgraph.get_frustum_points(cand_frame, map)
    map_descriptors = []
    map_point_ids = []
    map_pixels = []
    for p in map_points:
        # Get the descriptors from every observation of a point
        for obs in p.observations:
            map_descriptors.append(obs["descriptor"])
            map_point_ids.append(p.id)
            map_pixels.append(obs["keypoint"].pt)
    map_descriptors = np.array(map_descriptors)
    map_point_ids = np.array(map_point_ids)
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
        match_save_path = results_dir / "matches/tracking/point_assocation/relocalization" / f"map_{t_frame.id}.png"
        t_pxs = np.array([t_frame.keypoints[m.trainIdx].pt for m in unique_matches], dtype=np.float64)
        plot_pixels(t_frame.img, t_pxs, save_path=match_save_path)
    
    # Finally, filter using the epipolar constraint
    # q_pixels = np.array([map_pixels[m.queryIdx] for m in unique_matches], dtype=np.float64)
    # t_pixels = np.array([t_frame.keypoints[m.trainIdx].pt for m in unique_matches], dtype=np.float64)
    # epipolar_constraint_mask, _, _ = enforce_epipolar_constraint(q_pixels, t_pixels)
    # if epipolar_constraint_mask is None:
    #     print("Failed to apply epipolar constraint..")
    #     return []
    # unique_matches = np.array(unique_matches)[epipolar_constraint_mask].tolist()
            
    # # Save the matches
    # if debug:
    #     match_save_path = results_dir / "matches/tracking/point_assocation/1-epipolar_constraint" / f"map_{t_frame.id}.png"
    #     plot_pixels(t_frame.img, t_pixels, save_path=match_save_path)
    
    # Prepare results
    pairs = []  # list of (map_idx, frame_idx, best_dist)
    for m in unique_matches:
        feature = t_frame.features[m.trainIdx]
        pairs.append((map_point_ids[m.queryIdx], feature.id))
        t_frame.features[m.trainIdx].tracked = True

    if debug:
        print(f"\t Found {len(pairs)} Point Associations!")

    return pairs

def localPointAssociation(cgraph: ConvisibilityGraph, map: Map, 
                          t_frame: Frame, 
                          T_wc: np.ndarray, theta: int = None, search_window: int=None):
    """
    Associates map points with current frame keypoints by searching only within a local window
    around the predicted pixel location.

    When theta is used, we perform Adaptive Search Window Based on Scale:
    Multi-Scale Detection:
    Feature detectors like ORB operate over a scale space by processing 
    the image at multiple resolutions (or octaves). An octave corresponds 
    to a level in the pyramid of images at progressively lower resolutions. 
    Keypoints detected in higher octaves correspond to larger structures in the original image.

    Scale-Invariant Matching:
    Because features are detected across different scales, the intrinsic 
    size of the feature (or the area of support used to compute the descriptor) 
    varies with the octave. A feature detected in a higher octave has a larger 
    receptive field, meaning its description captures information from a broader 
    image region. Matching these features accurately requires accounting for the variability in scale.

    Relative Feature Size:
    When projecting a 3D point from one frame into another, 
    the associated keypoint may come from an octave where the 
    feature represents a large structure. A fixed search window 
    might be too small, potentially missing the correct match due 
    to scale differences and minor inaccuracies in projection. 
    Conversely, for small features detected at lower octaves, 
    a large search window may introduce unnecessary candidates 
    and increase the likelihood of false matches.

    Dynamic Adjustment:
    By multiplying a base threshold with the scale factor 
    corresponding to the keypoint’s octave, the search window 
    adapts to the expected feature size. For example, a keypoint 
    from a higher octave (with a larger scale factor) will have a 
    larger search radius than one from a lower octave. This dynamic 
    scaling improves the reliability of the matching process.

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
    frustum_points: set[mapPoint] = cgraph.get_frustum_points(t_frame, map)
    for map_point in frustum_points:
        # Project the map point's 3D location into the current frame.
        pred_px = project_point(map_point.pos, T_wc)
        if pred_px is None:
            continue  # Skip points that project behind the camera.
        else:
            u, v = pred_px
        
        # We choose the last descriptor to represent the map point
        map_desc = map_point.observations[-1]["descriptor"]
        
        # Collect candidate current frame keypoints whose pixel coordinates fall within 
        # a window around the predicted pixel
        radius = search_window // 2 if search_window else theta * t_frame.scale_factors[kp.octave]
        candidates = []
        for kp in enumerate(t_frame.keypoints):
            kp_pt = kp.pt
            if (abs(kp_pt[0] - u) <= radius and
                abs(kp_pt[1] - v) <= radius):
                candidates.append(kp.id)
        
        # If no keypoints are found in the window, skip to the next map point.
        if not candidates:
            continue
        
        # For each candidate, compute the descriptor distance using the Hamming norm.
        best_dist = np.inf
        best_feature_id = None
        for kpt_id in candidates:
            candidate_desc = t_frame.features[kpt_id].desc
            # Compute Hamming distance.
            d = cv2.norm(np.array(map_desc), np.array(candidate_desc), cv2.NORM_HAMMING)
            if d < best_dist:
                best_dist = d
                best_feature_id = kpt_id
        
        # Accept the match only if the best distance is below the threshold.
        if best_feature_id is not None and best_dist < HAMMING_THRESHOLD:
            pairs.append((map_point.id, best_feature_id))
            t_frame.features[best_feature_id].tracked = True
            
    # Save the matches
    if debug:
        match_save_path = results_dir / "matches/tracking/point_assocation/local" / f"map_{t_frame.id}.png"
        t_pxs = np.array([t_frame.keypoints[p[1]].pt for p in pairs], dtype=np.float64)
        plot_pixels(t_frame.img, t_pxs, save_path=match_save_path)
    
    if debug:
        print(f"\t Found {len(pairs)} Point Associations!")

    return pairs

def globalPointAssociation(map: Map, t_frame: Frame):
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
    map_points = map.get_frustum_points(t_frame, map)
    map_descriptors = []
    map_point_ids = []
    map_pixels = []
    for p in map_points:
        # Get the descriptors from every observation of a point
        for obs in p.observations:
            map_descriptors.append(obs["descriptor"])
            map_point_ids.append(p.id)
            map_pixels.append(obs["keypoint"].pt)
    map_descriptors = np.array(map_descriptors)
    map_point_ids = np.array(map_point_ids)
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
        match_save_path = results_dir / "matches/tracking/point_assocation/global" / f"map_{t_frame.id}.png"
        t_pxs = np.array([t_frame.keypoints[m.trainIdx].pt for m in unique_matches], dtype=np.float64)
        plot_pixels(t_frame.img, t_pxs, save_path=match_save_path)
    
    # Finally, filter using the epipolar constraint
    # q_pixels = np.array([map_pixels[m.queryIdx] for m in unique_matches], dtype=np.float64)
    # t_pixels = np.array([t_frame.keypoints[m.trainIdx].pt for m in unique_matches], dtype=np.float64)
    # epipolar_constraint_mask, _, _ = enforce_epipolar_constraint(q_pixels, t_pixels)
    # if epipolar_constraint_mask is None:
    #     print("Failed to apply epipolar constraint..")
    #     return []
    # unique_matches = np.array(unique_matches)[epipolar_constraint_mask].tolist()
            
    # # Save the matches
    # if debug:
    #     match_save_path = results_dir / "matches/tracking/point_assocation/1-epipolar_constraint" / f"map_{t_frame.id}.png"
    #     plot_pixels(t_frame.img, t_pixels, save_path=match_save_path)
    
    # Prepare results
    pairs = []  # list of (map_idx, frame_idx, best_dist)
    for m in unique_matches:
        pairs.append((map_point_ids[m.queryIdx], m.trainIdx))
        t_frame.features[m.trainIdx].tracked = True

    if debug:
        print(f"\t Found {len(pairs)} Point Associations!")

    return pairs

def mapPointAssociation(pairs: list[tuple], map: Map, t_frame: Frame, theta: int = 15):
    """
    Projects all un-matched map points to a frame and searches more correspondances.

    Returns:
        pairs: A list of tuples (map_idx, frame_idx) indicating the association of map points
               to current frame keypoints.      
    """
    # Extract the already matched features
    matched_kpt_ids: set = {p[1] for p in pairs}

    # Iterate over all the map points
    point: mapPoint
    for point in map.points:
        # 1) Compute the map point projection x in the current 
        # frame. Discard if it lays out of the image bounds.
        x = point.project2frame(t_frame)
        if x is None:
            continue
        else:
            u, v = x
        
        # 2) Compute the angle between the current viewing ray v
        # and the map point mean viewing direction n. Discard if v · n < cos(60◦).
        v = point.view_ray(t_frame)
        n = point.mean_view_ray()
        if v * n < np.cos(np.deg2rad(60)):
            continue

        # 3) Compute the distance d from map point to cameracenter. 
        # Discard if it is out of the scale invariance region
        # of the map point [dmin , dmax ].
        d = np.norm(point.pos - t_frame.pose[:3, 3])
        d_min, d_max = point.getScaleInvarianceLimits()
        if d < d_min or d > d_max:
            continue

        # 4) Compute the scale in the frame by the ration d/d_min
        scale = d / d_min

        # 5) Compare the representative descriptor D of the map point with the 
        # still unmatched ORB features in the frame, at the predicted scale, 
        # and near x, and associate the map point with the best match.
        D = point.best_descriptor

        # Collect candidate current frame un-matched keypoints whose pixel coordinates 
        # fall within a window around the predicted pixel
        octave_idx = bisect.bisect_left(t_frame.scale_factors, scale)
        radius = theta * t_frame.scale_factors[octave_idx]
        candidates = []
        for kp in t_frame.keypoints:
            if kp.class_id in matched_kpt_ids:
                continue
            kp_pt = kp.pt
            if (abs(kp_pt[0] - u) <= radius and
                abs(kp_pt[1] - v) <= radius):
                candidates.append(kp.id)
        
        # If no keypoints are found in the window, skip to the next map point.
        if len(candidates) == 0:
            continue
        
        # For each candidate, compute the descriptor distance using the Hamming norm.
        best_dist = np.inf
        best_feature_idx = None
        for kpt_id in candidates:
            candidate_desc = t_frame.features[kpt_id].desc
            # Compute Hamming distance.
            d = cv2.norm(np.array(D), np.array(candidate_desc), cv2.NORM_HAMMING)
            if d < best_dist:
                best_dist = d
                best_feature_idx = kpt_id
        
        pairs.append((point.id, best_feature_idx))
        t_frame.features[best_feature_idx].tracked = True


    return pairs
