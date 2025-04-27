import cv2
import numpy as np
import src.utils as utils
import src.globals as ctx
from config import SETTINGS, results_dir, K, log


DIST_THRESH = SETTINGS["point_association"]["hamming_threshold"]

def search_by_projection(q_frame: utils.Frame, t_frame: utils.Frame, theta: int=None, search_window: int=None):
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
        T_w2f: The predicted current camera pose (4x4 homogeneous transformation), mapping world
              coordinates to camera coordinates.

    Returns:
        pairs: A list of tuples (map_idx, frame_idx) indicating the association of map points
               to current frame keypoints.
    """
    matched_features = {}

    # Get the map points observed in the last frame
    q_map_points = cgraph.get_frustum_points(q_frame, map)

    # Loop over the points
    for point in q_map_points:
        # Check if the point is in the current camera's frustum
        result = t_frame.is_in_frustum(point, map.keyframes)
        if result is False:
            continue
        u, v, _ = result
        
        # Collect candidate current frame keypoints whose pixel coordinates fall within 
        # a window around the predicted pixel
        candidates = set()
        for kpt_id, feat in t_frame.features.items():
            kpt = feat.kpt
            radius = search_window // 2 if search_window else theta * t_frame.scale_factors[kpt.octave]
            kp_pt = kpt.pt
            if (abs(kp_pt[0] - u) <= radius and
                abs(kp_pt[1] - v) <= radius):
                candidates.add(kpt_id)
        
        # If no keypoints are found in the window, skip to the next map point.
        if len(candidates) == 0:
            continue
        
        # We choose the last descriptor to represent the map point
        map_desc = point.observations[-1].desc
        
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
            # Make sure that we only keep 1 match per frame pixel
            if best_feature_id not in matched_features.keys() or best_dist < matched_features[best_feature_id][1]:
                matched_features[best_feature_id] = (point.id, best_dist)
                
    # Update the frame<->map matches
    for feat_id, (pid, dist) in matched_features.items():
        feat = t_frame.features[feat_id]
        feat.match_map_point(pid, dist)
        map.points[pid].observe(map._kf_counter, t_frame.id, feat.kpt, feat.desc)

    # Save the matched points
    if debug and len(matched_features.keys()) > 0:
        match_save_path = results_dir / "matches/tracking/point_assocation/local" / f"map_{t_frame.id}.png"
        t_pxs = np.array([t_frame.features[feat_id].kpt.pt for feat_id in matched_features.keys()], dtype=np.float64)
        plot_pixels(t_frame.img, t_pxs, save_path=match_save_path)
    
    if debug:
        log.info(f"\t Found {len(matched_features)} Point Associations!")
    return len(matched_features)

def search_by_bow(t_frame: utils.Frame, n_frame: utils.Frame):
    """Matches the visual words between 2 frames and returns the matches between their features."""
    # For each unmatched ORB in Ki we search a match with an un-matched point in other keyframe
    pairs = {} # t_feature_id: (neighbor_frame_id, neighbor_feature_id, dist)

    # Compute the fundamental matrix
    F_tn = utils.compute_F12(t_frame, n_frame)

    # Iterate over all visual words
    for word_id in ctx.bow_db.keys():
        # Extract the features from the frames, if the word exists
        t_features = t_frame.get_features_for_word(word_id)
        if t_features is None:
            continue
        n_features = n_frame.get_features_for_word(word_id)
        if n_features is None:
            continue
        # log.info(f"\t Word #{word_id}: {len(t_features)} x {len(n_features)} candidates!") 

        # Iterate over features that match this word in the current frame
        for t_feature in t_features:
            # Skip already matched features
            if t_feature.matched: 
                break

            # Store feature matches
            desc_distances = {}

            # Iterate over features for the same word in the neighbor frame
            for n_feature in n_features:
                # Extract their distance
                dist = cv2.norm(t_feature.desc, n_feature.desc, cv2.NORM_HAMMING)
                # Check if the descriptors are similar enough
                if dist > DIST_THRESH:
                    continue
                # Keep the match distance and the feature id
                desc_distances[n_feature.id] = dist

            # Check if any matches were found
            if len(desc_distances.keys()) == 0:
                continue

            # Sort the match distances based on distance
            desc_distances_sorted = sorted(desc_distances.items(), key=lambda item: item[1])
            best_dist = desc_distances_sorted[0][1]
            dist_th = round(2*best_dist)

            # Iterate over all feature candidates for this word
            for n_feature_id, dist in desc_distances_sorted:
                n_feature = n_frame.features[n_feature_id]
                assert(n_feature.id == n_feature_id)
                # If the distance if much larger than the best candidate, stop searching
                if dist > dist_th:
                    break

                # Check if the candidate pair satisfies the epipolar constraint
                d_epi_sqr = utils.dist_epipolar_line(t_feature.kpt.pt, n_feature.kpt.pt, F_tn)
                d_threshold = 3.84 * n_frame.scale_uncertainties[n_feature.kpt.octave]
                if d_epi_sqr < d_threshold:
                    pairs[t_feature.id] = (n_feature.id, dist)
                    # log.info(f"\t\t Match at word #{word_id}: t_{t_feature.id} <-> n_{n_feature.id}!") 
                    # Stop if you find a good matching candidate
                    break

    # Drop duplicated features from the pairs
    unique_pairs = dedupe_pairs(pairs)
    return unique_pairs

def dedupe_pairs(pairs):
    """
    Given:
      pairs: dict[t_feat_id] = (n_feat_id, dist)
    Returns:
      new_pairs: dict[t_feat_id] = (n_feat_id, dist)
                 with no repeated t_feat_id or n_feat_id, keeping minimal dist.
    """
    # 1. Sort all data by ascending dist
    sorted_pairs = sorted(pairs.items(), key=lambda item: item[1][1])
    
    new_pairs = {}
    used_t = set()
    used_n = set()
    
    # 2. Greedily take each pair if both feature IDs are still unused
    for t_feat_id, (n_feat_id, dist) in sorted_pairs:
        if t_feat_id in used_t or n_feat_id in used_n:
            continue
        new_pairs[t_feat_id] = (n_feat_id, dist)
        used_t.add(t_feat_id)
        used_n.add(n_feat_id)
    
    return new_pairs
