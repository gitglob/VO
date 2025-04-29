import numpy as np
import cv2

import src.utils as utils
import src.visualization as vis
import src.globals as ctx


from config import SETTINGS, results_dir, K, log


scale_factor = SETTINGS["orb"]["scale_factor"]
n_levels = SETTINGS["orb"]["level_pyramid"]

DEBUG = SETTINGS["generic"]["debug"]
W = SETTINGS["camera"]["width"]
H = SETTINGS["camera"]["height"]
HAMMING_THRESHOLD = SETTINGS["point_association"]["hamming_threshold"]


def map_search(t_frame: utils.Frame, save_path: str, use_epipolar_constraint=True):
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
    if DEBUG:
        log.info(f"Searching for map<->frame #{t_frame.id} correspondences!")
    
    # Extract the in view descriptors
    map_descriptors = []
    map_point_ids = []
    map_pixels = []
    for p in ctx.map.points.values():
        # Get the descriptors from every observation of a point
        for obs in p.observations:
            map_descriptors.append(obs.desc)
            map_point_ids.append(p.id)
            map_pixels.append(obs.kpt.pt)
    map_descriptors = np.array(map_descriptors)
    map_point_ids = np.array(map_point_ids)
    map_pixels = np.array(map_pixels)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # Match descriptors
    matches = bf.knnMatch(map_descriptors, t_frame.descriptors, k=2)
    if len(matches) < 10:
        return []

    # Filter matches
    # Apply Lowe's ratio test to filter out false matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    if DEBUG:
        log.info(f"\t Lowe's Test filtered {len(matches) - len(good_matches)}/{len(matches)} matches!")

    if len(good_matches) < 10:
        return []
    
    # Next, ensure uniqueness by keeping only the best match per train descriptor.
    unique_matches = {}
    for m in good_matches:
        # If this train descriptor is not seen yet, or if the current match is better, update.
        if m.trainIdx not in unique_matches or m.distance < unique_matches[m.trainIdx].distance:
            unique_matches[m.trainIdx] = m

    # Convert the dictionary values to a list of unique matches
    unique_matches = list(unique_matches.values())
    if DEBUG:
        log.info(f"\t Uniqueness filtered {len(good_matches) - len(unique_matches)}/{len(good_matches)} matches!")

    if len(unique_matches) < 10:
        return []
    
    # Finally, filter using the epipolar constraint
    if use_epipolar_constraint:
        q_pixels = np.array([map_pixels[m.queryIdx] for m in unique_matches], dtype=np.float64)
        t_pixels = np.array([t_frame.keypoints[m.trainIdx].pt for m in unique_matches], dtype=np.float64)
        epipolar_constraint_mask, _, _ = utils.enforce_epipolar_constraint(q_pixels, t_pixels)
        if epipolar_constraint_mask is None:
            log.warning("Failed to apply epipolar constraint..")
            return []
        unique_matches = np.array(unique_matches)[epipolar_constraint_mask].tolist()
    
    # Prepare results
    for m in unique_matches:
        pid = map_point_ids[m.queryIdx]
        point = ctx.map.points[pid]

        t_kpt = t_frame.keypoints[m.trainIdx]
        feat = t_frame.features[t_kpt.class_id]

        feat.match_map_point(point, m.distance)
        ctx.map.add_observation(t_frame, feat, point)

    # Save the matches
    if DEBUG:
        t_pxs = np.array([t_frame.keypoints[m.trainIdx].pt for m in unique_matches], dtype=np.float64)
        vis.plot_pixels(t_frame.img, t_pxs, save_path=save_path)

    if DEBUG:
        log.info(f"\t Found {len(unique_matches)} Point Associations!")
    return len(unique_matches)

def search_for_triangulation(q_frame: utils.Frame, t_frame: utils.Frame):
    """Matches the visual words between 2 frames and returns the matches between their features."""
    # For each unmatched ORB in Ki we search a match with an un-matched point in other keyframe
    pairs = {} # t_feature_id: (neighbor_frame_id, neighbor_feature_id, dist)

    # Compute the fundamental matrix
    F_tn = utils.compute_F12(t_frame, q_frame)

    # Iterate over all visual words
    for word_id in ctx.bow_db.keys():
        # Extract the features from the frames, if the word exists
        t_features = t_frame.get_features_for_word(word_id)
        if t_features is None:
            continue
        q_features = q_frame.get_features_for_word(word_id)
        if q_features is None:
            continue
        # log.info(f"\t Word #{word_id}: {len(t_features)} x {len(q_features)} candidates!") 

        # Iterate over features that match this word in the current frame
        for t_feature in t_features:
            # Skip features that are already in the map
            if t_feature.in_map: 
                break

            # Store feature matches
            desc_distances = {}

            # Iterate over features for the same word in the neighbor frame
            for q_feat in q_features:
                # Extract their distance
                dist = cv2.norm(t_feature.desc, q_feat.desc, cv2.NORM_HAMMING)
                # Check if the descriptors are similar enough
                if dist > 50:
                    continue
                # Keep the match distance and the feature id
                desc_distances[q_feat.id] = dist

            # Check if any matches were found
            if len(desc_distances.keys()) == 0:
                continue

            # Sort the match distances based on distance
            desc_distances_sorted = sorted(desc_distances.items(), key=lambda item: item[1])
            best_dist = desc_distances_sorted[0][1]
            dist_th = round(2*best_dist)

            # Iterate over all feature candidates for this word
            for n_feature_id, dist in desc_distances_sorted:
                q_feat = q_frame.features[n_feature_id]
                assert(q_feat.id == n_feature_id)
                # If the distance if much larger than the best candidate, stop searching
                if dist > dist_th:
                    break

                # Check if the candidate pair satisfies the epipolar constraint
                d_epi_sqr = utils.dist_epipolar_line(t_feature.kpt.pt, q_feat.kpt.pt, F_tn)
                d_threshold = 3.84 * q_frame.scale_uncertainties[q_feat.kpt.octave]
                if d_epi_sqr < d_threshold:
                    pairs[t_feature.id] = (q_feat.id, dist)
                    # log.info(f"\t\t Match at word #{word_id}: t_{t_feature.id} <-> n_{q_feat.id}!") 
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

