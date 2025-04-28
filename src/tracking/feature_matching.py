import cv2
import numpy as np
import src.local_mapping as mapping
import src.utils as utils
import src.globals as ctx
import src.visualization as vis
from config import SETTINGS, results_dir, K, log


debug = SETTINGS["generic"]["debug"]
DIST_THRESH = SETTINGS["point_association"]["hamming_threshold"]


def global_search(t_frame: utils.Frame):
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
    map_points = ctx.cgraph.get_frustum_points(t_frame)
    map_descriptors = []
    map_point_ids = []
    map_pixels = []
    for pid, p in map_points:
        # Get the descriptors from every observation of a point
        for obs in p.observations:
            map_descriptors.append(obs.desc)
            map_point_ids.append(pid)
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
        if m.distance < 0.95 * n.distance:
            good_matches.append(m)
    if debug:
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
    if debug:
        log.info(f"\t Uniqueness filtered {len(good_matches) - len(unique_matches)}/{len(good_matches)} matches!")

    if len(unique_matches) < 10:
        return []
            
    # Save the matches
    if debug:
        match_save_path = results_dir / "matches/tracking/point_assocation/global" / f"map_{t_frame.id}.png"
        t_pxs = np.array([t_frame.keypoints[m.trainIdx].pt for m in unique_matches], dtype=np.float64)
        vis.plot_pixels(t_frame.img, t_pxs, save_path=match_save_path)
    
    # Finally, filter using the epipolar constraint
    # q_pixels = np.array([map_pixels[m.queryIdx] for m in unique_matches], dtype=np.float64)
    # t_pixels = np.array([t_frame.keypoints[m.trainIdx].pt for m in unique_matches], dtype=np.float64)
    # epipolar_constraint_mask, _, _ = enforce_epipolar_constraint(q_pixels, t_pixels)
    # if epipolar_constraint_mask is None:
    #     log.warning("Failed to apply epipolar constraint..")
    #     return []
    # unique_matches = np.array(unique_matches)[epipolar_constraint_mask].tolist()
            
    # # Save the matches
    # if debug:
    #     match_save_path = results_dir / "matches/tracking/point_assocation/1-epipolar_constraint" / f"map_{t_frame.id}.png"
    #     vis.plot_pixels(t_frame.img, t_pixels, save_path=match_save_path)
    
    # Prepare results
    for m in unique_matches:
        feat = t_frame.features[m.trainIdx]
        pid = map_points[m.queryIdx]
        point = ctx.map.points[pid]
        feat.match_map_point(point, m.distance)
        ctx.map.add_observation(t_frame, feat, point)

    if debug:
        log.info(f"\t Found {len(unique_matches)} Point Associations!")
    return len(unique_matches)

def window_search(q_keyframe: utils.Frame, t_frame: utils.Frame, radius: int, 
                  min_scale_level: int = -1, max_scale_level: int = -1 , save_path: str = None) -> int:
    """
    Match MapPoints seen in q_keyframe to keypoints in t_frame within a spatial window
    around each q_keyframe keypoint, applying descriptor ratio test and (optional)
    orientation consistency.

    Wokrs on a purely Feature Level! Does not project points!
    Uses Lowe's ratio test!

    :param q_keyframe:         reference frame containing MapPoints to match
    :param t_frame:         target frame whose keypoints we’ll search
    :param radius:          half‐size of the search square (pixels)
    :param min_scale_level: minimum octave level to consider (inclusive)
    :param max_scale_level: maximum octave level to consider (inclusive)
    :return:                number of successful matches, and internally
                            sets `t_frame.mvpMapPoints` entries for each match
    """
    matched_feat_ids = []
    matched_features = []

    # Flags to decide whether to check pyramid levels
    check_min = min_scale_level != -1
    check_max = max_scale_level != -1

    # Iterate over all map points of the previous frame
    q_map_points, q_mp_features = q_keyframe.get_map_points_and_features()
    point: mapping.mapPoint
    for point, q_feat in zip(q_map_points, q_mp_features):
        q_kpt = q_feat.kpt
        q_desc = q_feat.desc
        q_octave = q_kpt.octave

        # Enforce scale‐level constraints if requested
        if check_min and q_octave < min_scale_level:
            continue
        if check_max and q_octave > max_scale_level:
            continue

        # Get candidate keypoint indices in t_frame within the window at same level
        candidates = t_frame.get_features_in_area(
            u=q_kpt.pt[0],
            v=q_kpt.pt[1],
            r=radius,
            min_level=q_octave,
            max_level=q_octave
        )
        if not candidates:
            continue

        # Find the two best matches by Hamming (or L2) distance
        best_dist, best_dist2 = 999999, 999999
        best_feature_id = None

        # Compare the q map point descriptor with each candidate feature's descriptor.
        for t_feat in candidates:
            # Check if the candidate feature is already matched to a map point
            if t_feat in matched_feat_ids:
                continue

            # Compute Hamming distance.
            dist = cv2.norm(np.array(q_feat.desc), np.array(t_feat.desc), cv2.NORM_HAMMING)

            if dist < best_dist:
                best_dist2 = best_dist
                best_dist  = dist
                best_feature_id  = t_feat.id
            elif dist < best_dist2:
                best_dist2 = dist

        # Ratio test to accept best match
        if (best_dist <= best_dist2 * 0.6) and (best_dist <= 100):
            # Accept match
            matched_feat_ids.append(best_feature_id)
            matched_features.append((point, q_feat, t_feat, best_dist))
                
    # Update the frame<->map matches
    matches = []
    for (point, q_feat, t_feat, dist) in matched_features:
        t_feat.match_map_point(point, dist)
        ctx.map.add_observation(t_frame, t_feat, point)
        matches.append((q_feat.idx, t_feat.idx, dist))

    # Save the matched points
    if debug and len(matches) > 0:
        cv2_matches = [cv2.DMatch(t, n, d) for (t,n,d) in matches]
        vis.plot_matches(cv2_matches, q_keyframe, t_frame, save_path=save_path)
    
    if debug:
        log.info(f"\t Found {len(matched_features)} Point Associations!")

    return len(matched_features)

def search_by_projection(q_frame: utils.Frame, t_frame: utils.Frame, theta: int = None, radius: int = None, save_path: str = None):
    """
    Searches for map point correspondences by projecting map points from a previous frame 
    into the current frame and matching them by descriptor similarity.

    Projects map points to the from one frame to another using the pose information!
    Accepts all matches under a distance, which could lead to more false matches!
    """
    match_distances = {}
    matched_features = []

    # Iterate over all map points of the previous frame
    q_map_points, q_mp_features = q_frame.get_map_points_and_features()
    point: mapping.mapPoint
    for point, q_feat in zip(q_map_points, q_mp_features):
        # Project point into the current frame
        ret = point.project2frame(t_frame)
        if ret is None:
            continue
        u, v = ret

        # Define the search radius based on the keypoint scale
        octave = q_feat.kpt.octave
        if theta:
            radius = theta * t_frame.scale_factors[octave]

        # Retrieve indices of features within the area around the projected point.
        candidates = t_frame.get_features_in_area(u, v, radius, octave-1, octave+1)
        if len(candidates) == 0:
            continue

        # Get the best descriptor
        q_desc = q_feat.desc

        # Compare the map point descriptor with each candidate feature's descriptor.
        best_dist = np.inf
        best_feature_id = None
        for t_feat in candidates:
            # Check if the candidate feature is already matched to a map point
            if t_feat.matched:
                continue
            candidate_desc = t_feat.desc
            # Compute Hamming distance.
            d = cv2.norm(np.array(q_desc), np.array(candidate_desc), cv2.NORM_HAMMING)
            if d < best_dist:
                best_dist = d
                best_feature_id = t_feat.id

        # If the best descriptor is a good enough match, save it
        if best_feature_id is not None and best_dist < 100:
            # Make sure that we only keep 1 match per frame pixel
            if best_feature_id not in match_distances.keys() or best_dist < match_distances[best_feature_id]:
                match_distances[best_feature_id] = best_dist
                matched_features.append((point, t_feat, q_feat, best_dist))
                
    # Update the frame<->map matches
    matches = []
    for point, q_feat, t_feat, dist in matched_features:
        t_feat.match_map_point(point, dist)
        ctx.map.add_observation(t_frame, t_feat, point)
        matches.append((q_feat.idx, t_feat.idx, dist))    

    # Save the matched points
    if debug and len(matches) > 0:
        cv2_matches = [cv2.DMatch(t, n, d) for (t,n,d) in matches]
        vis.plot_matches(cv2_matches, q_frame, t_frame, save_path=save_path)
    
    if debug:
        log.info(f"\t Found {len(matched_features)} Point Associations!")

    return len(matched_features)

def search_by_bow(q_keyframe: utils.Frame, t_frame: utils.Frame, save_path: str):
    """
    Matches the visual words that exist in the map in a previous keyframe with the visual words in the current frame.
    
    Args:
        q_keyframe: Keyframe that already exists in the map
        t_frame: Current frame
    """
    # For each unmatched ORB in Ki we search a match with an un-matched point in other keyframe
    match_distances = {} # t_feature_id: (map_point_id, distance)
    matched_features = []

    # Iterate over all visual words of the previous keyframe
    for word_id in q_keyframe.feature_vector.keys():
        # Extract the features from the frames, if the word exists
        q_features = q_keyframe.get_features_for_word(word_id)
        if q_features is None:
            continue
        t_features = t_frame.get_features_for_word(word_id)
        if t_features is None:
            continue
        # log.info(f"\t Word #{word_id}: {len(t_features)} x {len(q_features)} candidates!") 

        # Iterate over features that match this word in the previous keyframe
        for q_feat in q_features:
            # Skip features that don't match a map point
            if not q_feat.in_map: 
                break

            # Find best and second best descriptor matched_features
            best_dist = 99999
            best_dist2 = 99999
            best_pid = -1

            # Iterate over features for the same word in the neighbor frame
            for t_feat in t_features:
                # Make sure every feature is only matched once
                if t_feat.id in match_distances.keys():
                    continue

                # Extract their distance
                dist = cv2.norm(q_feat.desc, t_feat.desc, cv2.NORM_HAMMING)

                # Check if it is the best or second best distance
                if dist < best_dist:
                    best_dist2 = best_dist
                    best_dist = dist
                    best_pid = q_feat.mp.id
                elif dist < best_dist2:
                    best_dist2 = dist

            # Check if the best distance is good enough
            if (best_dist < 50) and (best_dist < 0.75 * best_dist2):
                match_distances[t_feat.id] = best_dist
                point = ctx.map.point[best_pid]
                matched_features.append(point, q_feat, t_feat, best_dist)

    # Update the frame<->map matched_features
    matches = []
    for point, q_feat, t_feat, dist in matched_features:
        t_feat.match_map_point(point, dist)
        ctx.map.add_observation(t_frame, t_feat, point)
        matches.append((q_feat.idx, t_feat.idx, dist))    

    # Save the matched points
    if debug and len(matches) > 0:
        cv2_matches = [cv2.DMatch(t, n, d) for (t,n,d) in matched_features]
        vis.plot_matches(cv2_matches, q_keyframe, t_frame, save_path=save_path)
    
    if debug:
        log.info(f"\t Found {len(matches)} Point Associations!")

    return len(matches)

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
                if dist > DIST_THRESH:
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

    return len(unique_pairs)

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
