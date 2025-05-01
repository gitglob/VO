import cv2
import src.utils as utils
import src.globals as ctx


def search_for_triangulation(q_frame: utils.Frame, t_frame: utils.Frame):
    """Matches the visual words between 2 frames and returns the matches between their features."""
    # For each unmatched ORB in Ki we search a match with an un-matched point in other keyframe
    pairs = {} # t_feature_id: (q_feature_id, dist)

    # Compute the fundamental matrix for mapping points in q_frame to epipolar lines in t_frame
    F_qt = utils.compute_F12(q_frame, t_frame)

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
        for t_feat in t_features:
            # Skip features that are already in the map
            if t_feat.in_map: 
                break

            # Store feature matches
            desc_distances = {}

            # Iterate over features for the same word in the neighbor frame
            for q_feat in q_features:
                # Skip features that are already in the map
                if q_feat.in_map: 
                    continue
                # Extract their distance
                dist = cv2.norm(t_feat.desc, q_feat.desc, cv2.NORM_HAMMING)
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
            for q_feature_id, dist in desc_distances_sorted:
                q_feat = q_frame.features[q_feature_id]
                assert(q_feat.id == q_feature_id)
                # If the distance is much larger than the best candidate, stop searching
                if dist > dist_th:
                    break

                # Find the distance of the matched t_feat to the epipolar line in t_frame induced by q_feat
                d_epi_sqr = utils.dist_epipolar_line(q_feat.kpt.pt, t_feat.kpt.pt, F_qt)
                # Check if the candidate pair satisfies the epipolar constraint
                d_threshold = 3.84 * q_frame.scale_uncertainties[q_feat.kpt.octave]
                if d_epi_sqr < d_threshold:
                    pairs[t_feat.id] = (q_feat.id, dist)
                    # Stop if you find a good matching candidate
                    break

    # Drop duplicated features from the pairs
    unique_pairs = dedupe_pairs(pairs)

    # Prepare matches
    cv2_matches = []
    for t_feat_id, (q_feat_id, dist) in unique_pairs.items():
        q_feat = q_frame.features[q_feat_id]
        t_feat = t_frame.features[t_feat_id]
        match = cv2.DMatch(q_feat.idx, t_feat.idx, dist)
        cv2_matches.append(match)

    return cv2_matches

def dedupe_pairs(pairs):
    """
    Given:
      pairs: dict[t_feat_id] = (q_feat_id, dist)
    Returns:
      new_pairs: dict[t_feat_id] = (q_feat_id, dist)
                 with no repeated t_feat_id or q_feat_id, keeping minimal dist.
    """
    # 1. Sort all data by ascending dist
    sorted_pairs = sorted(pairs.items(), key=lambda item: item[1][1])
    
    new_pairs = {}
    used_t = set()
    used_n = set()
    
    # 2. Greedily take each pair if both feature IDs are still unused
    for t_feat_id, (q_feat_id, dist) in sorted_pairs:
        if t_feat_id in used_t or q_feat_id in used_n:
            continue
        new_pairs[t_feat_id] = (q_feat_id, dist)
        used_t.add(t_feat_id)
        used_n.add(q_feat_id)
    
    return new_pairs

# This is similar to search_for_triangulation, but uses Lowe's ratio test 
# and accepts matches immediately if they are good enough.
def search_by_bow(q_keyframe: utils.Frame, t_frame: utils.Frame):
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
