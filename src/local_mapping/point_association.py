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
            for q_feature_id, dist in desc_distances_sorted:
                q_feat = q_frame.features[q_feature_id]
                assert(q_feat.id == q_feature_id)
                # If the distance is much larger than the best candidate, stop searching
                if dist > dist_th:
                    break

                # Find the distance of the matched t_feat to the epipolar line in t_frame induced by q_feat
                d_epi_sqr = utils.dist_epipolar_line(q_feat.kpt.pt, t_feature.kpt.pt, F_qt)
                # Check if the candidate pair satisfies the epipolar constraint
                d_threshold = 3.84 * q_frame.scale_uncertainties[q_feat.kpt.octave]
                if d_epi_sqr < d_threshold:
                    pairs[t_feature.id] = (q_feat.id, dist)
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

