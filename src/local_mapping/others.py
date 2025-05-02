import cv2
import numpy as np
import src.local_mapping as mapping
import src.utils as utils
import src.globals as ctx
import src.visualization as vis
from config import SETTINGS, log


debug = SETTINGS["generic"]["debug"]
DIST_THRESH = SETTINGS["point_association"]["hamming_threshold"]
use_epipolar_constraint = False


def window_search(q_keyframe: utils.Frame, t_frame: utils.Frame, radius: int, 
                  min_scale_level: int = -1, max_scale_level: int = -1 , save_path: str = None) -> int:
    """
    Match MapPoints seen in q_keyframe to keypoints in t_frame within a spatial window
    around each q_keyframe keypoint, applying descriptor ratio test.

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
            if t_feat.in_map:
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
    
    log.info(f"\t Found {len(matched_features)} Point Associations!")

    return len(matched_features)
