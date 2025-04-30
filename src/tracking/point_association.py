import numpy as np
import cv2

import src.utils as utils
import src.visualization as vis
import src.globals as ctx


from config import SETTINGS, results_dir, K, log


scale_factor = SETTINGS["orb"]["scale_factor"]
n_levels = SETTINGS["orb"]["level_pyramid"]

LOWE_RATIO = SETTINGS["tracking"]["lowe_ratio"]
MIN_MATCHES = SETTINGS["tracking"]["min_matches"]

DEBUG = SETTINGS["generic"]["debug"]
W = SETTINGS["camera"]["width"]
H = SETTINGS["camera"]["height"]


def map_search(t_frame: utils.Frame):
    """
    Matches the map points seen in previous frames with the current frame.

    Args:
        t_frame: The current t_frame, which has .keypoints and .descriptors

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
    if DEBUG:
        log.info(f"\t Found {len(matches)} matches!")
    if len(matches) < MIN_MATCHES:
        return -1
    filtered_matches = utils.ratio_filter(matches, LOWE_RATIO)
    if len(filtered_matches) < MIN_MATCHES: return -1
    filtered_matches = utils.unique_filter(filtered_matches)
    if len(filtered_matches) < MIN_MATCHES: return -1
    
    # Finally, filter using the epipolar constraint
    q_pixels = np.array([map_pixels[m.queryIdx] for m in filtered_matches], dtype=np.float64)
    t_pixels = np.array([t_frame.keypoints[m.trainIdx].pt for m in filtered_matches], dtype=np.float64)
    epipolar_constraint_mask, _, _ = utils.enforce_epipolar_constraint(q_pixels, t_pixels)
    if epipolar_constraint_mask is None:
        log.warning("Failed to apply epipolar constraint..")
        return -1
    filtered_matches = np.array(filtered_matches)[epipolar_constraint_mask]
    
    # Prepare results
    for m in filtered_matches:
        pid = map_point_ids[m.queryIdx]
        point = ctx.map.points[pid]

        t_kpt = t_frame.keypoints[m.trainIdx]
        t_feat = t_frame.features[t_kpt.class_id]

        t_feat.match_map_point(point, m.distance)
        point.observe(ctx.map._kf_counter, t_frame.id, t_feat.kpt, t_feat.desc)

    # Save the matches
    if DEBUG:
        save_path=results_dir / "tracking/matches" / f"map_{t_frame.id}.png"
        t_pxs = np.array([t_frame.keypoints[m.trainIdx].pt for m in filtered_matches], dtype=np.float64)
        vis.plot_pixels(t_frame.img, t_pxs, save_path=save_path)

    if DEBUG:
        log.info(f"\t Found {len(filtered_matches)} Point Associations!")
    return len(filtered_matches)

