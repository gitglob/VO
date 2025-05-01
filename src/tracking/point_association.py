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
    log.info(f"Searching for map<->frame #{t_frame.id} correspondences!")
    
    # Extract the in view descriptors
    map_descriptors = []
    map_point_ids = []
    map_pixels = []
    for p in ctx.map.points.values():
        if p.num_observations == 0:
            continue
        # Get the descriptors from every observation of a point
        # an alternative would be to get the best point descriptor!
        desc, kpt = p.best_descriptor()

        map_descriptors.append(desc)
        map_point_ids.append(p.id)
        map_pixels.append(kpt.pt)
        
    map_descriptors = np.array(map_descriptors)
    map_point_ids = np.array(map_point_ids)
    map_pixels = np.array(map_pixels)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # Match descriptors
    matches_knn = bf.knnMatch(map_descriptors, t_frame.descriptors, k=2)
    log.info(f"\t Found {len(matches_knn)} matches!")
    if len(matches_knn) < MIN_MATCHES: return -1
    filtered_matches = utils.ratio_filter(matches_knn, LOWE_RATIO)
    log.info(f"\t Lowe's Test filtered {len(matches_knn) - len(filtered_matches)}/{len(matches_knn)} matches!")
    if len(filtered_matches) < MIN_MATCHES: return -1
    matches = utils.unique_filter(filtered_matches)
    log.info(f"\t Uniqueness filtered {len(filtered_matches) - len(matches)}/{len(filtered_matches)} matches!")
    if len(matches) < MIN_MATCHES: return -1
    
    # Finally, filter using the epipolar constraint
    q_pixels = np.array([map_pixels[m.queryIdx] for m in matches], dtype=np.float64)
    t_pixels = np.array([t_frame.keypoints[m.trainIdx].pt for m in matches], dtype=np.float64)
    ret = utils.enforce_epipolar_constraint(q_pixels, t_pixels)
    if ret is None:
        log.warning("Failed to apply epipolar constraint..")
        return -1
    epi_mask, _, use_homography = ret
    log.info(f"\t Epipolar Constraint: Filtered {sum(~epi_mask)}/{len(q_pixels)} matches! (Using: {'Homography' if use_homography else 'Essential'}.)")
    matches = np.array(matches)[epi_mask]
    if len(matches) < MIN_MATCHES: return -1
    
    # Prepare results
    for m in matches:
        pid = map_point_ids[m.queryIdx]
        point = ctx.map.points[pid]

        t_kpt = t_frame.keypoints[m.trainIdx]
        t_feat = t_frame.features[t_kpt.class_id]

        t_feat.match_map_point(point, m.distance)
        point.observe(ctx.map._kf_counter, t_frame.id, t_feat.kpt, t_feat.desc)

    # Save the matches
    if DEBUG:
        save_path=results_dir / "tracking/matches" / f"map_{t_frame.id}.png"
        t_pxs = np.array([t_frame.keypoints[m.trainIdx].pt for m in matches], dtype=np.float64)
        vis.plot_pixels(t_frame.img, t_pxs, save_path=save_path)

    log.info(f"\t Found {len(matches)} Point Associations!")
    return len(matches)

