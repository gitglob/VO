import numpy as np
import cv2

import src.backend as backend
import src.place_recognition as pr
import src.utils as utils
import src.visualization as vis
import src.globals as ctx
from .feature_matching import search_by_projection, search_by_bow, window_search
from .pnp import estimate_relative_pose
from .utils import constant_velocity_model

from config import SETTINGS, results_dir, K, log


scale_factor = SETTINGS["orb"]["scale_factor"]
n_levels = SETTINGS["orb"]["level_pyramid"]

debug = SETTINGS["generic"]["debug"]
W = SETTINGS["camera"]["width"]
H = SETTINGS["camera"]["height"]
HAMMING_THRESHOLD = SETTINGS["point_association"]["hamming_threshold"]


########################## Constant Velocity ##########################

def constant_velocity(q_frame: utils.Frame, t_frame: utils.Frame):
    """Performs tracking using a constant velocity model"""
    if ctx.map.num_keyframes() < 4 or t_frame.id < ctx.map.last_reloc + 2:
        ok = track_from_previous_frame(q_frame, t_frame)
    else:
        ok = track_with_motion_model(q_frame, t_frame)
        if not ok:
            ok = track_from_previous_frame(q_frame, t_frame)
            if not ok:
                return False

    # Compute the BOW representation of the keyframe
    t_frame.compute_bow()
    
    return True

def track_with_motion_model(q_frame: utils.Frame, t_frame: utils.Frame):
    """Search matches in a radius around the predicted keypoints positions assuming constant velocity"""
    # Predict the new pose based on a constant velocity model
    constant_velocity_model(t_frame)

    # Match these map points with the current frame
    num_matches = search_by_projection(q_frame, t_frame, theta=15, 
                                       save_path=results_dir / "tracking/motion_model" / f"{q_frame.id}_{t_frame.id}.png")
    if num_matches < 20:
        log.warning(f"Tracking with motion model failed! {num_matches} (<10) matches found!")
        return False

    # Perform pose optimization
    ctx.map.add_keyframe(t_frame)
    ba = backend.singlePoseBA(t_frame, verbose=debug)
    ba.optimize()

    return True

def track_from_previous_frame(q_frame: utils.Frame, t_frame: utils.Frame):
    """Search matches in windows around the previous frame keypoints"""
    ## Search for matches in windows around the previous keypoints
    # Use scale constraints
    minOctave = maxOctave/2+1 if ctx.map.num_keyframes() > 5 else 0
    maxOctave = 7
    num_matches = window_search(q_frame, t_frame, 200, minOctave, 
                                save_path = results_dir / "tracking/previous_frame" / f"{q_frame.id}_{t_frame.id}_0.png")
    if num_matches < 10:
        # Don't use scale constraints
        num_matches = window_search(q_frame, t_frame, 100, 0, 
                                    save_path = results_dir / "tracking/previous_frame" / f"{q_frame.id}_{t_frame.id}_1.png")
        
    # Copy the previous pose to the current (will be fixed by BA)
    t_frame.set_pose(q_frame.pose.copy())

    # If not enough matches were found ...
    if num_matches < 10:
        # Try searching again: No scale constraints and no Lowe's test
        num_matches = search_by_projection(q_frame, t_frame, radius=50, 
                                           save_path=results_dir / "tracking/previous_frame" / f"{q_frame.id}_{t_frame.id}_2.png")
    # If enough matches were found ...
    else:
        # Perform pose optimization
        ctx.map.add_keyframe(t_frame)
        ba = backend.singlePoseBA(t_frame, verbose=debug)
        ba.optimize()

        # Search for more points using the (now optimized) pose
        num_matches += search_by_projection(q_frame, t_frame, radius=15, 
                                            save_path=results_dir / "tracking/previous_frame" / f"{q_frame.id}_{t_frame.id}_3.png")

    # Matching failed
    if num_matches < 10:
        log.warning(f"Tracking from previous frame failed! {num_matches} (<10) matches found!")
        return False

    # Perform pose optimization
    ctx.map.add_keyframe(t_frame)
    ba = backend.singlePoseBA(t_frame, verbose=debug)
    ba.optimize()

    return True

########################## Relocalization ##########################

def relocalization(t_frame: utils.Frame):
    """Performs tracking using vBoW relocalization"""
    # Compute the BOW representation of the keyframe
    t_frame.compute_bow()

    # Find keyframe candidates from the BoW database for global relocalization
    kf_candidate_ids = pr.query_recognition_candidate(t_frame)

    # Iterate over all candidates
    log.info(f"Iterating over {len(kf_candidate_ids)} keyframe candidates!")
    for j, kf_id in enumerate(kf_candidate_ids):
        # Extract the candidate keyframe
        cand_frame = ctx.map.get_keyframe(kf_id)

        # Perform point association of its map points with the current frame
        num_matches = search_by_bow(cand_frame, t_frame, 
                                    save_path = results_dir / "tracking/relocalization" / f"{cand_frame.id}_{t_frame.id}_0.png")
        if num_matches < 15:
            continue

        # Estimate the new world pose using PnP (3d-2d)
        ok = estimate_relative_pose(t_frame)
        if ok:
            # Perform pose optimization
            ctx.map.add_keyframe(t_frame)
            ba = backend.singlePoseBA(t_frame, verbose=debug)
            ba.optimize()
            
            # Temporarily set the candidate frame as keyframe
            ctx.map.relocalize(t_frame.id)
            log.info(f"Candidate {j}, keyframe {kf_id}: solvePnP success!")
            return True

    # If global relocalization failed, we need to restart initialization
    ctx.map.remove_observation(t_frame.id)
    return False

########################## Local Map ##########################

def local_map(t_frame: utils.Frame):
    """Performs tracking using a local map (points of neighboring keyframes)"""
    # Set the visible mask
    ctx.map.view(t_frame)

    # Extract a local map from the map
    ctx.local_map = ctx.cgraph.create_local_map(t_frame)
    
    # Project the local map to the frame and search more correspondances
    track_map_points(t_frame)
    
    # Set the found mask
    ctx.map.tracked(t_frame)
    
    # Optimize the camera pose with all the map points found in the frame
    ba = backend.singlePoseBA(t_frame, verbose=debug)
    num_matched_inliers = ba.optimize()
    if num_matched_inliers < 30:
        ctx.map.remove_observation(t_frame.id)
        return False
    
    return True

def track_map_points(t_frame: utils.Frame, theta: int = 15):
    """
    Projects all un-matched map points to a frame and searches more correspondances.

    Returns:
        matched_features: A list of tuples (map_idx, frame_idx) indicating the association of map points
               to current frame keypoints.      
    """
    # Extract the already matched features and map points
    matched_point_ids = t_frame.get_map_point_ids()
    new_matched_features = {}

    # Iterate over all the map points
    for pid in ctx.local_map.point_ids:
        point = ctx.map.points[pid]

        # Skip matched map points
        if pid in matched_point_ids:
            continue

        # Check if the point is in the current camera's frustum
        result = t_frame.is_in_frustum(point)
        if result is False:
            continue
        u, v, scale = result

        # Compare the representative descriptor D of the map point with the 
        # still unmatched ORB features in the frame, at the predicted scale, 
        # and near x, and associate the map point with the best match.
        D = point.best_descriptor

        # Collect candidate current frame un-matched keypoints whose pixel coordinates 
        # fall within a window around the predicted pixel
        octave_idx = np.abs(t_frame.scale_factors - scale).argmin()
        radius = theta * t_frame.scale_factors[octave_idx]
        candidates = []
        for feat_id, feat in t_frame.features.items():
            # Skip already matched features
            if feat.matched:
                continue
            feat_px = feat.kpt.pt
            if (abs(feat_px[0] - u) <= radius and
                abs(feat_px[1] - v) <= radius):
                candidates.append(feat_id)
        
        # If no keypoints are found in the window, skip to the next map point.
        if len(candidates) == 0:
            continue
        
        # For each candidate, compute the descriptor distance using the Hamming norm.
        best_dist = np.inf
        best_feature_id = None
        for feat_id in candidates:
            candidate_desc = t_frame.features[feat_id].desc
            # Compute Hamming distance.
            d = cv2.norm(np.array(D), np.array(candidate_desc), cv2.NORM_HAMMING)
            if d < best_dist:
                best_dist = d
                best_feature_id = feat_id
        
        # Accept the match only if the best distance is below the threshold.
        if best_feature_id is not None and best_dist < HAMMING_THRESHOLD:
            # Make sure that we only keep 1 match per frame pixel
            if best_feature_id not in new_matched_features.keys() or best_dist < new_matched_features[best_feature_id][1]:
                new_matched_features[best_feature_id] = (pid, best_dist)
    
    if debug:
        log.info(f"\t Found {len(new_matched_features)} Point Associations!")

    # Update the frame<->map matches
    for feat_id, (pid, dist) in new_matched_features.items():
        feat = t_frame.features[feat_id]
        feat.match_map_point(pid, dist)
        point = ctx.map.points[pid]
        ctx.map.add_observation(t_frame, feat, point)
    
    if debug and len(new_matched_features.keys()) > 0:
        match_save_path = results_dir / "matches/tracking/map" / f"map_{t_frame.id}.png"
        t_pxs = np.array([t_frame.features[feat_id].kpt.pt for feat_id in new_matched_features.keys()], dtype=np.float64)        
        vis.plot_pixels(t_frame.img, t_pxs, save_path=match_save_path)

    return len(new_matched_features)

