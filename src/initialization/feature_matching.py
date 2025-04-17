from typing import List
import cv2
import numpy as np
from src.others.frame import Frame
from src.others.filtering import filterMatches
from src.others.visualize import plot_matches

from config import results_dir, SETTINGS, log


debug = SETTINGS["generic"]["debug"]


############################### Feature Matching ##########################################


def matchFeaturesXG(q_frame: Frame, t_frame: Frame, stage: str):
    """
    Matches features between two frames.
    
    Each match has the following attributes:
        distance: The distance between the descriptors (a measure of similarity; lower is better).
        trainIdx: The index of the descriptor in the training set (second image).
        queryIdx: The index of the descriptor in the query set (first image).
        imgIdx: The index of the image (if multiple images are being used).
    """
    log.info(f"Matching features between frames: {q_frame.id} & {t_frame.id}...")

    # Create BFMatcher object
    index_params = dict(algorithm=6,   # FLANN_INDEX_LSH
                    table_number=5, 
                    key_size=10, 
                    multi_probe_level=2)
    search_params = {}
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    # 1) Match descriptors (KNN)
    matches = matcher.match(q_frame.descriptors, t_frame.descriptors)
    if len(matches) < SETTINGS["matches"]["min"]:
        return []

    # 2) Filter matches
    min_dist = -9999
    max_dist =  9999
    for m in matches:
        dist = m.distance
        if dist < min_dist:
            min_dist = dist
        if dist > max_dist:
            max_dist = dist
    dist_threshold = max(min_dist * SETTINGS["matches"]["xiang_gao_match_ratio"], 30)

    good_matches = []
    for m in matches:
        if m.distance < dist_threshold:
            good_matches.append(m)

    if debug:
        log.info(f"\t Xiang Gao match ratio's ratio filtered {len(matches) - len(good_matches)}/{len(matches)} matches!")

    # Next, ensure uniqueness by keeping only the best match per train descriptor.
    unique_matches = {}
    for m in good_matches:
        # If this train descriptor is not seen yet, or if the current match is better, update.
        if m.trainIdx not in unique_matches or m.distance < unique_matches[m.trainIdx].distance:
            unique_matches[m.trainIdx] = m

    matches = list(unique_matches.values())

    if debug:
        log.info(f"\t Uniqueness filtered {len(good_matches) - len(matches)}/{len(good_matches)} matches!")

    # 3) **Propagate keypoint IDs**
    propagate_keypoints(q_frame, t_frame, matches)

    # 4) Store the matches in each frame
    q_frame.set_matches(t_frame.id, matches, "query")
    t_frame.set_matches(q_frame.id, matches, "train")
    if debug:
        log.info(f"\t{len(matches)} matches left!")
            
    # Save the matches
    if debug:
        match_save_path = results_dir / f"matches/{stage}" / f"{q_frame.id}_{t_frame.id}.png"
        plot_matches(matches, q_frame, t_frame, save_path=match_save_path)

    return matches

def matchFeatures(q_frame: Frame, t_frame: Frame, stage: str):
    """
    Matches features between two frames.
    
    Each match has the following attributes:
        distance: The distance between the descriptors (a measure of similarity; lower is better).
        trainIdx: The index of the descriptor in the training set (second image).
        queryIdx: The index of the descriptor in the query set (first image).
        imgIdx: The index of the image (if multiple images are being used).
    """
    if debug:
        log.info(f"Matching features between frames: {q_frame.id} & {t_frame.id}...")

    # Create BFMatcher object
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # 1) Match descriptors (KNN)
    matches = matcher.knnMatch(q_frame.descriptors, t_frame.descriptors, k=2)
    if len(matches) < SETTINGS["matches"]["min"]:
        return []

    # # 2) Filter matches with your custom filter (lowe ratio, distance threshold, etc.)
    matches = filterMatches(matches)
    if len(matches) < SETTINGS["matches"]["min"]:
        return []

    # 3) **Propagate keypoint IDs**
    propagate_keypoints(q_frame, t_frame, matches)

    # 4) Store the matches in each frame
    q_frame.set_matches(t_frame.id, matches, "query")
    t_frame.set_matches(q_frame.id, matches, "train")
    if debug:
        log.info(f"\t{len(matches)} matches left!")
            
    # Save the matches
    if debug:
        match_save_path = results_dir / f"matches/{stage}" / f"{q_frame.id}_{t_frame.id}.png"
        plot_matches(matches, q_frame, t_frame, save_path=match_save_path)

    return matches

def propagate_keypoints(q_frame: Frame, t_frame: Frame, matches: List[cv2.DMatch]):
    """Merges the keypoint identifiers for the matches features between query and train frames."""
    for m in matches:
        q_idx = m.queryIdx
        t_idx = m.trainIdx

        q_kp = q_frame.keypoints[q_idx]
        t_kp = t_frame.keypoints[t_idx]

        # If the train keypoint has no ID, copy from the query keypoint
        if t_kp.class_id < 0:  # or `t_kp.class_id is None`
            t_kp.class_id = q_kp.class_id

        # If the query keypoint has no ID, copy from the train keypoint
        elif q_kp.class_id <= 0:
            q_kp.class_id = t_kp.class_id

        # If both have IDs but they differ, pick a strategy (e.g., overwrite one)
        elif q_kp.class_id != t_kp.class_id:
            # Naive approach: unify by assigning query ID to train ID
            # or vice versa. Real SLAM systems often handle merges in a global map.
            t_kp.class_id = q_kp.class_id
