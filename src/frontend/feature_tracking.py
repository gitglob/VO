from typing import List
import cv2
import numpy as np
from src.frame import Frame
from src.visualize import plot_matches

from config import results_dir, debug, SETTINGS


############################### Feature Matching ##########################################

def match_features(q_frame: Frame, t_frame: Frame, K: np.ndarray, stage: str):
    """
    Matches features between two frames.
    
    Each match has the following attributes:
        distance: The distance between the descriptors (a measure of similarity; lower is better).
        trainIdx: The index of the descriptor in the training set (second image).
        queryIdx: The index of the descriptor in the query set (first image).
        imgIdx: The index of the image (if multiple images are being used).
    """
    print(f"Matching features between frames: {q_frame.id} & {t_frame.id}...")

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # 1) Match descriptors (KNN)
    matches = bf.knnMatch(q_frame.descriptors, t_frame.descriptors, k=2)

    # 2) Filter matches with your custom filter (lowe ratio, distance threshold, etc.)
    filtered_matches = filter_matches(matches)

    if len(matches) < SETTINGS["matches"]["min"]:
        return []

    # 3) Create masks indicating whether each keypoint is used in a match
    q_mask = np.zeros(len(q_frame.keypoints), dtype=bool)
    t_mask = np.zeros(len(t_frame.keypoints), dtype=bool)

    for m in filtered_matches:
        q_mask[m.queryIdx] = True
        t_mask[m.trainIdx] = True

    # 4) **Propagate keypoint IDs**  
    propagate_keypoints(q_frame, t_frame, filtered_matches)

    # 5) Store the matches in each frame
    q_frame.set_matches(t_frame.id, filtered_matches, q_mask, "query")
    t_frame.set_matches(q_frame.id, filtered_matches, t_mask, "train")
    print(f"\t{len(filtered_matches)} matches left!")
            
    # Save the matches
    if debug:
        match_save_path = results_dir / f"matches/{stage}" / f"{q_frame.id}_{t_frame.id}.png"
        plot_matches(q_frame, t_frame, save_path=match_save_path)

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

def filter_matches(matches, ratio=0.8):
    """Filter out matches using Lowe's Ratio Test"""
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    if debug:
        print(f"\tLowe's Test filtered {len(matches) - len(good_matches)}/{len(matches)} matches!")
    return good_matches
   
