from typing import List
import cv2
import numpy as np
from src.others.frame import Frame
from src.others.filtering import filterMatches
from src.others.visualize import plot_matches

from config import results_dir, SETTINGS, log


debug = SETTINGS["generic"]["debug"]
MIN_MATCHES = SETTINGS["initialization"]["matches"]["min"]
LOWE_RATIO = SETTINGS["initialization"]["matches"]["lowe_ratio"]


############################### Feature Matching ##########################################


def matchFeatures(q_frame: Frame, t_frame: Frame):
    """
    Matches features between two frames.
    
    Each match has the following attributes:
        distance: The distance between the descriptors (a measure of similarity; lower is better).
        trainIdx: The index of the descriptor in the training set (second image).
        queryIdx: The index of the descriptor in the query set (first image).
        imgIdx: The index of the image (if multiple images are being used).
    """
    if debug:
        log.info(f"[Initialization] Matching features between frames: {q_frame.id} & {t_frame.id}...")

    # Create BFMatcher object
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # 1) Match descriptors (KNN)
    matches = matcher.knnMatch(q_frame.descriptors, t_frame.descriptors, k=2)
    if len(matches) < MIN_MATCHES:
        return None

    # # 2) Filter matches with your custom filter (lowe ratio, distance threshold, etc.)
    matches = filterMatches(matches, LOWE_RATIO)
    if len(matches) < MIN_MATCHES:
        return None
    if debug:
        log.info(f"\t {len(matches)} matches left!")

    # Save the matches
    if debug:
        match_save_path = results_dir / f"matches/initialization/0-raw" / f"{q_frame.id}_{t_frame.id}.png"
        plot_matches(matches, q_frame, t_frame, save_path=match_save_path)

    return np.array(matches, dtype=object)
