import cv2
import numpy as np
import src.utils as utils
import src.visualization as vis

from config import results_dir, SETTINGS, log


debug = SETTINGS["generic"]["debug"]
MIN_MATCHES = SETTINGS["initialization"]["min_matches"]
LOWE_RATIO = SETTINGS["initialization"]["lowe_ratio"]


############################### Feature Matching ##########################################


def matchFeatures(q_frame: utils.Frame, t_frame: utils.Frame):
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
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # Match descriptors
    matches = matcher.knnMatch(q_frame.descriptors, t_frame.descriptors, k=2)
    if len(matches) < MIN_MATCHES:
        return None
    filtered_matches = utils.ratio_filter(matches, LOWE_RATIO)
    log.info(f"\t Lowe's Test filtered {len(matches) - len(filtered_matches)}/{len(matches)} matches!")
    if len(filtered_matches) < MIN_MATCHES:
        return None
    unique_matches = utils.unique_filter(filtered_matches)
    log.info(f"\t Uniqueness filtered {len(filtered_matches) - len(unique_matches)}/{len(filtered_matches)} matches!")
    if len(unique_matches) < MIN_MATCHES:
        return None

    # Save the matches
    if debug:
        match_save_path = results_dir / f"initialization/0-raw" / f"{q_frame.id}_{t_frame.id}.png"
        vis.plot_matches(unique_matches, q_frame, t_frame, save_path=match_save_path)

    return np.array(unique_matches, dtype=object)
