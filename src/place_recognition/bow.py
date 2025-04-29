import os
from typing import Literal
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import SETTINGS, log

import src.utils as utils
import src.local_mapping as mapping
import src.globals as ctx


SIM_THRESHOLD = SETTINGS["place_recognition"]["similarity_threshold"]
DEBUG = SETTINGS["generic"]["debug"]


def load_vocabulary(type: Literal["dbow", "cv2"]):
    """Loads a visual words vocabulary"""
    vocab_path = f"vocabulary/kitti_{type}.npy"
    if os.path.exists(Path(vocab_path)):
        vocabulary = np.load(vocab_path)
        return vocabulary
    else:
        raise(ValueError(f"Vocabulary {vocab_path} does not exist!"))

def query_recognition_candidate(frame: utils.Frame) -> list[tuple[int, float]]:
    """
    Compare the BoW descriptor in an image with all descriptors in a database.
    Returns the best matching frame id and the similarity score if the highest similarity exceeds the threshold.
    Otherwise, returns None.
    """
    if DEBUG:
        log.info(f"\t Querying database with frame {frame.id}")
    if frame.bow_hist is None:
        log.warning("\t No BoW descriptor computed for the new image.")
        return None

    candidates = []
    best_match_id = None
    best_similarity = 0.0

    # Gather unique keyframe IDs from the BoW DB
    all_db_frames = {kf_id 
                     for kf_list in ctx.bow_db.values() 
                     for kf_id in kf_list}
    # remove self and any kf not in the current map
    all_db_frames.discard(frame.id)
    map_frames = ctx.map.keyframe_ids
    frames_that_share_words = all_db_frames & map_frames

    # Iterate over the keyframes that share words with the current frame
    for other_kf_id in frames_that_share_words:
        assert other_kf_id != frame.id
        assert other_kf_id in ctx.map.keyframe_ids
        other_kf = ctx.map.keyframes[other_kf_id]

        # Compare the histograms of the 2 frames
        # Use cosine similarity: higher score indicates greater similarity.
        score = cosine_similarity(frame.bow_hist, other_kf.bow_hist)[0][0]
        if DEBUG:
            log.info(f"\t Comparing to frame {other_kf_id}, similarity score: {score:.3f}")
        if score > best_similarity:
            best_similarity = score
            best_match_id = other_kf_id

        # Find recognition candidates
        if score > SIM_THRESHOLD:
            candidates.append((other_kf_id, score))

    if len(candidates) == 0:
        log.warning("\t Recognition candidates not found!")
        return candidates

    if DEBUG:
        log.info(f"\t Found {len(candidates)} relocalization candidates.")
        log.info(f"\t Best match: Keyframe #{best_match_id} with similarity: {best_similarity:.3f}")

    return candidates
