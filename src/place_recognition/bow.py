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


def load_vocabulary(type: Literal["dbow", "cv2"]):
    """Loads a visual words vocabulary"""
    vocab_path = f"vocabulary/kitti_{type}.npy"
    if os.path.exists(Path(vocab_path)):
        vocabulary = np.load(vocab_path)
        return vocabulary
    else:
        raise(ValueError(f"Vocabulary {vocab_path} does not exist!"))

def query_recognition_candidate(frame: utils.Frame):
    """
    Compare the BoW descriptor in an image with all descriptors in a database.
    Returns the best matching frame id and the similarity score if the highest similarity exceeds the threshold.
    Otherwise, returns None.
    """
    log.info(f"\t Querying database with frame {frame.id}")
    if frame.bow_hist is None:
        log.warning("\t No BoW descriptor computed for the new image.")
        return None

    candidates_ids = set()
    best_match_id = None
    best_similarity = 0.0

    # Iterate over all database visual words
    for v_word_id in ctx.bow_db.keys():
        other_kf_id_list = ctx.bow_db[v_word_id]
        # Iterate over the keyframes that saw it
        for other_kf_id in other_kf_id_list:
            # Skip itself
            if other_kf_id == frame.id:
                continue
            other_kf = ctx.map.keyframes[other_kf_id]

            # Compare the histograms of the 2 frames
            # Use cosine similarity: higher score indicates greater similarity.
            score = cosine_similarity(frame.bow_hist, other_kf.bow_hist)[0][0]
            log.info(f"\t Comparing to frame {other_kf_id}, similarity score: {score:.3f}")
            if score > best_similarity:
                best_similarity = score
                best_match_id = other_kf_id

            # Find recognition candidates_ids
            if score > SIM_THRESHOLD:
                candidates_ids.add(other_kf_id)

    if len(candidates_ids) == 0:
        log.warning("\t Recognition candidate not found!")
    else:
        log.info(f"\t Found {len(candidates_ids)} relocalization candidates.")

    if best_match_id is not None:
        log.info(f"\t Best match: Keyframe #{best_match_id} with similarity: {best_similarity:.3f}")

    return candidates_ids
