import os
from typing import Literal
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import SETTINGS
from src.others.frame import Frame


SIM_THRESHOLD = SETTINGS["place_recognition"]["similarity_threshold"]


def load_vocabulary(type: Literal["dbow", "cv2"]):
    """Loads a visual words vocabulary"""
    vocab_path = f"vocabulary/kitti_{type}.npy"
    if os.path.exists(Path(vocab_path)):
        vocabulary = np.load(vocab_path)
        return vocabulary
    else:
        raise(ValueError(f"Vocabulary {vocab_path} does not exist!"))

def find_loop_closure(frame: Frame, database: list):
    """
    Compute the BoW descriptor for a new image and compare it against all descriptors in the database.
    Returns the best matching frame id and the similarity score if the highest similarity exceeds the threshold.
    Otherwise, returns None.
    """
    if frame.bow_hist is None:
        print("No BoW descriptor computed for the new image.")
        return None

    best_match = None
    best_similarity = 0.0

    # Compare the new histogram with each entry in the database.
    for entry in database:
        # Skip itself
        if entry["frame_id"] == frame.id:
            continue

        # Use cosine similarity: higher score indicates greater similarity.
        score = cosine_similarity(frame.bow_hist, entry["hist"])[0][0]
        print(f"Comparing to frame {entry['frame_id']}, similarity score: {score:.3f}")
        if score > best_similarity:
            best_similarity = score
            best_match = entry

    # If the best similarity exceeds the threshold, declare a loop closure
    if best_similarity >= SIM_THRESHOLD:
        print(f"Loop closure candidate found: Frame {best_match['frame_id']} with similarity {best_similarity:.3f}")
        return best_match["frame_id"], best_similarity
    else:
        print(f"No loop closure candidate found! Best similarity: {best_similarity:.3f}")
        return None
