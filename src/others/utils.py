import os
import shutil
import logging
import os
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def make_cv2_matches(pairs, distances=None):
    """
    pairs: list of (q_idx, t_idx, dist)
    distances:  optional list of floats, same length, or None → all zeros
    """
    cv_matches = []
    for i, (q_idx, t_idx, dist) in enumerate(pairs):
        m = cv2.DMatch(q_idx, t_idx, 0, dist)
        cv_matches.append(m)
    return cv_matches

def setup_logger(log_dir: str = "logs") -> logging.Logger:
    # 1. Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # 2. Build filename based on today's date
    timestamp = datetime.now().strftime("%d-%m-%y_%H:%M:%S")
    log_path = os.path.join(log_dir, f"{timestamp}.log")

    # 3. Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # capture INFO and above

    # avoid adding multiple handlers if called multiple times
    if not logger.handlers:
        # 4. File handler writes to YYYY‑MM‑DD.log
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)

        # 5. Formatter for timestamp, level and message
        fmt = logging.Formatter(
            "%(asctime)s - %(levelname)s : %(message)s",
            datefmt="%d-%m-%Y %H:%M:%S"
        )
        fh.setFormatter(fmt)

        # 6. Optionally also log to console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

def isnan(p: np.ndarray):
    """Checks if a 3d point is nan."""
    if np.isnan(p[0]) or np.isnan(p[1]) or np.isnan(p[2]):
        return True
    return False

def delete_subdirectories(data_dir):
    # Convert to Path object if not already
    data_dir = Path(data_dir)
    if not os.path.isdir(data_dir):
        return

    # Iterate through the contents of the directory
    for item in data_dir.iterdir():
        # Check if the item is a directory
        if item.is_dir():
            # Recursively delete the directory and its contents
            shutil.rmtree(item)
   
def save_image(image, save_path):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Check if the image is a PIL image, convert it to a NumPy array
    if isinstance(image, Image.Image):
        image = np.array(image)

    # If the image is a NumPy array, ensure it's in a format that can be saved by OpenCV
    if isinstance(image, np.ndarray):
        # Convert the image to BGR format if it's in RGB (PIL is usually in RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:  # Check if it's a 3-channel image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Save the image using OpenCV
    cv2.imwrite(save_path, image)

def get_yaw(R: np.ndarray):
    # return np.degrees(np.arctan2(R[1,0],R[0,0]))
    return np.degrees(np.arctan2(R[0, 2], R[2, 2]))
