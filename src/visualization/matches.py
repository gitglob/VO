import os
import cv2
import numpy as np
import matplotlib

from config import results_dir
matplotlib.use('tkAgg')


# Function to visualize the found feature matches
def plot_matches(matches, q_frame, t_frame, save_path: str = None):
    if isinstance(matches, np.ndarray):
        matches = matches.tolist()
        
    q_img = q_frame.img
    q_kpts = q_frame.keypoints

    t_img = t_frame.img
    t_kpts = t_frame.keypoints

    if len(matches) > 50:
        matches = matches[:50]

    # Draw the matches on the images
    matched_image = cv2.drawMatches(q_img, q_kpts, t_img, t_kpts, matches, outImg=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Save the image with matched features
    if not save_path:
        save_path = results_dir / "matches" / f"{q_frame.id}_{t_frame.id}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(str(save_path), matched_image)

def plot_reprojection(img: np.ndarray, pxs: np.ndarray, reproj_pxs: np.ndarray, path: str):
    reproj_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(len(pxs)):
        obs = tuple(np.int32(pxs[i]))
        reproj = tuple(np.int32(reproj_pxs[i]))
        cv2.circle(reproj_img, obs, 2, (0, 0, 255), -1)    # Observed points (red)
        cv2.circle(reproj_img, reproj, 3, (0, 255, 0), 1)  # Projected points (green)
        cv2.line(reproj_img, obs, reproj, (255, 0, 0), 1)  # Error line (blue)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, reproj_img)

