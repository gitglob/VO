import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')


# Function to plot keypoints
def plot_keypoints(image, keypoints, save_path):
    # Draw keypoints on the image
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the image with matched features
    cv2.imwrite(str(save_path), img_with_keypoints)

def plot_pixels(img: np.ndarray, pixels: np.ndarray, save_path: str):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for u, v in pixels:
        x, y = int(u), int(v)
        cv2.circle(img, (x, y), 3, (0, 255, 0), 1)  # Draw the keypoint
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img)
    