import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def invert_transform(T: np.ndarray) -> np.ndarray:
    """
    Efficiently invert a 4x4 transformation matrix assuming it is composed of
    a 3x3 orthonormal rotation part (R) and a 3x1 translation part (t).

    Parameters
    ----------
    T : np.ndarray
        A 4x4 homogeneous transformation matrix of the form:
        [ R  t ]
        [ 0  1 ]

    Returns
    -------
    T_inv : np.ndarray
        The inverse of T, also a 4x4 homogeneous transformation matrix.
    """
    # Extract rotation (R) and translation (t)
    R = T[:3, :3]
    t = T[:3, 3]

    # Create an empty 4x4 identity matrix for the result
    T_inv = np.eye(4)

    # R^T goes in the top-left 3x3
    T_inv[:3, :3] = R.T

    # -R^T * t goes in the top-right 3x1
    T_inv[:3, 3] = -R.T @ t

    return T_inv

def transform_points(points_3d: np.ndarray, T: np.ndarray):
    """
    Apply a 4x4 transformation matrix T to a Nx3 array of 3D points.
    Returns a Nx3 array of transformed 3D points.

    Args:
        points_3d: Point with shape (N, 3)
        T: Transformation with shape (4, 4)
    """
    # 1. Convert Nx3 -> Nx4 (homogeneous)
    ones = np.ones((points_3d.shape[0], 1))
    points_hom = np.hstack([points_3d, ones]).T # (4, N)

    # 2. Multiply by the transform (assume row vectors)
    transformed_hom = (T @ points_hom).T        # (N, 4)

    # 3. Normalize back to 3D
    w = transformed_hom[:, 3]
    x = transformed_hom[:, 0] / w
    y = transformed_hom[:, 1] / w
    z = transformed_hom[:, 2] / w
    transformed_3d = np.column_stack((x, y, z)) # (N, 3)

    return transformed_3d

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
