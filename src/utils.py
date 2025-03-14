import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


############################### Saving ###############################

def save_depth(image, save_path):
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
    depth_path = save_path.parent / (save_path.name + '_depth.png')
    cv2.imwrite(depth_path, image)

    # Convert the depth image to 3d points
    points = depth_to_3d_points(image)

    # Extract the Z (depth) component from the points
    Z = points[:, 2]
    
    # Normalize the Z values to an 8-bit range
    # Handling NaNs or Infs if they exist:
    Z = Z[np.isfinite(Z)]
    if Z.size == 0:
        print("Warning: No valid Z points to create a heatmap.")
        return
    
    # Z_norm is a 1D array. Reshape into a w*h image
    h, w = image.shape[:2]
    Z_img = Z.reshape(h, w)
    
    # Use matplotlib to create a heatmap with a colorbar
    plt.figure(figsize=(8, 6))
    plt.imshow(Z_img, cmap='jet', aspect='auto', origin='upper')
    plt.colorbar(label="Depth (m)")
    plt.title("Depth Heatmap")
    heatmap_path = save_path.parent / (save_path.name + '_heat.png')
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()
    
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

def save_2_images(image1, image2, save_path):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convert images to NumPy arrays if they are PIL images
    if isinstance(image1, Image.Image):
        image1 = np.array(image1)
    if isinstance(image2, Image.Image):
        image2 = np.array(image2)

    # Ensure both images are in a format that can be saved by OpenCV
    if isinstance(image1, np.ndarray):
        if len(image1.shape) == 3 and image1.shape[2] == 3:  # Check if it's a 3-channel image
            image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    if isinstance(image2, np.ndarray):
        if len(image2.shape) == 3 and image2.shape[2] == 3:  # Check if it's a 3-channel image
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

    # Ensure both images have the same height
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    if height1 != height2:
        # Resize the second image to match the height of the first image
        image2 = cv2.resize(image2, (int(width2 * height1 / height2), height1))

    # Concatenate images side by side
    combined_image = np.hstack((image1, image2))

    # Save the combined image using OpenCV
    cv2.imwrite(save_path, combined_image)

############################### Transformations ###############################

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
    """
    # 1. Convert Nx3 -> Nx4 (homogeneous)
    ones = np.ones((points_3d.shape[0], 1))
    points_hom = np.hstack([points_3d, ones])  # shape (N, 4)

    # 2. Multiply by the transform (assume row vectors)
    transformed_hom = points_hom @ T.T  # shape (N, 4)

    # 3. Normalize back to 3D
    w = transformed_hom[:, 3]
    x = transformed_hom[:, 0] / w
    y = transformed_hom[:, 1] / w
    z = transformed_hom[:, 2] / w
    transformed_3d = np.column_stack((x, y, z))

    return transformed_3d

def get_yaw(R: np.ndarray):
    # return np.degrees(np.arctan2(R[1,0],R[0,0]))
    return np.degrees(np.arctan2(R[0, 2], R[2, 2]))

############################### Depth ###############################

def depth_to_3d_points(depth_image, cx = 319.5, cy = 239.5, fx=525.0, fy=525.0, factor=5000):
    """
    Convert a depth image to 3D points using camera intrinsics.

    Parameters:
        depth_image (np.ndarray): The depth image.
        cx (float): The x-coordinate of the principal point.
        cy (float): The y-coordinate of the principal point.
        fx (float): The focal length in the x-axis.
        fy (float): The focal length in the y-axis.
        factor (float): The scaling factor for depth values (default is 5000 for 16-bit depth images).

    Returns:
        np.ndarray: An array of 3D points in the camera coordinate frame.
    """
    height, width = depth_image.shape
    points = []

    for v in range(height):
        for u in range(width):
            Z = depth_image[v, u] / factor
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append((X, Y, Z))

    return np.array(points, dtype=np.float64)

def keypoints_depth_to_3d_points(kpts, depth_image, cx, cy, fx, fy, factor=5000):
    """
    Convert 2D keypoints to 3D points using the depth image and camera intrinsics.

    Parameters:
        kpts (np.ndarray): The 2D keypoints (Nx2).
        depth_image (np.ndarray): The depth image.
        cx (float): The x-coordinate of the principal point.
        cy (float): The y-coordinate of the principal point.
        fx (float): The focal length in the x-axis.
        fy (float): The focal length in the y-axis.
        factor (float): The scaling factor for depth values (default is 5000 for 16-bit depth images).

    Returns:
        np.ndarray: An array of 3D points.
    """
    points_3d = []
    valid_mask = np.zeros(len(kpts), dtype=bool)
    j = 0
    for i, pt in enumerate(kpts):
        # Extract pixel coordinates
        u, v = int(pt[0]), int(pt[1])

        # Extract the depth (z) at that coordinate
        Z = depth_image[v, u] / factor

        # Skip points with zero depth and too far away points
        if Z <= 0:
            j += 1
            continue
        # if Z > 5:
        #     j += 1
        #     continue

        # Convert the pixel coordinates to the X, Y coordinates
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        points_3d.append((X, Y, Z))
        valid_mask[i] = True
    # print(f"Removed {j}/{len(kpts)} points that are too far away!")

    points_3d = np.array(points_3d, dtype=np.float64)
    
    return points_3d, valid_mask

############################### Others ###############################

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