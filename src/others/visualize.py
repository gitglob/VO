import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.spatial.transform import Rotation as R

from config import results_dir
from src.others.utils import get_yaw
matplotlib.use('TkAgg')


############################### Pose Visualization ###############################

def plot_trajectory(poses, gt_poses, i, save_path=results_dir / "trajectory", ba_poses=None):
    if ba_poses is not None:
        save_path = save_path / f"{i}_ba.png"
    else:
        save_path = save_path / f"{i}.png"

    num_poses = len(poses)
    poses = np.array(poses)
    gt_poses = np.array(gt_poses)

    fig = plt.figure(figsize=(12, 6))

    # First subplot: XZ 2D view
    ax1 = fig.add_subplot(121)
    ax1.plot(poses[:, 0, 3], poses[:, 2, 3], 'b-', label='XZ')
    ax1.plot(gt_poses[:, 0, 3], gt_poses[:, 2, 3], 'r--', label='Ground Truth')
    
    # Mark the start and end points with bubbles
    ax1.scatter(poses[0,0,3], poses[0,2,3], color='black', s=100, alpha=0.7, label='Start')
   
    # Extract Euler angles (roll, pitch, yaw)
    pitch = np.zeros((num_poses, 1))
    gt_pitch = np.zeros((num_poses, 1))
    for i in range(num_poses):
        # Extract the estimation
        Rot = poses[i, :3, :3]
        pitch[i] = get_yaw(Rot)

        # Extract the ground truth
        gt_Rot = gt_poses[i, :3, :3]
        gt_pitch[i] = get_yaw(gt_Rot)

    pitch = np.unwrap(pitch)
    gt_pitch = np.unwrap(gt_pitch)

    # Second subplot: angle plot
    ax2 = fig.add_subplot(122)

    ax2.plot(np.arange(num_poses), -pitch, 'b-', label='-Pitch')
    ax2.plot(np.arange(num_poses), -gt_pitch, 'r--', label='g.t.')

    # Add the BA poses if provided
    if ba_poses is not None:
        ba_poses = np.array(ba_poses)
        ax1.plot(ba_poses[:, 0, 3], ba_poses[:, 2, 3], 'g-*', label='BA')
        ba_pitch = np.zeros((num_poses, 1))
        for i in range(num_poses):
            # Extract the estimation
            Rot = ba_poses[i, :3, :3]
            ba_pitch[i] = get_yaw(Rot)

        ba_pitch = np.unwrap(ba_pitch)
        ax2.plot(np.arange(num_poses), -ba_pitch, 'g-*', label='BA')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_title('XZ')
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Degrees')
    ax2.set_title('Angle')
    ax2.legend()
    ax2.grid(True)

    fig.suptitle('2D Trajectory')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.close(fig)

def plot_trajectory_3d(poses, save_path=results_dir / "vo" / "final_trajectory.png"):
    poses = np.array(poses)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the trajectory
    ax.plot(poses[:, 0, 3], poses[:, 2, 3], -poses[:, 1, 3], 'b-', label='Trajectory')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('-Y')
    ax.set_title('Map and Trajectory')
    ax.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.close(fig)

############################### Ground Truth Visualization ###############################

def plot_ground_truth(ground_truth):
    plot_ground_truth_2d(ground_truth)
    plot_ground_truth_3d(ground_truth)

def plot_ground_truth_3d(ground_truth, save_path=results_dir / "ground_truth/3d.png"):
    """ Reads the ground truth data from a file and plots the robot trajectory. """
    # Extract the positions (tx, ty, tz)
    tx = ground_truth.iloc[:, 1].values
    ty = ground_truth.iloc[:, 2].values
    tz = ground_truth.iloc[:, 3].values

    # Plot the trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(tx, tz, -ty, label='Robot Trajectory')
    
    # Mark the start and end points with bubbles
    ax.scatter(tx[0], tz[0], -ty[0], color='green', s=100, label='Start')
    ax.scatter(tx[-1], tz[-1], -ty[-1], color='red', s=100, label='End')

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('-Y')
    ax.set_title('Robot Trajectory')
    ax.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.close(fig)

def plot_ground_truth_2d(ground_truth, save_path=results_dir / "ground_truth/2d.png"):
    """ 
    Plots the robot trajectory in XY coordinates and the yaw angle over time.

    Args:
        ground_truth_df (pd.DataFrame): DataFrame of shape (N, 12) containing rows of flattened 3x4 pose matrices.
        save_path (str, optional): Path to save the plot.
    """
    # Extract translations (tx, ty)
    tx = ground_truth.iloc[:, 1].values
    ty = ground_truth.iloc[:, 3].values
    # tz = ground_truth.iloc[:, 3].values

    # Extract rotation
    pitch = np.empty((len(ground_truth), 3))
    for i in range(len(ground_truth)):
        qx, qy, qz, qw = ground_truth.iloc[i, 4:8].values
        Rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
        pitch[i] = get_yaw(Rot)

    pitch = np.unwrap(pitch[:,1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the XY trajectory
    ax1.plot(tx, ty, 'b-', label='XY Trajectory')
    ax1.scatter(tx[0], ty[0], color='green', s=100, label='Start')  # Start point
    ax1.scatter(tx[-1], ty[-1], color='red', s=100, label='End')    # End point
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Road Trajectory')
    ax1.legend()
    ax1.grid(True)

    # Plot the yaw trajectory
    ax2.plot(np.arange(len(ground_truth)), -pitch, 'g-', label='-pitch')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Degrees')
    ax2.set_title('Vehicle Orientation over Time')
    ax2.legend()
    ax2.grid(True)

    fig.suptitle('2D Trajectory')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.close(fig)
        
############################### Feature Visualization ###############################

# Function to plot keypoints
def plot_keypoints(image, keypoints, save_path):
    # Draw keypoints on the image
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the image with matched features
    cv2.imwrite(save_path, img_with_keypoints)

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
        save_path = results_dir / f"matches/" / f"{q_frame.id}_{t_frame.id}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, matched_image)

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

def plot_pixels(img: np.ndarray, pixels: np.ndarray, save_path: str):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for u, v in pixels:
        x, y = int(u), int(v)
        cv2.circle(img, (x, y), 3, (0, 255, 0), 1)  # Draw the keypoint
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, img)
    