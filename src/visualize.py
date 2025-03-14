import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.spatial.transform import Rotation as R

from config import results_dir
from src.utils import get_yaw
matplotlib.use('TkAgg')


############################### Pose Visualization ###############################

def plot_trajectory(poses, gt, i, save_path=results_dir / "trajectory"):
    plot_trajectory_2d(poses, gt, save_path / "2d" / f"{i}.png")
    plot_trajectory_6dof(poses, gt, save_path / "6dof" / f"{i}.png")

def plot_trajectory_6dof(poses, gt_poses, save_path=None):
    num_poses = len(poses)
    poses = np.array(poses)
    gt_poses = np.array(gt_poses)
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))
    
    # Extract translations
    tx, ty, tz = poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3]
    gt_tx, gt_ty, gt_tz = gt_poses[:, 0, 3], gt_poses[:, 1, 3], gt_poses[:, 2, 3]
    
    # Extract Euler angles (roll, pitch, yaw)
    euler_angles = np.zeros((num_poses, 3))
    gt_euler_angles = np.zeros((num_poses, 3))
    for i in range(num_poses):
        # Extract the estimation
        pose = poses[i, :3, :3]
        euler_angles[i] = R.from_matrix(pose[:, :3]).as_euler('zyz', degrees=True)

        # Extract the ground truth
        gt_pose = gt_poses[i, :3, :3]
        gt_euler_angles[i] = R.from_matrix(gt_pose[:, :3]).as_euler('zyz', degrees=True)

    rolls = np.degrees(np.unwrap(euler_angles[:, 0]))
    pitches = np.degrees(np.unwrap(euler_angles[:, 1]))
    yaws = np.degrees(np.unwrap(euler_angles[:, 2]))
    
    gt_rolls = np.degrees(np.unwrap(gt_euler_angles[:, 0]))
    gt_pitches = np.degrees(np.unwrap(gt_euler_angles[:, 1]))
    gt_yaws = np.degrees(np.unwrap(gt_euler_angles[:, 2]))

    # Plot translations
    axs[0, 0].plot(np.arange(num_poses), tx, 'b-', label='X Pose')
    axs[0, 0].plot(np.arange(num_poses), gt_tx, 'r-', label='Ground Truth X')
    axs[0, 0].set_title('X')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('m')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    axs[0, 1].plot(np.arange(num_poses), ty, 'b-', label='Y Pose')
    axs[0, 1].plot(np.arange(num_poses), gt_ty, 'r-', label='Ground Truth Y')
    axs[0, 1].set_title('Y')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('m')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    axs[0, 2].plot(np.arange(num_poses), tz, 'b-', label='Z Pose')
    axs[0, 2].plot(np.arange(num_poses), gt_tz, 'r-', label='Ground Truth Z')
    axs[0, 2].set_title('Z')
    axs[0, 2].set_xlabel('Time')
    axs[0, 2].set_ylabel('m')
    axs[0, 2].legend()
    axs[0, 2].grid(True)
    
    # Plot rotations (yaw, pitch, roll in degrees)
    axs[1, 0].plot(np.arange(num_poses), rolls, 'b-', label='Roll')
    axs[1, 0].plot(np.arange(num_poses), gt_rolls, 'r-', label='Ground Truth Roll')
    axs[1, 0].set_title('Roll')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('deg')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(np.arange(num_poses), pitches, 'b-', label='Pitch')
    axs[1, 1].plot(np.arange(num_poses), gt_pitches, 'r-', label='Ground Truth Pitch')
    axs[1, 1].set_title('Pitch')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('deg')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    axs[1, 2].plot(np.arange(num_poses), yaws, 'b-', label='Yaw')
    axs[1, 2].plot(np.arange(num_poses), gt_yaws, 'r-', label='Ground Truth Yaw')
    axs[1, 2].set_title('Yaw')
    axs[1, 2].set_xlabel('Time')
    axs[1, 2].set_ylabel('deg')
    axs[1, 2].legend()
    axs[1, 2].grid(True)

    # Adjust layout
    total_RMSE = 0
    for i in range(num_poses):
        last_RMSE = np.sqrt(np.mean((poses[i, :3, 3] - gt_poses[i, :3, 3])**2)).item()
        total_RMSE += last_RMSE

    suptitle = 'Translation and Rotation vs Ground Truth'
    suptitle += f'\nRMSE (last): {total_RMSE:.2f} ({last_RMSE:.2f})'
    
    if len(poses) > 1:
        dx = tx[-1] - tx[-2]
        dx_gt = gt_tx[-1] - gt_tx[-2]
        dy = ty[-1] - ty[-2]
        dy_gt = gt_ty[-1] - gt_ty[-2]
        dpitch = pitches[-1] - gt_pitches[-2]
        dpitch_gt = gt_pitches[-1] - gt_pitches[-2]
        suptitle += f'\nDisplacement (gt): dx: {dx:.3f} ({dx_gt:.3f}), dy: {dy:.3f} ({dy_gt:.3f}), dpitch: {dpitch:.3f} ({dpitch_gt:.3f})'

    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save or display the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.close(fig)

def plot_trajectory_2d(poses, gt_poses, save_path=None):
    num_poses = len(poses)
    poses = np.array(poses)
    gt_poses = np.array(gt_poses)

    fig = plt.figure(figsize=(12, 6))

    # First subplot: XZ 2D view
    ax1 = fig.add_subplot(121)
    ax1.plot(poses[:, 0, 3], poses[:, 2, 3], 'b-', label='XZ')
    ax1.plot(gt_poses[:, 0, 3], gt_poses[:, 2, 3], 'r-', label='Ground Truth')
    
    # Mark the start and end points with bubbles
    ax1.scatter(poses[0,0,3], poses[0,2,3], color='blue', s=100, alpha=0.7, label='Start')
    ax1.scatter(gt_poses[0,0,3], gt_poses[0,2,3], color='red', s=100, alpha=0.3, label='gt: Start')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_title('XZ')
    ax1.legend()
    ax1.grid(True)
   
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
    ax2.plot(np.arange(num_poses), -gt_pitch, 'm--', label='g.t. -Pitch')

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
    plot_ground_truth_6dof(ground_truth)

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
        
def plot_ground_truth_6dof(ground_truth, save_path=results_dir / "ground_truth/6dof.png"):
    """
    Plots the robot's 6DoF trajectory components (x, y, z, roll, pitch, yaw) over time.

    Args:
        ground_truth (pd.DataFrame): DataFrame of shape (N, 12) containing rows of flattened 3x4 pose matrices.
        save_path (str, optional): Path to save the plot.

    Returns:
        None
    """
    num_poses = ground_truth.shape[0]

    # Extract translations
    tx = ground_truth.iloc[:, 1].values
    ty = ground_truth.iloc[:, 2].values
    tz = ground_truth.iloc[:, 3].values

    # Extract Euler angles (roll, pitch, yaw)
    rpy = np.empty((num_poses, 3))
    for i in range(num_poses):
        qx, qy, qz, qw = ground_truth.iloc[i, 4:8].values
        r = R.from_quat([qx, qy, qz, qw])
        rpy[i] = r.as_euler('zyz', degrees=True)

    rolls = np.unwrap(rpy[:, 0])
    pitches = np.unwrap(rpy[:, 1])
    yaws = np.unwrap(rpy[:, 2])

    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    components = [
        (tx, 'X Position (m)'),
        (ty, 'Y Position (m)'),
        (tz, 'Z Position (m)'),
        (rolls, 'Roll (deg)'),
        (pitches, 'Pitch (deg)'),
        (yaws, 'Yaw (deg)')
    ]

    for ax, (data_y, label) in zip(axes.flatten(), components):
        ax.plot(np.arange(num_poses), data_y)
        ax.set_xlabel('Frame Index')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True)

    fig.suptitle('Ground Truth: 6DoF Trajectory Components over Time')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

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
def plot_matches(q_frame, t_frame, save_path: str = None):
    img1 = q_frame.img
    kpts1 = q_frame.keypoints
    img2 = t_frame.img
    kpts2 = t_frame.keypoints
    matches = q_frame.get_matches(t_frame.id)

    if len(matches) > 50:
        matches = matches[:50]

    # Draw the matches on the images
    matched_image = cv2.drawMatches(img1, kpts1, img2, kpts2, matches, outImg=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Save the image with matched features
    if not save_path:
        save_path = results_dir / f"matches/" / f"{q_frame.id}_{t_frame.id}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, matched_image)