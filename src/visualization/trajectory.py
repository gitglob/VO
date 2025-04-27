import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from src.utils.utils import get_yaw
import src.globals as ctx

from config import results_dir
matplotlib.use('tkAgg')


def plot_trajectory(i: int, ba=True, save_path=results_dir / "trajectory"):
    save_path = save_path / f"{i}.png"

    poses = ctx.map.poses()
    gt_poses = ctx.map.ground_truth()

    num_poses = len(poses)

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

    # Add the BA poses 
    if ba is True:
        ba_poses = ctx.map.poses(ba=True)
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

def plot_trajectory_3d(save_path=results_dir / "vo" / "final_trajectory.png"):
    frames = ctx.map.keyframes

    poses = ctx.map.poses()
    gt_poses = ctx.map.ground_truth()

    frames = ctx.map.keyframes
    poses = np.array([f.pose for f in list(frames.values())])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the trajectory
    ax.plot(poses[:, 0, 3], poses[:, 2, 3], -poses[:, 1, 3], 'b-', label='Trajectory')
    ax.plot(gt_poses[:, 0, 3], gt_poses[:, 2, 3], 'r--', label='Ground Truth')
    
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

