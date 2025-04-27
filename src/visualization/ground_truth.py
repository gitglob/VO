import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.spatial.transform import Rotation as R

from src.utils.utils import get_yaw

from config import results_dir
matplotlib.use('tkAgg')



def plot_ground_truth(ground_truth):
    plot_ground_truth_2d(ground_truth)
    plot_ground_truth_3d(ground_truth)

def plot_ground_truth_3d(ground_truth, save_path=results_dir / "ground_truth/3d.png"):
    """ Reads the ground truth data from a file and plots the robot trajectory. """
    # Extract the positions (tx, ty, tz)
    tx = ground_truth.iloc[:, 3].values
    ty = ground_truth.iloc[:, 7].values
    tz = ground_truth.iloc[:, 11].values

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
    tx = ground_truth.iloc[:, 3].values
    tz = ground_truth.iloc[:, 11].values

    # Extract rotation
    pitch = np.empty((len(ground_truth), 3))
    for i in range(len(ground_truth)):
        qx, qy, qz, qw = ground_truth.iloc[i, 4:8].values
        Rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
        pitch[i] = get_yaw(Rot)

    pitch = np.unwrap(pitch[:,1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the XY trajectory
    ax1.plot(tx, tz, 'b-', label='XY Trajectory')
    ax1.scatter(tx[0], tz[0], color='green', s=100, label='Start')  # Start point
    ax1.scatter(tx[-1], tz[-1], color='red', s=100, label='End')    # End point
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
        
