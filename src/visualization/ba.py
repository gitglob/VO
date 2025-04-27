import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from config import results_dir
matplotlib.use('tkAgg')


def plot_BA(prev_point_positions, point_positions):
    """
    Visualize the map points in 3D.
    """
    # Create a new figure and a 3D subplot
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the previous point positions in red
    ax.scatter(prev_point_positions[:, 0],
            prev_point_positions[:, 1],
            prev_point_positions[:, 2],
            facecolors='none', edgecolors='r', marker='o', label='Landmarks')
    
    # Plot the current point positions in blue
    ax.scatter(point_positions[:, 0],
            point_positions[:, 1],
            point_positions[:, 2],
            c='b', marker='o', alpha=0.2, label='Optimized Landmarks')
    
    # Label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add a legend to differentiate the point clouds
    ax.legend()
    errors = point_positions - prev_point_positions
    errors_norm = np.linalg.norm(errors, axis=1)
    ax.set_title("Map Points <-> Error" + 
                    f"\nTotal: {np.sum(errors):.2f}" +
                    f", Mean: {np.mean(errors_norm):.2f}" +
                    f", Median: {np.median(errors_norm):.2f}" +
                    f"\nMin: {np.min(errors_norm):.2f}" +
                    f", Max: {np.max(errors_norm):.2f}")
    
    # Display the plot
    plt.show()

def plot_BA2d(prev_point_positions, point_positions, i, save_path=results_dir / "ba"):
    """
    Visualize the map points in 3D.
    """
    save_path = save_path / f"{i}.png"

    # Create a new figure and a 3D subplot
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    
    # Plot the previous point positions in red
    ax.scatter(prev_point_positions[:, 0],
            prev_point_positions[:, 1],
            facecolors='none', edgecolors='r', marker='o', label='Landmarks')
    
    # Plot the current point positions in blue
    ax.scatter(point_positions[:, 0],
            point_positions[:, 1],
            c='b', marker='o', alpha=0.2, label='Optimized Landmarks')
    
    # Label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Add a legend to differentiate the point clouds
    ax.legend()
    errors = point_positions - prev_point_positions
    errors_norm = np.linalg.norm(errors, axis=1)
    ax.set_title("Map Points <-> Error" + 
                    f"\nTotal: {np.sum(errors):.2f}" +
                    f", Mean: {np.mean(errors_norm):.2f}" +
                    f", Median: {np.median(errors_norm):.2f}" +
                    f"\nMin: {np.min(errors_norm):.2f}" +
                    f", Max: {np.max(errors_norm):.2f}")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.close(fig)
