import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

import src.globals as ctx

from config import results_dir
matplotlib.use('tkAgg')


def plot_BA():
    """
    Visualize the map points in 3D.
    """
    # Create a new figure and a 3D subplot
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the previous point positions in red
    prev_pts = ctx.map.point_positions(ba=False)
    ax.scatter(prev_pts[:, 0], prev_pts[:, 1], prev_pts[:, 2],
            facecolors='none', edgecolors='r', marker='o', label='Landmarks')
    
    # Plot the current point positions in blue
    pts = ctx.map.point_positions(ba=True)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
            c='b', marker='o', alpha=0.2, label='Optimized Landmarks')
    
    # Label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add a legend to differentiate the point clouds
    ax.legend()
    errors = pts - prev_pts
    errors_norm = np.linalg.norm(errors, axis=1)
    ax.set_title("Map Points <-> Error" + 
                    f"\nTotal: {np.sum(errors):.2f}" +
                    f", Mean: {np.mean(errors_norm):.2f}" +
                    f", Median: {np.median(errors_norm):.2f}" +
                    f"\nMin: {np.min(errors_norm):.2f}" +
                    f", Max: {np.max(errors_norm):.2f}")
    
    # Display the plot
    plt.show()

def plot_BA2d(save_path):
    """Visualize the map points in 3D."""

    # Create a new figure and a 3D subplot
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    
    # Plot the pre-BA point positions in red
    prev_pts = ctx.map.point_positions(ba=False)
    ax.scatter(prev_pts[:, 0], prev_pts[:, 1],
            facecolors='none', edgecolors='r', marker='o', label='Landmarks')
    
    # Plot the post-BA point positions in blue
    pts = ctx.map.point_positions(ba=True)
    ax.scatter(pts[:, 0], pts[:, 1],
            c='b', marker='o', alpha=0.2, label='Optimized Landmarks')
    
    # Label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Add a legend to differentiate the point clouds
    ax.legend()
    errors = pts - prev_pts
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
