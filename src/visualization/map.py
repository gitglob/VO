import os
import matplotlib.pyplot as plt
import matplotlib

import src.globals as ctx
from config import results_dir

matplotlib.use('tkAgg')


def plot_map():
    """
    Visualize the map points in 3D.
    """
    # Create a new figure and a 3D subplot
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the previous point positions in red
    positions = map.point_positions()
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
            facecolors='none', edgecolors='r', marker='o', label='Landmarks')
    
    # Label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title("Map Points")
    plt.show()

def plot_map2d(i, save_path=results_dir / "map"):
    """
    Visualize the map points in 3D.
    """
    save_path = save_path / f"{i}.png"

    # Create a new figure and a 3D subplot
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    
    # Plot the previous point positions in red
    positions = ctx.map.point_positions()
    ax.scatter(positions[:, 0], positions[:, 1],
            facecolors='none', edgecolors='r', marker='o', label='Landmarks')
    
    # Label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.set_title("Map Points")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.close(fig)
