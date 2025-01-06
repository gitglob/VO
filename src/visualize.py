import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


def plot_trajectory_components(poses, gt_poses, reproj_error, save_path=None, show_plot=False):
    poses = np.array(poses)
    gt_poses = np.array(gt_poses)
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))
    
    # Extract translations
    x_poses, y_poses, z_poses = poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3]
    gt_x_poses, gt_y_poses, gt_z_poses = gt_poses[:, 0, 3], gt_poses[:, 1, 3], gt_poses[:, 2, 3]
    
    # Extract rotations (in degrees)
    yaw_poses = np.degrees(np.arctan2(poses[..., 1, 0], poses[..., 0, 0]))
    pitch_poses = np.degrees(np.arctan2(-poses[..., 2, 0], np.sqrt(poses[..., 2, 1]**2 + poses[..., 2, 2]**2)))
    roll_poses = np.degrees(np.arctan2(poses[..., 2, 1], poses[..., 2, 2]))
    
    gt_yaw_poses = np.degrees(np.arctan2(gt_poses[..., 1, 0], gt_poses[..., 0, 0]))
    gt_pitch_poses = np.degrees(np.arctan2(-gt_poses[..., 2, 0], np.sqrt(gt_poses[..., 2, 1]**2 + gt_poses[..., 2, 2]**2)))
    gt_roll_poses = np.degrees(np.arctan2(gt_poses[..., 2, 1], gt_poses[..., 2, 2]))

    # Plot translations
    axs[0, 0].plot(np.arange(poses.shape[0]), x_poses, 'b-', label='X Pose')
    axs[0, 0].plot(np.arange(gt_poses.shape[0]), gt_x_poses, 'r-', label='Ground Truth X')
    axs[0, 0].set_title('X')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('m')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    axs[0, 1].plot(np.arange(poses.shape[0]), y_poses, 'b-', label='Y Pose')
    axs[0, 1].plot(np.arange(gt_poses.shape[0]), gt_y_poses, 'r-', label='Ground Truth Y')
    axs[0, 1].set_title('Y')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('m')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    axs[0, 2].plot(np.arange(poses.shape[0]), z_poses, 'b-', label='Z Pose')
    axs[0, 2].plot(np.arange(gt_poses.shape[0]), gt_z_poses, 'r-', label='Ground Truth Z')
    axs[0, 2].set_title('Z')
    axs[0, 2].set_xlabel('Time')
    axs[0, 2].set_ylabel('m')
    axs[0, 2].legend()
    axs[0, 2].grid(True)
    
    # Plot rotations (yaw, pitch, roll in degrees)
    axs[1, 0].plot(np.arange(poses.shape[0]), yaw_poses, 'b-', label='Yaw Pose')
    axs[1, 0].plot(np.arange(gt_poses.shape[0]), gt_yaw_poses, 'r-', label='Ground Truth Yaw')
    axs[1, 0].set_title('Yaw')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('deg')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(np.arange(poses.shape[0]), pitch_poses, 'b-', label='Pitch Pose')
    axs[1, 1].plot(np.arange(gt_poses.shape[0]), gt_pitch_poses, 'r-', label='Ground Truth Pitch')
    axs[1, 1].set_title('Pitch')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('deg')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    axs[1, 2].plot(np.arange(poses.shape[0]), roll_poses, 'b-', label='Roll Pose')
    axs[1, 2].plot(np.arange(gt_poses.shape[0]), gt_roll_poses, 'r-', label='Ground Truth Roll')
    axs[1, 2].set_title('Roll')
    axs[1, 2].set_xlabel('Time')
    axs[1, 2].set_ylabel('deg')
    axs[1, 2].legend()
    axs[1, 2].grid(True)

    # Adjust layout
    total_RMSE = 0
    for i in range(len(poses)):
        last_RMSE = np.sqrt(np.mean((poses[i, :3, 3] - gt_poses[i, :3, 3])**2)).item()
        total_RMSE += last_RMSE
    suptitle = 'Translation and Rotation vs Ground Truth' + f'\nReprojection Error: {reproj_error:.2f} pixels' + f'\nRMSE (last): {total_RMSE:.2f} ({last_RMSE:.2f})'
    if len(poses) > 1:
        dx = x_poses[-1] - x_poses[-2]
        dx_gt = gt_x_poses[-1] - gt_x_poses[-2]
        dy = y_poses[-1] - y_poses[-2]
        dy_gt = gt_y_poses[-1] - gt_y_poses[-2]
        dyaw = yaw_poses[-1] - yaw_poses[-2]
        dyaw_gt = gt_yaw_poses[-1] - gt_yaw_poses[-2]
        suptitle += f'\nDisplacement (gt): dx: {dx:.3f} ({dx_gt:.3f}), dy: {dy:.3f} ({dy_gt:.3f}), dyaw: {dyaw:.3f} ({dyaw_gt:.3f})'
    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save or display the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def plot_2d_trajectory(poses, gt_poses, ground_truth=True, save_path=None, show_plot=False, limits=False):
    poses = np.array(poses)
    gt_poses = np.array(gt_poses)
    
    fig = plt.figure(figsize=(12, 6))

    # First subplot: XY 2D view
    ax1 = fig.add_subplot(121)
    ax1.plot(poses[:, 0, 3], poses[:, 1, 3], 'b-', label='XY Trajectory')
    if ground_truth:
        ax1.plot(gt_poses[:, 0, 3], gt_poses[:, 1, 3], 'r-', label='Ground Truth')
    
    # Mark the start and end points with bubbles
    ax1.scatter(poses[0,0,3], poses[0,1,3], color='blue', s=100, alpha=0.7, label='Start')
    ax1.scatter(gt_poses[0,0,3], gt_poses[0,1,3], color='red', s=100, alpha=0.3, label='gt: Start')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('XY Trajectory')
    ax1.legend()
    ax1.grid(True)

    # Second subplot: Z 1D plot
    ax2 = fig.add_subplot(122)
    yaw = np.degrees(np.arctan2(poses[..., 1, 0], poses[..., 0, 0]))
    ax2.plot(np.arange(poses.shape[0]), yaw, 'b-', label='Yaw')
    if ground_truth:
        gt_yaw = np.degrees(np.arctan2(gt_poses[..., 1, 0], gt_poses[..., 0, 0]))
        ax2.plot(np.arange(gt_poses.shape[0]), gt_yaw, 'r-', label='Ground Truth')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Yaw')
    ax2.set_title('Yaw Trajectory')
    ax2.legend()
    ax2.grid(True)

    if limits:
        ax1.set_xlim([-2, 5]) 
        ax1.set_ylim([-2, 4])
        ax2.set_ylim([-0.1, 0.1])

    fig.suptitle('Map and Trajectory Views')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    if show_plot:
        plt.show(block=False)
        plt.pause(0.2)
        plt.close()
    else:
        plt.close(fig)

# Function to visualize the 3D map and trajectory
def plot_vo_trajectory(poses, save_path=None, show_plot=False):
    poses = np.array(poses)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the trajectory
    ax.plot(poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3], 'b-', label='Trajectory')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Map and Trajectory')
    ax.legend()
    
    ax.set_xlim([-2, 5]) 
    ax.set_ylim([-1, 6])
    ax.set_zlim([-0.1, 0.1])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    if show_plot:
        plt.show(block=False)
        plt.pause(0.2)
        plt.close()
    else:
        plt.close(fig)

# Function to plot keypoints
def plot_keypoints(image, keypoints, save_path, show_plot=False):
    # Draw keypoints on the image
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the image with matched features
    cv2.imwrite(save_path, img_with_keypoints)
    
    # Display the image if show_plot is True
    if show_plot:
        cv2.imshow('Keypoints', img_with_keypoints)
        cv2.waitKey(200)
        cv2.destroyAllWindows()

# Function to visualize the found feature matches
def plot_matches(img1, keypoints1, img2, keypoints2, matches, save_path, show_plot=False):
    if len(matches) > 20:
        matches_to_draw = matches[:20]
    else:
        matches_to_draw = matches
    
    # Draw the matches on the images
    matched_image = cv2.drawMatches(img1, keypoints1, 
                                    img2, keypoints2, 
                                    matches_to_draw, 
                                    None, 
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the image with matched features
    cv2.imwrite(save_path, matched_image)
    
    # Display the image if show_plot is True
    if show_plot:
        cv2.imshow('Feature Matches', matched_image)
        cv2.waitKey(200)
        cv2.destroyAllWindows()
        
def plot_ground_truth(ground_truth, save_path=None, show_plot=False, block=True):
    """ Reads the ground truth data from a file and plots the robot trajectory. """
    # Extract the positions (tx, ty, tz)
    tx = ground_truth["tx"].values
    ty = ground_truth["ty"].values
    tz = ground_truth["tz"].values

    # Plot the trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(tx, ty, tz, label='Robot Trajectory')
    
    # Mark the start and end points with bubbles
    ax.scatter(tx[0], ty[0], tz[0], color='green', s=100, label='Start')
    ax.scatter(tx[-1], ty[-1], tz[-1], color='red', s=100, label='End')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Robot Trajectory')
    ax.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    if show_plot:
        plt.show(block=block)
        plt.close()
    else:
        plt.close(fig)

def plot_ground_truth_2d(ground_truth, save_path=None, show_plot=False, block=True):
    """ 
    Plots the robot trajectory in 2D XY view and a separate Z plot. 
    """
    # Extract the positions (tx, ty, tz)
    tx = ground_truth["tx"].values
    ty = ground_truth["ty"].values
    tz = ground_truth["tz"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the XY trajectory
    ax1.plot(tx, ty, 'b-', label='XY Trajectory')
    ax1.scatter(tx[0], ty[0], color='green', s=100, label='Start')  # Start point
    ax1.scatter(tx[-1], ty[-1], color='red', s=100, label='End')    # End point
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    # ax1.set_xlim([-2, 5]) 
    # ax1.set_ylim([-2, 4])
    ax1.set_title('XY Trajectory')
    ax1.legend()
    ax1.grid(True)
    ax1.set_aspect('equal', 'box')

    # Plot the Z trajectory
    ax2.plot(np.arange(len(tz)), tz, 'r-', label='Z Trajectory')
    ax2.scatter(0, tz[0], color='green', s=100, label='Start')  # Start point
    ax2.scatter(len(tz) - 1, tz[-1], color='red', s=100, label='End')  # End point
    # ax2.set_ylim([-0.1, 0.1])
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Z')
    ax2.set_title('Z Trajectory')
    ax2.legend()
    ax2.grid(True)

    fig.suptitle('2D XY and Z Trajectory')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    if show_plot:
        plt.show(block=block)
        plt.close()
    else:
        plt.close(fig)
        