import numpy as np
import cv2
from pathlib import Path
import glob
import os
import matplotlib.pyplot as plt

# Dataset
main_dir = Path(__file__).parent.parent
data_dir = Path.home() / "Documents" / "data" / "kitti"
scene = "06"
image_dir = data_dir / scene / "image_0"
image_paths = glob.glob(os.path.join(image_dir, "*.png"))
ground_truth_txt = data_dir / "data_odometry_poses" / "dataset" / "poses" / (scene + ".txt")

# Camera intrinsic matrix
fx, fy = 707.0912, 707.0912
cx, cy = 601.8873, 183.1104
K = np.array([[fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]])

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

def filter_matches(matches):
    """Filter out matches using Lowe's Ratio Test."""
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def load_ground_truth(gt_file):
    """Load ground truth poses from a text file.
       Each line has 12 numbers representing a 3x4 transformation matrix.
    """
    gt_poses = []
    with open(gt_file, 'r') as f:
        for line in f:
            nums = list(map(float, line.split()))
            # Reshape the first 12 numbers into a 3x4 matrix.
            pose_3x4 = np.array(nums).reshape(3, 4)
            # Create a 4x4 homogeneous transformation matrix.
            T = np.eye(4)
            T[:3, :3] = pose_3x4[:, :3]
            T[:3, 3] = pose_3x4[:, 3]
            gt_poses.append(T)
    return gt_poses

def visualize(i, gt_poses, estimated_poses):
    # Extract (x, z) coordinates for plotting (KITTI: x is lateral, z is forward)
    est_traj = np.array([[pose[0, 3], pose[2, 3]] for pose in estimated_poses])
    gt_traj = np.array([[pose[0, 3], pose[2, 3]] for pose in gt_poses[:len(estimated_poses)]])
    
    # Plot the trajectories in 2 subplots: one for estimated, one for ground truth
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.plot(est_traj[:, 0], est_traj[:, 1], markersize=1, label="Estimated")
    ax1.set_title("Estimated Trajectory")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Z")
    ax1.legend()
    ax1.grid(True)

    ax1.scatter(est_traj[0, 0], est_traj[0, 1], color='green', s=100, label='Start')
    ax1.scatter(est_traj[-1, 0], est_traj[-1, 1], color='red', s=100, label='End')
    
    ax2.plot(gt_traj[:, 0], gt_traj[:, 1], markersize=1, label="Ground Truth")
    ax2.set_title("Ground Truth Trajectory")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax2.legend()
    ax2.grid(True)

    ax2.scatter(gt_traj[0, 0], gt_traj[0, 1], color='green', s=100, label='Start')
    ax2.scatter(gt_traj[-1, 0], gt_traj[-1, 1], color='red', s=100, label='End')
    
    plt.tight_layout()
    save_path = main_dir / "results" / "test" / f"{i}_traj2d.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def run_vo_and_plot(reprojection=True):
    # Load ground truth poses
    gt_poses = load_ground_truth(ground_truth_txt)
    
    # List and sort all image files in the directory
    image_files = sorted(image_dir.glob('*.png'))
    if not image_files:
        raise IOError("No images found in the image directory.")
    
    # Initialize the estimated global pose (4x4 identity)
    estimated_poses = [gt_poses[0]]
    
    # Initialize ORB detector and BFMatcher for Hamming distance
    orb = cv2.ORB_create(nfeatures=5000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # Process each consecutive image pair
    for i in range(1, len(image_files)):
        # Load consecutive frames in grayscale
        img1 = cv2.imread(str(image_files[i-1]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(image_files[i]), cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            print(f"Error loading images {image_files[i-1]} or {image_files[i]}. Skipping frame pair.")
            continue
        
        # Detect ORB keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        if des1 is None or des2 is None:
            print(f"Descriptors not found for frames {i-1} and {i}. Skipping.")
            continue
        
        # Match descriptors and sort them by distance
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = filter_matches(matches)
        
        # Extract matched keypoints
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        if len(pts1) < 5 or len(pts2) < 5:
            print(f"Not enough good matches between frames {i-1} and {i}.")
            continue
        
        # Compute the Essential Matrix using RANSAC
        E, mask_E = cv2.findEssentialMat(pts1, pts2, K, threshold=1.0, prob=0.999)
        if E is None:
            print(f"Essential matrix could not be computed between frames {i-1} and {i}.")
            continue
        mask_E = mask_E.ravel().astype(bool)
        pts1_inliers = pts1[mask_E]
        pts2_inliers = pts2[mask_E]

        # Recover pose (rotation and translation) from the Essential Matrix
        _, R, t, mask_pose = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)
        mask_pose = mask_pose.ravel().astype(bool)
        pts1_final = pts1_inliers[mask_pose]
        pts2_final = pts2_inliers[mask_pose]

        # Compute the scale using ground truth translation difference
        if i == 1:
            scale = np.linalg.norm(gt_poses[i][:3, 3] - gt_poses[i-1][:3, 3])

        # Construct the relative transformation matrix
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = (scale * t).flatten()
        T_rel_inv = invert_transform(T_rel)
        
        if reprojection:
            # Triangulation requires projection matrices for both frames.
            # First camera: P1 = K [I|0]
            # Second camera: P2 = K [R|t_scaled]
            t_scaled = scale * t
            P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
            P2 = K @ np.hstack((R, t_scaled))
            
            # Triangulate points; note that cv2.triangulatePoints expects 2xN arrays.
            pts1_for_tri = pts1_final.T  # shape (2, N)
            pts2_for_tri = pts2_final.T  # shape (2, N)
            pts4d_hom = cv2.triangulatePoints(P1, P2, pts1_for_tri, pts2_for_tri)
            pts3d = pts4d_hom[:3, :] / pts4d_hom[3, :]
            
            # Reproject the triangulated 3D points into both images.
            pts1_proj_hom = P1 @ np.vstack((pts3d, np.ones((1, pts3d.shape[1]))))
            pts1_proj = (pts1_proj_hom[:2, :] / pts1_proj_hom[2, :]).T  # shape (N,2)
            
            pts2_proj_hom = P2 @ np.vstack((pts3d, np.ones((1, pts3d.shape[1]))))
            pts2_proj = (pts2_proj_hom[:2, :] / pts2_proj_hom[2, :]).T  # shape (N,2)
            
            # Compute reprojection errors for both views.
            error1 = np.linalg.norm(pts1_final - pts1_proj, axis=1)
            error2 = np.linalg.norm(pts2_final - pts2_proj, axis=1)
            reproj_error = (error1 + error2) / 2.0
            
            # Keep only triangulated points with average reprojection error less than 1 pixel.
            valid_idx = reproj_error < 1.0
            pts1_valid = pts1_final[valid_idx]
            pts2_valid = pts2_final[valid_idx]
            if len(pts1_valid) != len(pts1_final):
                print(f"Reprojection filtered out {len(pts1_final) - len(pts1_valid)}/{len(pts1_final)} points.")
            
            if len(pts1_valid) < 5:
                print(f"Too few valid triangulated points after reprojection filtering between frames {i-1} and {i}.")
                continue
            
            # Re-run pose estimation on the filtered (valid) points.
            E_filtered, mask_filtered = cv2.findEssentialMat(pts1_valid, pts2_valid, K)
            if E_filtered is None:
                print(f"Filtered essential matrix could not be computed between frames {i-1} and {i}. Using previous estimation.")
                T_rel = np.eye(4)
                T_rel[:3, :3] = R
                T_rel[:3, 3] = (scale * t).flatten()
            else:
                _, R_filtered, t_filtered, mask_filtered_pose = cv2.recoverPose(E_filtered, pts1_valid, pts2_valid, K)
                T_rel = np.eye(4)
                T_rel[:3, :3] = R_filtered
                T_rel[:3, 3] = (scale * t_filtered).flatten()
        
        # Update the global pose estimate by chaining the transformation.
        new_pose = estimated_poses[-1] @ T_rel_inv
        estimated_poses.append(new_pose)

        visualize(i, gt_poses, estimated_poses)

    visualize(gt_poses, estimated_poses)

if __name__ == '__main__':
    run_vo_and_plot()
