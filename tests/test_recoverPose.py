import cv2
import numpy as np


def invert_transform(T):
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

def test_recover_pose():
    # --- Setup camera intrinsics ---
    camera_matrix = np.array([[707.0912,   0.    , 601.8873],
                              [  0.    , 707.0912, 183.1104],
                              [  0.    ,   0.    ,   1.    ]])

    # --- Generate synthetic 3D points ---
    # Generate 100 random points with Z=5 (X and Y uniformly in [-1, 1])
    num_points = 100
    pts_xy = np.random.uniform(-1, 1, (2, num_points))
    pts_z = 5 * np.ones((1, num_points))
    pts_3d = np.vstack((pts_xy, pts_z))  # shape: (3, N)

    # --- Projection matrices for two camera positions ---
    # First camera at origin (identity pose)
    P0 = camera_matrix @ np.eye(3,4)
    
    # Second camera: simulate a forward movement (translation along Z)
    # For this example, we use an identity rotation and a translation of 1 unit along Z.
    R_true = np.eye(3)
    t_true = np.array([[0], [0], [-1]], dtype=np.float64)
    P1 = camera_matrix @ np.hstack((R_true, t_true))
    
    # --- Project the 3D points into both camera images ---
    pts_3d_h = np.vstack((pts_3d, np.ones((1, pts_3d.shape[1]))))  # convert to homogeneous coordinates
    proj1 = P0 @ pts_3d_h
    proj2 = P1 @ pts_3d_h
    
    # Normalize to get pixel coordinates
    proj1 /= proj1[2, :]
    proj2 /= proj2[2, :]
    pts1 = proj1[:2, :].T
    pts2 = proj2[:2, :].T

    # --- Compute the Essential matrix ---
    E, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # --- Recover the pose from the Essential matrix ---
    inliers, R, t, mask = cv2.recoverPose(E, pts1, pts2, camera_matrix)
    print("Recovered rotation:\n", np.round(R, 3))
    print("Recovered translation:\n", np.round(t, 3))
    
    # Create a homogeneous transformation matrix from the recovered R and t
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()

    # Get the new camera pose in the world frame
    T_cam_world = invert_transform(T)
    print("New camera pose:\n", np.round(T_cam_world, 3))
    
    # --- Simulate updating the robot's pose ---
    # Assume the robot's initial pose is the identity (at the origin)
    robot_pose = np.eye(4)
    
    # Update the robot's pose (new_pose = old_pose * T)
    new_robot_pose = robot_pose @ np.linalg.inv(T)
    print("Updated robot pose:\n", np.round(new_robot_pose,3))

if __name__ == '__main__':
    test_recover_pose()
