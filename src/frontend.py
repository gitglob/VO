import cv2
import numpy as np
import open3d as o3d
from src.utils import keypoints_depth_to_3d_points


# Function to extract features using ORB
def extract_features(image):
    """
    Extract image features using ORB.
    
    keypoints: The detected keypoints. A 1-by-N structure array with the following fields:
        - pt: pixel coordinates of the keypoint [x,y]
        - size: diameter of the meaningful keypoint neighborhood
        - angle: computed orientation of the keypoint (-1 if not applicable); it's in [0,360) degrees and measured relative to image coordinate system (y-axis is directed downward), i.e in clockwise.
        - response: the response by which the most strong keypoints have been selected. Can be used for further sorting or subsampling.
        - octave: octave (pyramid layer) from which the keypoint has been extracted.
        - class_id: object class (if the keypoints need to be clustered by an object they belong to).
    descriptors: Computed descriptors. Descriptors are vectors that describe the image patch around each keypoint.
        Output concatenated vectors of descriptors. Each descriptor is a 32-element vector, as returned by cv.ORB.descriptorSize, 
        so the total size of descriptors will be numel(keypoints) * obj.descriptorSize(), i.e a matrix of size N-by-32 of class uint8, one row per keypoint.
    """
    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)
    
    return keypoints, descriptors

def match_features(frame, prev_frame, debug=False):
    """
    Matches features between two frames.
    
    Each match has the following attributes:
        distance: The distance between the descriptors (a measure of similarity; lower is better).
        trainIdx: The index of the descriptor in the training set (second image).
        queryIdx: The index of the descriptor in the query set (first image).
        imgIdx: The index of the image (if multiple images are being used).

    args:
        curr_desc: the descriptors of the current frame
        prev_desc: the descriptors of the previous frame
    """
    curr_desc = frame.descriptors
    prev_desc = prev_frame.descriptors

    # Create BFMatcher object
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # Match descriptors
    # matches = bf.match(curr_desc, prev_desc)
    matches = bf.knnMatch(curr_desc, prev_desc, k=2)

    # Filter matches with high dissimilarity
    # matches = filter_matches(matches, debug)

    # Filter outlier matches
    # matches = remove_outlier_matches(matches, frame.keypoints, prev_frame.keypoints, debug)
    
    if debug:
        print(f"{len(matches)} matches!")

    return [m for m, _ in matches]

def filter_matches(matches, debug=False):
    """Filter out matches using Lowe's Ratio Test"""
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if debug:
        print(f"Lowe's Test filtered {len(matches) - len(good_matches)}/{len(matches)} matches!")
    return good_matches

def remove_outlier_matches(matches, keypoints, prev_keypoints, debug=False):
    # Extract the keypoint pixel coordinates
    pixel_coords = np.float32([keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    prev_pixel_coords = np.float32([prev_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Find the homography matrix and mask using RANSAC
    H, mask = cv2.findHomography(pixel_coords, prev_pixel_coords, cv2.RANSAC,
                                 ransacReprojThreshold=2.0, maxIters=5000, confidence=0.95)

    # Use the mask to filter inlier matches
    inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]

    if debug:
        print(f"Ransac filtered {len(matches) - len(inlier_matches)}/{len(matches)} matches!")
    return inlier_matches

def is_significant_motion(P, t_threshold=0.3, yaw_threshold=1, debug=False):
    """ Determine if motion expressed by t, R is significant by comparing to tresholds. """
    R = P[:3, :3]
    t = P[:3, 3]

    dx = abs(t[0])
    dy = abs(t[1])
    yaw = abs(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

    is_keyframe = dx > t_threshold or dy > t_threshold or yaw > yaw_threshold
    if debug:
        print(f"Displacement: t: {dx:.3f}, {dy:.3f}, yaw: {yaw:.3f}")
        if is_keyframe:
            print("Keyframe!")
        else:
            print("Not a keyframe!")

    return is_keyframe

# Function to estimate the relative pose using solvePnP
def estimate_relative_pose(matches, cur_keypts, cur_depth, prev_keypts, prev_depth, K, debug=False):
    """
    Estimate the relative pose between two frames using matched keypoints and depth information.

    solvePnP: Estimates the pose of a 3D object given a set of 3D points in the object coordinate space and their corresponding 2D projections in the image plane. 
    solvePnP Parameters:
        - matches (list): List of matched keypoints 
        - cur_keypts (list): List of keypoints in the current frame 
        - prev_keypts (list): List of keypoints in the previous frame 
        - prev_depth (np.ndarray): Depth map of the previous frame
        - K (np.ndarray): Camera intrinsic matrix 
    solvePnP Returns:
        success: A boolean flag indicating if the function successfully found a solution.
        rvec: The is a Rodrigues rotation vector representing the rotation of the object in the camera coordinate system.
        tvec: The vector that represents the translation of the object in the camera coordinate system.

    Returns:
        - pose or None: The new pose as a 4x4 transformation matrix
    """
    pose = np.eye(4)  # Placeholder for the final transformation

    # Extract matched keypoints' coordinates
    cur_keypt_pixel_coords = np.float64([cur_keypts[m.queryIdx].pt for m in matches])
    prev_keypt_pixel_coords = np.float64([prev_keypts[m.trainIdx].pt for m in matches])

    # Convert the previous keypoints to 3D coordinates using the previous frame's depth map
    prev_pts_3d, _ = keypoints_depth_to_3d_points(prev_keypt_pixel_coords, prev_depth, 
                                                             cx=K[0, 2], cy=K[1, 2], 
                                                             fx=K[0, 0], fy=K[1, 1])

    # Convert the current keypoints to 3D coordinates using the current frame's depth map
    cur_pts_3d, _ = keypoints_depth_to_3d_points(cur_keypt_pixel_coords, cur_depth, 
                                                           cx=K[0, 2], cy=K[1, 2], 
                                                           fx=K[0, 0], fy=K[1, 1])

    # Ensure we have enough 3D points
    if len(prev_pts_3d) < 6 or len(cur_pts_3d) < 6:
        print(f"Warning: Not enough 3D points for pose estimation. Got {len(prev_pts_3d)} and {len(cur_pts_3d)}, expected at least 6.")
        return None, None

    # Convert the 3D points into Open3D point cloud format
    prev_pcd = o3d.geometry.PointCloud()
    cur_pcd = o3d.geometry.PointCloud()
    prev_pcd.points = o3d.utility.Vector3dVector(prev_pts_3d)
    cur_pcd.points = o3d.utility.Vector3dVector(cur_pts_3d)

    # Paint the source and target point clouds
    cur_pcd.paint_uniform_color([0, 0, 1])  # Blue for the current frame
    prev_pcd.paint_uniform_color([1, 0, 0])  # Red for the previous frame

    # Visualize the point clouds before alignment
    # o3d.visualization.draw_geometries([prev_pcd, cur_pcd], window_name="Before ICP", point_show_normal=False)

    # Compute normals for the point clouds
    prev_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    cur_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Apply ICP to estimate the transformation between the two point clouds
    threshold = 0.5  # distance threshold (can be adjusted)

    # Fine ICP on the full-resolution point clouds, using the initial result
    icp_result = o3d.pipelines.registration.registration_icp(
        source=prev_pcd,
        target=cur_pcd,
        max_correspondence_distance=threshold / 2,  # Use a smaller threshold for the fine step
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    # Check if ICP converged and return the transformation
    if icp_result.fitness > 0.0:
        if debug:
            print(f"ICP Fitness: {icp_result.fitness:.3f}, Inlier RMSE: {icp_result.inlier_rmse:.3f}")

            # Transform the current point cloud to align with the previous one
            cur_pcd.transform(icp_result.transformation)

            # Visualize the point clouds after alignment
            # o3d.visualization.draw_geometries([prev_pcd, cur_pcd], window_name="After ICP", point_show_normal=False)

        pose = icp_result.transformation
        return pose, icp_result.fitness
    else:
        print("ICP failed to converge")
        return None, None

def downsample_point_cloud(pcd, voxel_size):
    """
    Downsample a point cloud using a voxel grid filter.
    """
    return pcd.voxel_down_sample(voxel_size)
