import cv2
import numpy as np
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
    orb = cv2.ORB_create(nfeatures=5000)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)
    
    return keypoints, descriptors

def match_features(prev_frame, frame, debug=False):
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
    prev_desc = prev_frame.descriptors
    curr_desc = frame.descriptors

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # Match descriptors
    matches = bf.knnMatch(prev_desc, curr_desc, k=2)

    # Filter matches with high dissimilarity
    matches = filter_matches(matches, debug)

    # Filter outlier matches
    matches = remove_outlier_matches(matches, prev_frame.keypoints, frame.keypoints, debug)
    
    # print(f"Left with {len(matches)} matches!")
    return matches

def filter_matches(matches, debug=False):
    """Filter out matches using Lowe's Ratio Test"""
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if debug:
        print(f"Lowe's Test filtered {len(matches) - len(good_matches)}/{len(matches)} matches!")
    return good_matches

def remove_outlier_matches(matches, prev_keypoints, keypoints, debug=False):
    # Extract the keypoint pixel coordinates
    prev_pixel_coords = np.float32([prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pixel_coords = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Find the homography matrix and mask using RANSAC
    H, mask = cv2.findHomography(prev_pixel_coords, pixel_coords, cv2.RANSAC,
                                 ransacReprojThreshold=2.0, maxIters=5000, confidence=0.95)

    # Use the mask to filter inlier matches
    inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]

    if debug:
        print(f"Ransac filtered {len(matches) - len(inlier_matches)}/{len(matches)} matches!")
    return inlier_matches

def initialize(prev_frame, cur_frame, matches, K):
    if len(matches) < 5:
        print("Not enough matches to compute the Essential Matrix!")
        return prev_frame.pose, None, False

    # Extract locations of matched keypoints
    prev_kpt_pixel_coords = np.float32([prev_frame.keypoints[m.queryIdx].pt for m in matches])
    cur_kpt_pixel_coords = np.float32([cur_frame.keypoints[m.trainIdx].pt for m in matches])

    # Compute the Essential Matrix
    E, mask_E = cv2.findEssentialMat(
        cur_kpt_pixel_coords, prev_kpt_pixel_coords, K, method=cv2.RANSAC, prob=0.99, threshold=1.5
    )

    # Compute the Homography Matrix
    H, mask_H = cv2.findHomography(
        cur_kpt_pixel_coords, prev_kpt_pixel_coords, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )

    # Compute symmetric transfer error for Essential Matrix
    error_E, num_inliers_E = compute_symmetric_transfer_error(
        E, cur_kpt_pixel_coords, prev_kpt_pixel_coords, 'E', K=K
    )

    # Compute symmetric transfer error for Homography Matrix
    error_H, num_inliers_H = compute_symmetric_transfer_error(
        H, cur_kpt_pixel_coords, prev_kpt_pixel_coords, 'H', K=K
    )

    # Decide which matrix to use based on the ratio of inliers
    ratio = num_inliers_H / (num_inliers_E + num_inliers_H)
    print(f"Inliers E: {num_inliers_E}, Inliers H: {num_inliers_H}, Ratio H/(E+H): {ratio}")

    if ratio > 0.45:
        use_homography = True
    else:
        use_homography = False

    if not use_homography:
        # Decompose Essential Matrix
        points, R, t, mask_pose = cv2.recoverPose(E, cur_kpt_pixel_coords, prev_kpt_pixel_coords, K)

        # mask_pose indicates inliers used in cv2.recoverPose (1 for inliers, 0 for outliers)
        mask_pose = mask_pose.ravel().astype(bool)

        # Extract inlier points
        inlier_prev_pts = prev_kpt_pixel_coords[mask_pose]
        inlier_curr_pts = cur_kpt_pixel_coords[mask_pose]
    else:
        # Use inliers from the initial findHomography
        mask_H = mask_H.ravel().astype(bool)

        # Extract inlier points
        inlier_prev_pts = prev_kpt_pixel_coords[mask_H]
        inlier_curr_pts = cur_kpt_pixel_coords[mask_H]

        # Decompose Homography Matrix
        num_solutions, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)

        # Select the best solution based on criteria
        best_solution = None
        max_front_points = 0
        best_alignment = -1
        desired_normal = np.array([0, 0, 1])

        for i in range(num_solutions):
            R_candidate = Rs[i]
            t_candidate = Ts[i]
            n_candidate = Ns[i]

            # Check if points are in front of camera
            front_points = 0
            alignment = np.dot(n_candidate.flatten(), desired_normal)
            for j in range(len(inlier_curr_pts)):
                p1_cam = np.linalg.inv(K) @ np.array([inlier_curr_pts[j][0], inlier_curr_pts[j][1], 1])
                p2_cam = np.linalg.inv(K) @ np.array([inlier_prev_pts[j][0], inlier_prev_pts[j][1], 1])

                depth1 = np.dot(n_candidate.T, p1_cam) / np.dot(n_candidate.T, R_candidate @ p1_cam + t_candidate)
                depth2 = np.dot(n_candidate.T, p2_cam)

                if depth1 > 0 and depth2 > 0:
                    front_points += 1

            if front_points > max_front_points and alignment > best_alignment:
                max_front_points = front_points
                best_alignment = alignment
                best_solution = i

        # Use the best solution
        R = Rs[best_solution]
        t = Ts[best_solution]

    # Construct the initial pose transformation matrix
    initial_pose = np.eye(4)
    initial_pose[:3, :3] = R
    initial_pose[:3, 3] = t.flatten()

    # Triangulate
    points_3d = triangulate(inlier_prev_pts, inlier_curr_pts, R, t, K)

    # Filter points with small triangulation angles
    valid_indices = filter_small_triangulation_angles(points_3d, R, t)

    # If initialization is successful, return the initial pose and filtered points
    if valid_indices is not None:
        return initial_pose, points_3d[:, valid_indices], True
    else:
        return None, None, False

def triangulate(inlier_prev_pts, inlier_curr_pts, R, t, K):
    # Compute projection matrices for triangulation
    M1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # First camera at origin
    M2 = K @ np.hstack((R, t))  # Second camera at R, t

    # Prepare points for triangulation (must be in homogeneous coordinates)
    # Transpose the points to match OpenCV's expectations (2 x N arrays)
    inlier_prev_pts_hom = inlier_prev_pts.T  # Shape: 2 x N
    inlier_curr_pts_hom = inlier_curr_pts.T  # Shape: 2 x N

    # Triangulate points
    points_4d_hom = cv2.triangulatePoints(M1, M2, inlier_prev_pts_hom, inlier_curr_pts_hom)

    # Convert homogeneous coordinates to 3D
    points_3d = points_4d_hom[:3] / points_4d_hom[3]

    return points_3d

def filter_small_triangulation_angles(points_3d, R, t):
    # Compute triangulation angles for each point
    # Camera centers
    C1 = np.zeros((3, 1))  # First camera at origin
    C2 = -R.T @ t  # Second camera center in world coordinates

    # Vectors from camera centers to points
    vec1 = points_3d - C1  # Shape: 3 x N
    vec2 = points_3d - C2  # Shape: 3 x N

    # Normalize vectors
    vec1_norm = vec1 / np.linalg.norm(vec1, axis=0)
    vec2_norm = vec2 / np.linalg.norm(vec2, axis=0)

    # Compute cosine of angles between vectors
    cos_angles = np.sum(vec1_norm * vec2_norm, axis=0)

    # Ensure values are within valid range [-1, 1] due to numerical errors
    cos_angles = np.clip(cos_angles, -1.0, 1.0)

    # Compute angles in degrees
    angles = np.arccos(cos_angles) * (180.0 / np.pi)  # Shape: N

    # Filter points with triangulation angle less than 1 degree
    valid_indices = angles >= 1.0

    # Filter points and angles
    filtered_points_3d = points_3d[:, valid_indices]
    filtered_angles = angles[valid_indices]

    # Check conditions to decide whether to discard frame 2
    num_remaining_points = filtered_points_3d.shape[1]
    median_angle = np.median(filtered_angles)

    # Check if triangulation failed
    if num_remaining_points < 40 or median_angle < 2.0:
        print("Discarding frame 2 due to insufficient triangulation quality.")
        return None
    
    return valid_indices
        
def compute_symmetric_transfer_error(E_or_H, cur_kpt_pixel_coords, prev_kpt_pixel_coords, matrix_type='E', K=None):
    errors = []
    num_inliers = 0

    if matrix_type == 'E':
        F = np.linalg.inv(K.T) @ E_or_H @ np.linalg.inv(K)
    else:
        F = np.linalg.inv(K) @ E_or_H @ K

    for i in range(len(cur_kpt_pixel_coords)):
        p1 = np.array([cur_kpt_pixel_coords[i][0], cur_kpt_pixel_coords[i][1], 1])
        p2 = np.array([prev_kpt_pixel_coords[i][0], prev_kpt_pixel_coords[i][1], 1])

        # Epipolar lines
        l2 = F @ p1
        l1 = F.T @ p2

        # Normalize the lines
        l1 /= np.sqrt(l1[0]**2 + l1[1]**2)
        l2 /= np.sqrt(l2[0]**2 + l2[1]**2)

        # Distances
        d1 = abs(p1 @ l1)
        d2 = abs(p2 @ l2)

        error = d1 + d2
        errors.append(error)

        if error < 4.0:  # Threshold for inliers (you may adjust this value)
            num_inliers += 1

    mean_error = np.mean(errors)
    return mean_error, num_inliers

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
def estimate_relative_pose(matches, prev_keypts, prev_depth, cur_keypts, K, dist_coeffs=None, debug=False):
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
    pose = np.eye(4) # placeholder for displacement

    # Extract matched keypoints' coordinates
    prev_keypt_pixel_coords = np.float64([prev_keypts[m.queryIdx].pt for m in matches])
    cur_keypt_pixel_coords = np.float64([cur_keypts[m.trainIdx].pt for m in matches])

    # Convert the keypoints to 3D coordinates using the depth map
    prev_pts_3d, indices = keypoints_depth_to_3d_points(prev_keypt_pixel_coords, prev_depth, 
                                                        cx=K[0, 2], cy=K[1, 2], 
                                                        fx=K[0, 0], fy=K[1, 1])
    
    # Check if enough 3D points exist to estimate the camera pose using the Direct Linear Transformation (DLT) algorithm
    if len(prev_pts_3d) < 6:
        print(f"Warning: Not enough points for pose estimation. Got {len(prev_pts_3d)}, expected at least 6.")
        return None, None
    
    # Use solvePnP to estimate the pose
    success, rvec, tvec, inliers = cv2.solvePnPRansac(prev_pts_3d, cur_keypt_pixel_coords[indices], 
                                                      cameraMatrix=K, distCoeffs=dist_coeffs, 
                                                      reprojectionError=0.2, confidence=0.999, 
                                                      iterationsCount=5000)

    # Compute reprojection error and print it
    error = compute_reprojection_error(prev_pts_3d[inliers], cur_keypt_pixel_coords[indices][inliers], rvec, tvec, K, dist_coeffs)
    if debug:
        print(f"Reprojection error: {error:.2f} pixels")

    if success:
        # Refine pose using inliers by calling solvePnP again without RANSAC
        success_refine, rvec_refined, tvec_refined = cv2.solvePnP(prev_pts_3d[inliers], cur_keypt_pixel_coords[indices][inliers], 
                                                                  K, dist_coeffs, rvec, tvec, useExtrinsicGuess=True)

        if success_refine:
            # Compute the refined reprojection error
            error_refined = compute_reprojection_error(prev_pts_3d[inliers], cur_keypt_pixel_coords[indices][inliers], rvec_refined, tvec_refined, K, dist_coeffs)
            if debug:
                print(f"Refined reprojection error: {error_refined:.2f} pixels")

            # Convert the refined rotation vector to a rotation matrix
            R_refined, _ = cv2.Rodrigues(rvec_refined)

            # Construct the refined pose matrix
            pose[:3, :3] = R_refined
            pose[:3, 3] = tvec_refined.flatten()

            return np.linalg.inv(pose), error_refined
        else:
            return None, None
    else:
        return None, None
    
def compute_reprojection_error(pts_3d, pts_2d, rvec, tvec, K, dist_coeffs=None):
    """Compute the reprojection error for the given 3D-2D point correspondences and pose."""
    # Project the 3D points to 2D using the estimated pose
    projected_pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist_coeffs)
    
    # Calculate the reprojection error
    error = np.sqrt(np.mean(np.sum(((pts_2d - projected_pts_2d).squeeze()) ** 2, axis=1)))
    
    return error
