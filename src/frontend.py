from typing import List
import cv2
import numpy as np
from src.frame import Frame
from src.utils import isnan, transform_points


############################### Feature Matching ##########################################

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

def match_features(prev_frame: Frame, frame: Frame, K: np.ndarray, scale_computed=False, debug=False):
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
    # Get frame descriptors
    if scale_computed:
        # If the 3d points have been initialized with triangulation, we need to keep tracking the same features
        print(f"Matching landmarks from frame #{prev_frame.id} and features from frame #{frame.id}...")
    else:
        # If not, we use all the features
        print(f"Matching features between frames: {prev_frame.id} & {frame.id}...")

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # Match descriptors
    matches = bf.knnMatch(prev_frame.descriptors, frame.descriptors, k=2)

    # Filter matches with high dissimilarity
    matches = filter_matches(matches, debug)

    # Filter outlier matches
    matches = remove_outlier_matches(matches, prev_frame.keypoints, frame.keypoints, K, debug)
    prev_frame.set_matches(frame.id, matches)
    frame.set_matches(prev_frame.id, matches)
    print(f"\t{len(matches)} matches left!")

    # If the 3d points have been initialized, then the matched features are also landmarks
    if scale_computed:
        ## The features of the previous frame need to be updated, as some might not be present in the new frame
        prev_frame_landmark_indices = []
        frame_landmark_indices = []

        # Iterate over all matches
        for m in matches:
            # Extract the query and train keypoints
            q = m.queryIdx
            t = m.trainIdx

            # Check if the query is a previously found landmark
            if q in prev_frame.landmark_indices:
                # If it is, then it is also found in the new frame
                prev_frame_landmark_indices.append(q)
                frame_landmark_indices.append(t)

        print(f"\t{len(prev_frame_landmark_indices)} landmarks from frame #{prev_frame.id} detected in frame #{frame.id}!")
        prev_frame.set_landmark_indices(prev_frame_landmark_indices)
        frame.set_landmark_indices(frame_landmark_indices)

    return matches

def filter_matches(matches, debug=False):
    """Filter out matches using Lowe's Ratio Test"""
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if debug:
        print(f"\tLowe's Test filtered {len(matches) - len(good_matches)}/{len(matches)} matches!")
    return good_matches

def remove_outlier_matches(matches, prev_keypoints, keypoints, K, debug=False):
    # Extract the keypoint pixel coordinates
    prev_pixels = np.float32([prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    curr_pixels = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Find the homography matrix and mask using RANSAC
    # _, mask = cv2.findHomography(prev_pixels, curr_pixels, cv2.RANSAC,
    #                              ransacReprojThreshold=2.0, maxIters=5000, confidence=0.95)
    _, mask = cv2.findEssentialMat(prev_pixels, curr_pixels, K, cv2.RANSAC, prob=0.99, threshold=1.5)

    # Use the mask to filter inlier matches
    inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]

    if debug:
        print(f"\tRansac filtered {len(matches) - len(inlier_matches)}/{len(matches)} matches!")
    return inlier_matches

############################### Triangulation ##########################################

def initialize(prev_frame: Frame, frame: Frame, K: np.ndarray):
    """
    Initialize camera pose by estimating relative rotation and translation 
    between two frames, then triangulating 3D points.

    Args:
        prev_frame (Frame): The previous frame (contains keypoints, descriptors, etc.).
        frame (Frame): The current frame.
        K (np.ndarray): The camera intrinsic matrix (3x3).
        homography_inlier_ratio_threshold (float, optional): Threshold for deciding 
            whether to favor a homography over an essential matrix. Default is 0.45.
        essential_ransac_threshold (float, optional): RANSAC reprojection threshold 
            used in cv2.findEssentialMat. Default is 1.5.
        homography_ransac_threshold (float, optional): RANSAC reprojection threshold 
            used in cv2.findHomography. Default is 3.0.

    Returns:
        pose_inv (np.ndarray or None): The inverse of the [4x4] pose matrix if successful, 
            otherwise None.
        success (bool): True if initialization was successful, False otherwise.
    """
    print(f"Initializing pose by triangulating points between frames {prev_frame.id} & {frame.id}...")
    
    # ------------------------------------------------------------------------
    # 1. Get keypoint matches
    # ------------------------------------------------------------------------

    # Extract the matches between the previous and current frame
    matches = prev_frame.get_matches(frame.id)
    if len(matches) < 5:
        print("Not enough matches to compute the Essential Matrix!")
        return prev_frame.pose, False

    # Extract keypoint pixel coordinates and indices for both frames from the feature match
    prev_kpt_indices = np.array([m.queryIdx for m in matches])
    prev_kpt_pixels = np.float32([prev_frame.keypoints[idx].pt for idx in prev_kpt_indices])
    curr_kpt_indices = np.array([m.trainIdx for m in matches])
    curr_kpt_pixels = np.float32([frame.keypoints[idx].pt for idx in curr_kpt_indices])

    # ------------------------------------------------------------------------
    # 2. Compute Essential & Homography matrices
    # ------------------------------------------------------------------------

    # Compute the Essential Matrix
    E, mask_E = cv2.findEssentialMat(
        curr_kpt_pixels, prev_kpt_pixels, K, method=cv2.RANSAC, prob=0.99, threshold=1.5
    )
    mask_E = mask_E.ravel().astype(bool)

    # Compute the Homography Matrix
    H, mask_H = cv2.findHomography(
        curr_kpt_pixels, prev_kpt_pixels, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    mask_H = mask_H.ravel().astype(bool)

    # ------------------------------------------------------------------------
    # 3. Compute symmetric transfer errors & decide which model to use
    # ------------------------------------------------------------------------

    # Compute symmetric transfer error for Essential Matrix
    error_E, num_inliers_E = compute_symmetric_transfer_error(
        E, curr_kpt_pixels, prev_kpt_pixels, 'E', K=K
    )

    # Compute symmetric transfer error for Homography Matrix
    error_H, num_inliers_H = compute_symmetric_transfer_error(
        H, curr_kpt_pixels, prev_kpt_pixels, 'H', K=K
    )

    # Decide which matrix to use based on the ratio of inliers
    ratio = num_inliers_H / (num_inliers_E + num_inliers_H)
    print(f"\tInliers E: {num_inliers_E}, Inliers H: {num_inliers_H}, Ratio H/(E+H): {ratio}")
    use_homography = (ratio > 0.45)

    # ------------------------------------------------------------------------
    # 4. Recover pose (R, t) from Essential or Homography
    # ------------------------------------------------------------------------

    inlier_prev_pixels, inlier_curr_pixels = None, None  # Placeholder for the filtered pixels
    R, t = None, None                                    # Placeholder for the Rotation and Translation vectors

    # Check if we will use homography
    if not use_homography:
        # Filter points and indices
        inlier_prev_pixels = prev_kpt_pixels[mask_E]
        inlier_curr_pixels = curr_kpt_pixels[mask_E]
        prev_kpt_indices = prev_kpt_indices[mask_E]
        curr_kpt_indices = curr_kpt_indices[mask_E]

        # Decompose Essential Matrix
        points, R, t, mask_pose = cv2.recoverPose(E, inlier_prev_pixels, inlier_curr_pixels, K)

        # mask_pose indicates inliers used in cv2.recoverPose (1 for inliers, 0 for outliers)
        mask_pose = mask_pose.ravel().astype(bool)

        # Filter inlier points and indices
        inlier_prev_pixels = inlier_prev_pixels[mask_pose]
        inlier_curr_pixels = inlier_curr_pixels[mask_pose]
        prev_kpt_indices = prev_kpt_indices[mask_pose]
        curr_kpt_indices = curr_kpt_indices[mask_pose]
    else:
        # Filters inlier points and indices
        inlier_prev_pixels = prev_kpt_pixels[mask_H]
        inlier_curr_pixels = curr_kpt_pixels[mask_H]
        prev_kpt_indices = prev_kpt_indices[mask_H]
        curr_kpt_indices = curr_kpt_indices[mask_H]

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
            for j in range(len(inlier_curr_pixels)):
                p1_cam = np.linalg.inv(K) @ np.array([inlier_curr_pixels[j][0], inlier_curr_pixels[j][1], 1])
                p2_cam = np.linalg.inv(K) @ np.array([inlier_prev_pixels[j][0], inlier_prev_pixels[j][1], 1])

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

    # If we failed to recover R and t
    if R is None or t is None:
        print("[initialize] Failed to recover a valid pose from either E or H.")
        return None, False

    # ------------------------------------------------------------------------
    # 5. Build the 4x4 Pose matrix
    # ------------------------------------------------------------------------
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t.flatten()

    # Print the transformation
    yaw_deg = abs(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    print(f"\tTransformation: dx:{pose[0,3]:.3f}, dy:{pose[1,3]:.3f}, yaw: {yaw_deg:.3f}")

    # ------------------------------------------------------------------------
    # 6. Triangulate 3D points
    # ------------------------------------------------------------------------

    ## Compute relative Scale
    # Triangulate
    prev_points_3d = triangulate(inlier_prev_pixels, inlier_curr_pixels, R, t, K)
    if prev_points_3d is None or len(prev_points_3d) == 0:
        print("[initialize] Triangulation returned no 3D points.")
        return None, False

    # ------------------------------------------------------------------------
    # 7. Filter out points with small triangulation angles (cheirality check)
    # ------------------------------------------------------------------------

    large_angles_mask, prev_points_3d = filter_small_triangulation_angles(prev_points_3d, R, t)
    if large_angles_mask is None:
        return None, False

    # Calculate the 3d positions of the triangulated points in the current frame
    curr_points_3d = transform_points(prev_points_3d, pose)
    
    # Filter indices with small triangulation angles
    filtered_prev_kpt_indices = prev_kpt_indices[large_angles_mask]
    filtered_curr_kpt_indices = curr_kpt_indices[large_angles_mask]

    # ------------------------------------------------------------------------
    # 8. Store triangulated points back to frames
    # ------------------------------------------------------------------------
    
    # Restructure the 3d points to match the matches length
    # placing None at the features where a point was not triangulated
    prev_points = np.full((len(prev_frame.keypoints), 3), np.nan, dtype=np.float32)
    for i, idx in enumerate(filtered_prev_kpt_indices):
        prev_points[idx] = prev_points_3d[i]
    curr_points = np.full((len(frame.keypoints), 3), np.nan, dtype=np.float32)
    for i, idx in enumerate(filtered_curr_kpt_indices):
        curr_points[idx] = curr_points_3d[i]

    # Set the new triangulated points and their corresponding valid keypoint indices to each frame
    prev_frame.set_points(prev_points)
    frame.set_points(curr_points)

    # Return the initial pose and filtered points
    return np.linalg.inv(pose), True
      
def compute_symmetric_transfer_error(E_or_H, curr_kpt_pixels, prev_kpt_pixels, matrix_type='E', K=None):
    errors = []
    num_inliers = 0

    if matrix_type == 'E':
        F = np.linalg.inv(K.T) @ E_or_H @ np.linalg.inv(K)
    else:
        F = np.linalg.inv(K) @ E_or_H @ K

    for i in range(len(curr_kpt_pixels)):
        p1 = np.array([curr_kpt_pixels[i][0], curr_kpt_pixels[i][1], 1])
        p2 = np.array([prev_kpt_pixels[i][0], prev_kpt_pixels[i][1], 1])

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

def triangulate(prev_pixels, curr_pixels, R, t, K):
    # Compute projection matrices for triangulation
    M1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # First camera at origin
    M2 = K @ np.hstack((R, t))  # Second camera at R, t

    # Triangulate points
    prev_points_4d_hom = cv2.triangulatePoints(M1, M2, prev_pixels.T, curr_pixels.T)

    # Convert homogeneous coordinates to 3D
    points_3d = prev_points_4d_hom[:3] / prev_points_4d_hom[3]

    return points_3d.T # (N, 3)

def filter_small_triangulation_angles(points_3d, R, t, angle_threshold=.5, median_threshold=1.5):
    """
    Filter out 3D points that have a small triangulation angle between two camera centers.

    When points are triangulated from two views, points that lie almost directly on 
    the line connecting the two camera centers have a very small triangulation angle. 
    These points are often numerically unstable and can degrade the accuracy of the 
    reconstruction. This function discards such points by thresholding the angle to 
    at least 1 degree, and further checks that enough points remain and that the 
    median angle is at least 2 degrees.

    points_3d : (N, 3)
    R, t      : the rotation and translation from camera1 to camera2
    Returns:   valid_angles_mask (bool array of shape (N,)),
               filtered_points_3d (N_filtered, 3) or (None, None)
    """
    print("\t\tFiltering points with small triangulation angles...")

    # Camera centers in the coordinate system where camera1 is at the origin.
    # For convenience, flatten them to shape (3,).
    C1 = np.zeros(3)                 # First camera at origin
    C2 = (-R.T @ t).reshape(3)       # Second camera center in the same coords

    # Vectors from camera centers to each 3D point
    # points_3d is (N, 3), so the result is (N, 3)
    vec1 = points_3d - C1[None, :]   # shape: (N, 3)
    vec2 = points_3d - C2[None, :]   # shape: (N, 3)

    # Compute norms along axis=1 (per row)
    norms1 = np.linalg.norm(vec1, axis=1)  # shape: (N,)
    norms2 = np.linalg.norm(vec2, axis=1)  # shape: (N,)

    # Normalize vectors (element-wise division)
    vec1_norm = vec1 / norms1[:, None]     # shape: (N, 3)
    vec2_norm = vec2 / norms2[:, None]     # shape: (N, 3)

    # Compute dot product along axis=1 to get cos(theta)
    cos_angles = np.sum(vec1_norm * vec2_norm, axis=1)  # shape: (N,)

    # Clip to avoid numerical issues slightly outside [-1, 1]
    cos_angles = np.clip(cos_angles, -1.0, 1.0)

    # Convert to angles in degrees
    angles = np.degrees(np.arccos(cos_angles))  # shape: (N,)

    # Filter out points with triangulation angle < 1 degree
    valid_angles_mask = angles >= angle_threshold

    # Filter points and angles
    filtered_points_3d = points_3d[valid_angles_mask, :]  # shape: (N_filtered, 3)
    filtered_angles = angles[valid_angles_mask]
    print(f"\t\t{len(filtered_points_3d)} points left after filtering low triangulation angles.")

    # Check conditions to decide whether to discard
    num_remaining_points = filtered_points_3d.shape[0]
    median_angle = np.median(filtered_angles) if num_remaining_points > 0 else 0
    print(f"\t\tThe median angle is {median_angle:.3f} deg.")

    # If too few points or too small median angle, return None
    if num_remaining_points < 40 or median_angle < median_threshold:
        print("Discarding frame 2 due to insufficient triangulation quality.")
        return None, None

    return valid_angles_mask, filtered_points_3d # (N,) , (N, 3)

############################### Scale Estimation ##########################################

def compute_relative_scale(pre_prev_frame: Frame, prev_frame: Frame, frame: Frame):
    """Computes the relative scale between 2 frames"""
    # Get the common features between frames t-2, t-1, t
    print(f"Estimating scale using frames: {pre_prev_frame.id}, {prev_frame.id} & {frame.id}...")
    pre_prev_pair_indices, prev_pair_indices = get_common_match_indices(pre_prev_frame, prev_frame, frame)

    # If there are less than 2 common point matches, we cannot compute the scale
    if len(prev_pair_indices) < 2:
        return None, False
                 
    # Extract the 3D points of the previous frame
    pre_prev_frame_points = pre_prev_frame.points

    # Iterate over the found common point matches
    pre_prev_distances = []
    # Compute all the distances between common point 3D coordinates in the pre-prev frame
    for i, l1 in enumerate(pre_prev_pair_indices):
        # Extract the index and 3D point of the pre-prev frame on the common point
        p1 = pre_prev_frame_points[l1]
        if isnan(p1): continue

        # Extract the distance between that point and every other common point
        for l2 in pre_prev_pair_indices[i+1:]:
            p2 = pre_prev_frame_points[l2]
            if isnan(p2): continue
            pre_prev_distances.append(euclidean_distance(p1, p2))

    # Extract the 3D points of the previous frame
    prev_frame_points = prev_frame.points
    # Compute all the distances between common point 3D coordinates in the prev frame
    prev_distances = []
    for i, k1 in enumerate(prev_pair_indices):
        # Extract the index and 3D point of the prev frame on the common point
        p1 = prev_frame_points[k1]
        if isnan(p1): continue

        # Extract the distance between that point and every other common point
        for k2 in prev_pair_indices[i+1:]:
            p2 = prev_frame_points[k2]
            if isnan(p2): continue
            dist = np.max((euclidean_distance(p1, p2), 1e-6)) # Avoid division with 0!
            prev_distances.append(dist)

    # Calculate the median scale
    scales = [d1/d2 for (d1,d2) in zip(pre_prev_distances, prev_distances)]
    scale = np.median(scales)

    return scale, True

def get_common_match_indices(frame: Frame, frame1: Frame, frame2: Frame):
    """Given 3 consecutive frames, it returns the indices of the common features between all of them."""
    # Extract the matches between the frames -2 and -1
    f_f1_matches = frame.get_matches(frame1.id)

    # Extract the indices of the query keypoints from frame -1
    f_f1_query_indices = [m.queryIdx for m in f_f1_matches]
    f_f1_train_indices = [m.trainIdx for m in f_f1_matches]

    # Extract the matches between the frames -1 and 0
    f1_f2_matches = frame1.get_matches(frame2.id)

    # Extract the indices of the query keypoints from frame -1
    f1_f2_query_indices = [m.queryIdx for m in f1_f2_matches]
    f1_f2_train_indices = [m.trainIdx for m in f1_f2_matches]

    # Find the same matched points in matches [-2, -1] and [-1, 0]
    f_landmarks = []
    f1_landmarks = []
    f2_landmarks = []
    # Iterate over matches [-2, -1]
    for i in range(len(f_f1_train_indices)):
        f_f1_query_idx = f_f1_query_indices[i]
        f_f1_train_idx = f_f1_train_indices[i]
        # Iterate over matches [-1, 0]
        for j in range(len(f1_f2_query_indices)):
            f1_f2_query_idx = f1_f2_query_indices[j]
            f1_f2_train_idx = f1_f2_train_indices[j]
            # Check if the matches involve the same point of the frame -1
            if f_f1_train_idx == f1_f2_query_idx:
                f_landmarks.append(f_f1_query_idx)
                f1_landmarks.append(f_f1_train_idx)
                f2_landmarks.append(f1_f2_train_idx)
                break

    # Mark the common features as landmarks, so that they are tracked until they are no longer detected
    frame.set_landmark_indices(f_landmarks)
    frame1.set_landmark_indices(f1_landmarks)
    frame2.set_landmark_indices(f2_landmarks)

    return f_landmarks, f1_landmarks

def euclidean_distance(p1: np.ndarray, p2: np.ndarray):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
    
############################### Pose Estimation ##########################################

# Function to estimate the relative pose using solvePnP
def estimate_relative_pose(prev_frame: Frame, frame: Frame, K: np.ndarray, debug=False, dist_coeffs=None):
    """
    Estimate the relative pose between two frames using matched keypoints and depth information.

    solvePnP: Estimates the pose of a 3D object given a set of 3D points in the object coordinate space and their corresponding 2D projections in the image plane. 
    solvePnP Parameters:
        - matches (list): List of matched keypoints 
        - prev_frame: The previoud frame
        - frame: The current frame
        - K: Camera intrinsic matrix 
    solvePnP Returns:
        success: A boolean flag indicating if the function successfully found a solution.
        rvec: The is a Rodrigues rotation vector representing the rotation of the object in the camera coordinate system.
        tvec: The vector that represents the translation of the object in the camera coordinate system.

    Returns:
        - pose or None: The new pose as a 4x4 transformation matrix
    """
    print(f"Estimating relative pose using frames: {prev_frame.id} & {frame.id}...")
    pose = np.eye(4) # placeholder for displacement
    
    # Check if enough 3D points exist to estimate the camera pose using the Direct Linear Transformation (DLT) algorithm
    if len(prev_frame.landmark_points) < 6:
        print(f"\tWarning: Not enough points for pose estimation. Got {len(prev_frame.landmark_points)}, expected at least 6.")
        return None, None

    # Use solvePnP to estimate the pose
    success, rvec, tvec, inliers = cv2.solvePnPRansac(prev_frame.landmark_points, frame.landmark_pixels, K, dist_coeffs,
                                                      reprojectionError=3.0, confidence=0.99, iterationsCount=1000)
    if not success:
        return None, None

    # Compute reprojection error and print it
    error = compute_reprojection_error(prev_frame.landmark_points[inliers], frame.landmark_pixels[inliers], rvec, tvec, K, dist_coeffs)
    if debug:
        print(f"\tReprojection error: {error:.2f} pixels")

    # Refine pose using inliers by calling solvePnP again without RANSAC
    success_refined, rvec_refined, tvec_refined = cv2.solvePnP(prev_frame.landmark_points[inliers], frame.landmark_pixels[inliers], 
                                                                K, dist_coeffs, rvec, tvec, useExtrinsicGuess=True)
    if not success_refined:
        return None, None

    # Compute the refined reprojection error
    error_refined = compute_reprojection_error(prev_frame.landmark_points[inliers], frame.landmark_pixels[inliers], rvec_refined, tvec_refined, K, dist_coeffs)
    if debug:
        print(f"\tRefined reprojection error: {error_refined:.2f} pixels")

    # Convert the refined rotation vector to a rotation matrix
    R_refined, _ = cv2.Rodrigues(rvec_refined)

    # Construct the refined pose matrix
    pose[:3, :3] = R_refined
    pose[:3, 3] = tvec_refined.flatten()

    # Print the transformation
    yaw_deg = abs(np.degrees(np.arctan2(R_refined[1, 0], R_refined[0, 0])))
    print(f"\tTransformation: dx:{pose[0,3]:.3f}, dy:{pose[1,3]:.3f}, yaw: {yaw_deg:.3f}")

    # Set as 3D points in the new frame the 3D points of the previous frame, transformed
    frame_3d_points = np.full((len(frame.keypoints), 3), np.nan, dtype=np.float32)
    prev_frame_3d_points = prev_frame.landmark_points[inliers].reshape(-1,3)
    frame_3d_points[inliers.flatten(), :] = transform_points(prev_frame_3d_points, pose)
    frame.set_points(frame_3d_points)
    # Also update the indices of the tracked features
    frame_landmark_indices = [frame.landmark_indices[i] for i in inliers.flatten()]
    frame.set_landmark_indices(frame_landmark_indices)

    return pose, error_refined
    
def compute_reprojection_error(pts_3d, pts_2d, rvec, tvec, K, dist_coeffs):
    """Compute the reprojection error for the given 3D-2D point correspondences and pose."""
    # Project the 3D points to 2D using the estimated pose
    projected_pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist_coeffs)
    
    # Calculate the reprojection error
    error = np.sqrt(np.mean(np.sum(((pts_2d - projected_pts_2d).squeeze()) ** 2, axis=1)))
    
    return error

def is_significant_motion(P, t_threshold=0.3, yaw_threshold=1, debug=False):
    """ Determine if motion expressed by t, R is significant by comparing to tresholds. """
    R = P[:3, :3]
    t = P[:3, 3]

    dx = abs(t[0])
    dy = abs(t[1])
    yaw_deg = abs(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

    is_keyframe = dx > t_threshold or dy > t_threshold or yaw_deg > yaw_threshold
    if debug:
        print(f"\tDisplacement: dx: {dx:.3f}, dy:{dy:.3f}, yaw: {yaw_deg:.3f}")
        if is_keyframe:
            print("\t\tKeyframe!")
        else:
            print("\t\tNot a keyframe!")

    return is_keyframe
