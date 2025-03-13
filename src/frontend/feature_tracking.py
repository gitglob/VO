from typing import List
import cv2
import numpy as np
from src.frame import Frame


############################### Feature Matching ##########################################

def match_features(q_frame: Frame, t_frame: Frame, K: np.ndarray, debug=False):
    """
    Matches features between two frames.
    
    Each match has the following attributes:
        distance: The distance between the descriptors (a measure of similarity; lower is better).
        trainIdx: The index of the descriptor in the training set (second image).
        queryIdx: The index of the descriptor in the query set (first image).
        imgIdx: The index of the image (if multiple images are being used).
    """
    if debug:
        print(f"Matching features between frames: {q_frame.id} & {t_frame.id}...")

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # 1) Match descriptors (KNN)
    matches = bf.knnMatch(q_frame.descriptors, t_frame.descriptors, k=2)

    # 2) Filter matches with your custom filter (lowe ratio, distance threshold, etc.)
    filtered_matches = filter_matches(matches, debug)
    filtered_matches = remove_outlier_matches(filtered_matches, q_frame.keypoints, t_frame.keypoints, K, debug)

    if debug:
        print(f"\t{len(filtered_matches)} matches left!")

    if not filtered_matches:
        return []

    # 4) **Propagate keypoint IDs**  
    propagate_keypoints(t_frame, q_frame, filtered_matches)

    # 5) Store the matches in each t_frame
    q_frame.set_matches(t_frame.id, filtered_matches, "query")
    t_frame.set_matches(q_frame.id, filtered_matches, "train")

    return filtered_matches

def propagate_keypoints(t_frame: Frame, q_frame: Frame, matches: List[cv2.DMatch]):
    """Merges the keypoint identifiers for the matches features between query and train frames."""
    for m in matches:
        q_kp = q_frame.keypoints[m.queryIdx]
        t_kp = t_frame.keypoints[m.trainIdx]

        # If the train keypoint has no ID, copy from the query keypoint
        if t_kp.class_id < 0:  # or `t_kp.class_id is None`
            t_kp.class_id = q_kp.class_id

        # If the query keypoint has no ID, copy from the train keypoint
        elif q_kp.class_id <= 0:
            q_kp.class_id = t_kp.class_id

        # If both have IDs but they differ, pick a strategy (e.g., overwrite one)
        elif q_kp.class_id != t_kp.class_id:
            # Naive approach: unify by assigning query ID to train ID
            # or vice versa. Real SLAM systems often handle merges in a global map.
            t_kp.class_id = q_kp.class_id

def filter_matches(matches, debug=False):
    """Filter out matches using Lowe's Ratio Test"""
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if debug:
        print(f"\tLowe's Test filtered {len(matches) - len(good_matches)}/{len(matches)} matches!")
    return good_matches
   
def remove_outlier_matches(matches, q_kpts, t_kpts, K, debug=False):
    """Remove matches that don't match the Essential or Homography Matrix"""
    # Extract the keypoint pixel coordinates
    q_frame_kpt_pixels = np.float32([q_kpts[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    t_kpt_pixels = np.float32([t_kpts[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Compute the Essential Matrix
    E, mask_E = cv2.findEssentialMat(q_frame_kpt_pixels, t_kpt_pixels, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    mask_E = mask_E.ravel().astype(bool)

    # Compute the Homography Matrix
    H, mask_H = cv2.findHomography(q_frame_kpt_pixels, t_kpt_pixels, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    mask_H = mask_H.ravel().astype(bool)

    # Compute symmetric transfer error for Essential Matrix
    error_E, num_inliers_E = compute_symmetric_transfer_error(E, q_frame_kpt_pixels, t_kpt_pixels, 'E', K=K)

    # Compute symmetric transfer error for Homography Matrix
    error_H, num_inliers_H = compute_symmetric_transfer_error(H, q_frame_kpt_pixels, t_kpt_pixels, 'H', K=K)

    # Decide which matrix to use based on the ratio of inliers
    if debug:
        print(f"\tInliers E: {num_inliers_E}, Inliers H: {num_inliers_H}")
    if num_inliers_E == 0 and num_inliers_H == 0:
        print("\tAll keypoint pairs yield errors > threshold..")
        return []
    ratio = num_inliers_H / (num_inliers_E + num_inliers_H)
    if debug:
        print(f"\tRatio H/(E+H): {ratio}")
    use_homography = (ratio > 0.45)

    # Filter keypoints based on the chosen mask
    inlier_match_mask = mask_H if use_homography else mask_E

    # Use the mask to filter inlier matches
    inlier_matches = [matches[i] for i in range(len(matches)) if inlier_match_mask[i]]

    if debug:
        print(f"\tRansac filtered {len(matches) - len(inlier_matches)}/{len(matches)} matches!")

    return inlier_matches

def compute_symmetric_transfer_error(E_or_H, q_kpt_pixels, t_kpt_pixels, matrix_type='E', K=None, threshold=4.0):
    """
    Computes the symmetric transfer error for a set of corresponding keypoints using
    either an Essential matrix or a Homography matrix. This function is used to evaluate 
    how well the estimated transformation aligns the corresponding keypoints between two images.

    For each keypoint pair (one from the target image and one from the query image), 
    the function performs the following steps:
      - Computes the Fundamental matrix F from the provided Essential/Homography matrix and
        the camera intrinsic matrix K. For an Essential matrix, F is computed as:
            F = inv(K.T) @ E_or_H @ inv(K)
        For a Homography, it is computed as:
            F = inv(K) @ E_or_H @ K
      - Converts the keypoints to homogeneous coordinates.
      - Computes the corresponding epipolar lines (l1 in the target image and l2 in the query image).
      - Normalizes these lines using the Euclidean norm of their first two coefficients.
      - Calculates the perpendicular distances from each keypoint to its respective epipolar line.
      - Sums these distances to obtain a symmetric transfer error for the correspondence.

    The function then returns the mean error across all valid keypoint pairs and counts how many
    of these pairs have an error below the specified threshold (i.e., inliers).
    """
    errors = []
    num_inliers = 0

    if matrix_type == 'E':
        F = np.linalg.inv(K.T) @ E_or_H @ np.linalg.inv(K)
    else:
        F = np.linalg.inv(K) @ E_or_H @ K

    # Loop over paired keypoints
    for pt_target, pt_query in zip(t_kpt_pixels, q_kpt_pixels):
        # Convert to homogeneous coordinates
        p1 = np.array([pt_target[0], pt_target[1], 1.0])
        p2 = np.array([pt_query[0], pt_query[1], 1.0])

        # Compute the corresponding epipolar lines
        l2 = F @ p1    # Epipolar line in the query image corresponding to p1
        l1 = F.T @ p2  # Epipolar line in the target image corresponding to p2

        # Normalize the lines using the Euclidean norm of the first two coefficients
        norm_l1 = np.hypot(l1[0], l1[1])
        norm_l2 = np.hypot(l2[0], l2[1])
        if norm_l1 == 0 or norm_l2 == 0:
            continue
        l1_normalized = l1 / norm_l1
        l2_normalized = l2 / norm_l2

        # Compute perpendicular distances (absolute value of point-line dot product)
        d1 = abs(np.dot(p1, l1_normalized))
        d2 = abs(np.dot(p2, l2_normalized))
        error = d1 + d2

        errors.append(error)
        if error < threshold:
            num_inliers += 1

    mean_error = np.mean(errors) if errors else float('inf')
    return mean_error, num_inliers
