from typing import List
import cv2
import numpy as np
from src.frame import Frame
from src.visualize import plot_matches

from config import results_dir, debug

############################### Feature Matching ##########################################

def match_features(q_frame: Frame, t_frame: Frame, K: np.ndarray, match_threshold=20):
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
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # 1) Match descriptors (KNN)
    matches = matcher.knnMatch(q_frame.descriptors, t_frame.descriptors, k=2)
    if len(matches) < match_threshold:
        return []

    # 2) Filter matches with your custom filter (lowe ratio, distance threshold, etc.)
    matches = filter_matches(matches, debug)
    if len(matches) < match_threshold:
        return []

    # Filter using epipolar constraint
    q_pxs = np.float64([q_frame.keypoints[m.queryIdx].pt for m in matches])
    t_pxs = np.float64([t_frame.keypoints[m.trainIdx].pt for m in matches])
    epipolar_constraint_mask, _, _ = enforce_epipolar_constraint(q_pxs, t_pxs, K)
    matches = np.array(matches, dtype=object)
    matches = matches[epipolar_constraint_mask]
    if len(matches) < match_threshold:
        return []

    # Save the matches
    if debug:
        match_save_path = results_dir / f"matches/" / f"{q_frame.id}_{t_frame.id}.png"
        plot_matches(matches, q_frame, t_frame, save_path=match_save_path)

    return matches

def filter_matches(matches, debug):
    """Filter out matches using Lowe's Ratio Test"""
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    if debug:
        print(f"\t Lowe's Test filtered {len(matches) - len(good_matches)}/{len(matches)} matches!")

    # Next, ensure uniqueness by keeping only the best match per train descriptor.
    unique_matches = {}
    for m in good_matches:
        # If this train descriptor is not seen yet, or if the current match is better, update.
        if m.trainIdx not in unique_matches or m.distance < unique_matches[m.trainIdx].distance:
            unique_matches[m.trainIdx] = m

    # Convert the dictionary values to a list of unique matches
    unique_matches = list(unique_matches.values())

    if debug:
        print(f"\t Uniqueness filtered {len(good_matches) - len(unique_matches)}/{len(good_matches)} matches!")

    return unique_matches
   
def enforce_epipolar_constraint(q_kpt_pixels, t_kpt_pixels, K):
    # Compute Essential & Homography matrices

    ## Compute the Essential Matrix
    E, mask_E = cv2.findEssentialMat(q_kpt_pixels, t_kpt_pixels, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    mask_E = mask_E.ravel().astype(bool)

    ## Compute the Homography Matrix
    H, mask_H = cv2.findHomography(q_kpt_pixels, t_kpt_pixels, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    mask_H = mask_H.ravel().astype(bool)

    # Compute symmetric transfer errors & decide which model to use

    ## Compute symmetric transfer error for Essential Matrix
    error_E, num_inliers_E = compute_symmetric_transfer_error(E, q_kpt_pixels, t_kpt_pixels, 'E', K=K)

    ## Compute symmetric transfer error for Homography Matrix
    error_H, num_inliers_H = compute_symmetric_transfer_error(H, q_kpt_pixels, t_kpt_pixels, 'H', K=K)
        
    if num_inliers_E == 0 and num_inliers_H == 0:
        print("0 Inliers. All keypoint pairs yield errors > threshold..")
        return None, None, None
    
    ## Decide which matrix to use based on the ratio of inliers
    ratio = num_inliers_H / (num_inliers_E + num_inliers_H)
    if debug:
        print(f"\t Inliers E/H: {num_inliers_E} / {num_inliers_H}. Ratio: {ratio}")

    use_homography = (ratio > 0.45)
    if debug:
        print(f"\t\t Using {'Homography' if use_homography else 'Essential'} Matrix...")

    epipolar_constraint_mask = mask_H if use_homography else mask_E
    if debug:
        print(f"\t\t Epipolar Constraint filtered {len(q_kpt_pixels) - epipolar_constraint_mask.sum()}/{len(q_kpt_pixels)} matches!")

    M = H if use_homography else E

    return epipolar_constraint_mask, M, use_homography

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
    