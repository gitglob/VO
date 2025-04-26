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
        if m.distance < 0.9 * n.distance:
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
    E, mask_E = cv2.findEssentialMat(q_kpt_pixels, t_kpt_pixels, 
                                     cameraMatrix=K, method=cv2.RANSAC, 
                                     prob=0.999, threshold=2.0)
    mask_E = mask_E.ravel().astype(bool)

    ## Compute the Homography Matrix
    H, mask_H = cv2.findHomography(q_kpt_pixels, t_kpt_pixels, 
                                   method=cv2.RANSAC, 
                                   ransacReprojThreshold=2.0)
    mask_H = mask_H.ravel().astype(bool)

    # Compute symmetric transfer errors & decide which model to use
        
    ## Compute symmetric transfer error for Essential Matrix
    score_F = compute_symmetric_transfer_error(E, K, q_kpt_pixels, t_kpt_pixels, 'E')

    ## Compute symmetric transfer error for Homography Matrix
    score_H = compute_symmetric_transfer_error(H, K, q_kpt_pixels, t_kpt_pixels, 'H')
    
    ## Decide which matrix to use based on the ratio of inliers
    ratio_H = score_H / (score_H + score_F)
    use_homography = (ratio_H > 0.45)
    epipolar_constraint_mask = mask_H if use_homography else mask_E
    M = H if use_homography else E
    if debug:
        print(f"\t\t Ratio: {ratio_H:.2f}. Using {'Homography' if use_homography else 'Essential'} Matrix...")
        print(f"\t\t Epipolar Constraint filtered {sum(~epipolar_constraint_mask)}/{len(q_kpt_pixels)} matches!")

    return epipolar_constraint_mask, M, use_homography

def compute_symmetric_transfer_error(E_or_H, K, 
                                     q_kpt_pixels, t_kpt_pixels, 
                                     matrix_type='E', 
                                     T_H = 23.96, T_F = 15.36):
    """
    Computes the symmetric transfer error (in pixels) for a set of corresponding keypoints using
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
    score = 0

    if matrix_type == 'E':
        if K is None:
            raise ValueError("Camera intrinsic matrix K must be provided when using an essential matrix.")
        try:
            K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            raise ValueError("Provided homography is non-invertible.")
        
        # Compute fundamental matrix F from the essential matrix
        # The fundamental matrix maps pixels in one image to an epiline in the other image
        F = K_inv.T @ E_or_H @ K_inv
        
        for q_px, t_px in zip(q_kpt_pixels, t_kpt_pixels):
            # Convert pixel coordinates to homogeneous coordinates
            qp = np.array([q_px[0], q_px[1], 1.0])
            tp = np.array([t_px[0], t_px[1], 1.0])
            # Compute the epipolar lines in the other image
            # That means that the q_pixels will be found in the t)image on the t_line and the t_pixels on the q_image on the q_line
            # A line is in the form of ax + by + c, where [x,y] are the pixel coordinates
            q_line = F @ tp   # Line in query image corresponding to tp
            t_line = F.T @ qp # Line in train image corresponding to qp
            
            # Normalize the lines using the norm of the first two components
            q_line_norm = np.hypot(q_line[0], q_line[1])
            t_line_norm = np.hypot(t_line[0], t_line[1])
            if q_line_norm == 0 or t_line_norm == 0:
                continue
            q_line_normalized = q_line / q_line_norm
            t_line_normalized = t_line / t_line_norm

            # Compute the perpendicular distances from the points to the lines
            dq = abs(np.dot(qp, q_line_normalized))
            q_error = dq**2
            dt = abs(np.dot(tp, t_line_normalized))
            t_error = dt**2

            # Compute the score
            sq = T_F - q_error
            st = T_F - t_error
            score += sq + st
    elif matrix_type == 'H':
        # For homography, compute the symmetric reprojection error.
        # The Homography matrix in general maps points from the train image to the query image: p1 = H @ p2
        H = E_or_H  # H: mapping from target to query image
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            raise ValueError("Provided homography is non-invertible.")
        
        for q_px, t_px in zip(q_kpt_pixels, t_kpt_pixels):
            # Convert pixel coordinates to homogeneous coordinates
            qp = np.array([q_px[0], q_px[1], 1.0])
            tp = np.array([t_px[0], t_px[1], 1.0])
            
            # Map train points to the query image
            qp_est = H @ tp
            if np.isclose(qp_est[2], 0):
                continue
            qp_est = qp_est / qp_est[2]
            
            # Map query points to the train image
            tp_est = H_inv @ qp
            if np.isclose(tp_est[2], 0):
                continue
            tp_est = tp_est / tp_est[2]
            
            # Compute reprojection errors in both directions (Euclidean distances)
            dq = np.linalg.norm(tp - tp_est)
            q_error = dq**2
            dt = np.linalg.norm(qp - qp_est)
            t_error = dt**2

            # Compute the score
            sq = T_H - q_error
            st = T_H - t_error
            score += sq + st
    else:
        raise ValueError("matrix_type must be either 'E' (essential matrix) or 'H' (homography)")

    return score
    