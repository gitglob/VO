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
    # orb = cv2.ORB_create(nfeatures=2000,   # maximum number of keypoints to be detected, default = 500
    #                      scaleFactor=1.02,  # lower means more keypoints, default = 1.2
    #                      nlevels=10,       # higher means more keypoints, default = 8
    #                      edgeThreshold=31, # lower means detecting more keypoints near the image borders, default = 31
    #                      firstLevel=1,     # higher means more focus on smaller features
    #                      WTA_K=4,          # higher increases the distinctiveness of features, default = 2
    #                      patchSize=11)     # higher means each keypoint captures more context, default = 31

    
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
        if m.distance < 0.75 * n.distance:
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
