import numpy as np
from src.others.frame import Frame
from src.others.utils import isnan
from config import debug


def validate_scale(scaled_poses: list[np.ndarray], gt_poses: list[np.ndarray]):
    """
    Validate that the estimated scaled poses are in the same scale as the ground truth poses.
    """
    pose_diff = np.linalg.norm(scaled_poses[-1][:3, 3] - scaled_poses[-2][:3, 3])
    gt_pose_diff = np.linalg.norm(gt_poses[-1][:3, 3] - gt_poses[-2][:3, 3])
    
    if np.diff([pose_diff, gt_pose_diff]) > 1e-3:
        raise ValueError("Estimated and ground truth poses are not in the same scale.")

def estimate_depth_scale(estimated_poses: list[np.ndarray], gt_poses: list[np.ndarray]):
    """
    Estimate the depth scale factor from two estimated poses using their ground truth counterparts.

    Args:
        estimated_poses (list of np.ndarray): List of two estimated poses (4x4).
        gt_poses (list of np.ndarray): List of two ground truth poses (4x4).

    Returns:
        float: Estimated scale factor.
    """
    if debug:
        print(f"\tEstimating depth scale ...")

    if len(estimated_poses) != 2 or len(gt_poses) != 2:
        raise ValueError("Both input lists must contain exactly two poses.")

    # Extract translations
    est_translation_diff = np.linalg.norm(estimated_poses[1][:3, 3] - estimated_poses[0][:3, 3])
    gt_translation_diff = np.linalg.norm(gt_poses[1][:3, 3] - gt_poses[0][:3, 3])

    if est_translation_diff == 0:
        raise ValueError("Estimated translation difference is zero, cannot estimate scale.")

    # Compute scale
    scale_factor = gt_translation_diff / est_translation_diff
    if debug:
        print(f"\t\tScale factor: {scale_factor:.2f}")

    return scale_factor


####################################################################################################

def compute_relative_scale(frame: Frame, prev_frame: Frame, pre_prev_frame: Frame):
    """Computes the relative scale between 2 frames"""
    # Get the common features between frames t-2, t-1, t
    print(f"Estimating scale using frames: {pre_prev_frame.id}, {prev_frame.id} & {frame.id}...")
    pre_prev_pair_indices, prev_pair_indices = get_common_match_indices(pre_prev_frame, prev_frame, frame)

    # If there are less than 2 common point matches, we cannot compute the scale
    if len(prev_pair_indices) < 2:
        return None, False
                 
    # Extract the 3D points of the previous frame
    pre_prev_frame_points = pre_prev_frame.match[prev_frame.id]["points"]

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
    prev_frame_points = prev_frame.match[frame.id]["points"]
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
    f_f1_matches = frame.match[frame1.id]["matches"]
    f_f1_matches = f_f1_matches[frame.match[frame1.id]["epipolar_constraint_mask"]]

    # Extract the indices of the query keypoints from frame -1
    f_f1_query_indices = [m.queryIdx for m in f_f1_matches]
    f_f1_train_indices = [m.trainIdx for m in f_f1_matches]

    # Extract the matches between the frames -1 and 0
    f1_f2_matches = frame1.match[frame2.id]["matches"]
    f1_f2_matches = f1_f2_matches[frame1.match[frame2.id]["epipolar_constraint_mask"]]

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

    return f_landmarks, f1_landmarks

def euclidean_distance(p1: np.ndarray, p2: np.ndarray):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)