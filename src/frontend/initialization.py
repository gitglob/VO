import numpy as np
import cv2
from src.others.frame import Frame
from src.others.utils import invert_transform, transform_points
from src.others.visualize import plot_matches
from src.others.filtering import enforce_epipolar_constraint, filter_by_reprojection, filter_triangulation_points

from config import results_dir, SETTINGS, K


debug = SETTINGS["generic"]["debug"]
MIN_NUM_TRIANG_POINTS = SETTINGS["triangulation"]["min_num_init_points"]


def initialize_pose(q_frame: Frame, t_frame: Frame):
    """
    Initializes the camera pose by estimating the relative rotation and translation 
    between two consecutive frames using feature matches.

    This function computes the Essential and Homography matrices to determine 
    the best motion model. It then recovers the relative pose (rotation and 
    translation) using the Essential matrix if the motion is mostly translational 
    or the Homography matrix if the scene is planar. Finally, the pose is used 
    to initialize the frames and triangulate 3D points.

    Args:
        q_frame (Frame): The previous frame.
        t_frame (Frame): The current frame.
        K (np.ndarray): The camera intrinsic matrix (3x3).
        debug (bool, optional): If True, saves and visualizes matches.

    Returns:
        Tuple[np.ndarray or None, bool]: 
            - The inverse 4x4 transformation matrix (q_frame -> frame) if successful, otherwise None.
            - A boolean indicating whether the initialization was successful.
    """
    if debug:
        print(f"Initializing the camera pose using frames {q_frame.id} & {t_frame.id}...")
    
    # ------------------------------------------------------------------------
    # 1. Get keypoint matches
    # ------------------------------------------------------------------------

    # Extract the matches between the previous and current frame
    matches = q_frame.get_matches(t_frame.id)

    # Extract keypoint pixel coordinates and indices for both frames from the feature match
    q_kpt_pixels = np.float64([q_frame.keypoints[m.queryIdx].pt for m in matches])
    t_kpt_pixels = np.float64([t_frame.keypoints[m.trainIdx].pt for m in matches])

    # ------------------------------------------------------------------------
    # 2. Enforce Epipolar Constraint
    # ------------------------------------------------------------------------

    epipolar_constraint_mask, M, use_homography = enforce_epipolar_constraint(q_kpt_pixels, t_kpt_pixels)
    if epipolar_constraint_mask is None:
        print("[initialize] Failed to apply epipolar constraint..")
        return None, False

    # Save the matches
    if debug:
        match_save_path = results_dir / f"matches/initialization/1-epipolar_constraint" / f"{q_frame.id}_{t_frame.id}a.png"
        plot_matches(matches[~epipolar_constraint_mask], q_frame, t_frame, save_path=match_save_path)
        match_save_path = results_dir / f"matches/initialization/1-epipolar_constraint" / f"{q_frame.id}_{t_frame.id}b.png"
        plot_matches(matches[epipolar_constraint_mask], q_frame, t_frame, save_path=match_save_path)
    matches = matches[epipolar_constraint_mask]
    inlier_q_pixels = q_kpt_pixels[epipolar_constraint_mask]
    inlier_t_pixels = t_kpt_pixels[epipolar_constraint_mask]

    # ------------------------------------------------------------------------
    # 3. Recover pose (R, t) from Essential or Homography
    # ------------------------------------------------------------------------

    # Check if we will use homography
    R, t, reproj_mask = None, None, None
    if not use_homography:
        # Decompose Essential Matrix
        _, R_est, t_est, mask_pose = cv2.recoverPose(M, inlier_q_pixels, inlier_t_pixels)

        # mask_pose indicates inliers used in cv2.recoverPose (1 for inliers, 0 for outliers)
        mask_pose = mask_pose.ravel().astype(bool)
        if debug:
            print(f"\t\t Pose Recovery filtered {epipolar_constraint_mask.sum() - mask_pose.sum()}/{epipolar_constraint_mask.sum()} matches!")

        if R_est is not None and t_est is not None and np.any(mask_pose):
            R, t = R_est, t_est
            matches = matches[mask_pose]

        # Reprojection filter
        reproj_mask = filter_by_reprojection(
            matches, q_frame, t_frame,
            R, t,
            save_path=results_dir / f"matches/initialization/2-reprojection"
        )
    else:
        # Decompose Homography Matrix
        num_solutions, Rs, Ts, Ns = cv2.decomposeHomographyMat(M)

        # Select the best solution based on criteria
        best_solution = None
        max_front_points = 0
        best_alignment = -1
        desired_normal = np.array([[0, 0, 1]])

        for i in range(num_solutions):
            R_candidate = Rs[i]
            t_candidate = Ts[i]
            n_candidate = Ns[i]

            # Check if the normal aligns with the 'upward' direction (optional criterion)
            alignment = np.dot(n_candidate, desired_normal)

            # Check if points are in front of camera
            front_points = 0
            invK = np.linalg.inv(K)
            for j in range(len(inlier_q_pixels)):
                # Current frame pixel in camera coords
                p_curr_cam = invK @ np.array([*inlier_q_pixels[j], 1.0])  
                # Previous frame pixel in camera coords
                p_prev_cam = invK @ np.array([*inlier_t_pixels[j], 1.0])

                # Depth for current pixel after transformation
                denom = np.dot(n_candidate, R_candidate @ p_curr_cam + t_candidate)
                depth_curr = np.dot(n_candidate, p_curr_cam) / (denom + 1e-12)  # small eps for safety
                
                # Depth for previous pixel (just dot product since it's reference plane)
                depth_prev = np.dot(n_candidate, p_prev_cam)

                if depth_prev > 0 and depth_curr > 0:
                    front_points += 1

            # Update best solution if it meets criteria
            if front_points > max_front_points and alignment > best_alignment:
                max_front_points = front_points
                best_alignment = alignment
                best_solution = i

        # Use the best solution
        R = Rs[best_solution]
        t = Ts[best_solution]

        # Reprojection filter
        reproj_mask = filter_by_reprojection(
            matches,
            q_frame, t_frame,
            R, t,
            save_path=results_dir / f"matches/initialization/2-reprojection/"
        )

    # If we failed to recover R and t
    if R is None or t is None:
        print("[initialize] Failed to recover a valid pose from either E or H.")
        return None, False
            
    # Save the matches
    if debug:
        match_save_path = results_dir / "matches/initialization/3-reprojection" / f"{q_frame.id}_{t_frame.id}a.png"
        plot_matches(matches[~reproj_mask], q_frame, t_frame, save_path=match_save_path)
        match_save_path = results_dir / "matches/initialization/3-reprojection" / f"{q_frame.id}_{t_frame.id}b.png"
        plot_matches(matches[reproj_mask], q_frame, t_frame, save_path=match_save_path)

    matches = matches[reproj_mask]

    if debug:
        print(f"\t Total filtering: {len(matches) - reproj_mask.sum()}/{len(matches)}. {reproj_mask.sum()} matches left!")

    # ------------------------------------------------------------------------
    # 4. Build the 4x4 Pose matrix
    # ------------------------------------------------------------------------
    # Extract the c1 to c2 pose
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t.flatten()
    # Extract the c2 to c1 pose (this is the new robot's pose in the old coordinate system)
    inv_pose = invert_transform(pose)
    
    # Initialize the frames
    q_frame.initialize(t_frame.id, use_homography, pose)
    q_frame.match[t_frame.id]["init_matches"] = matches
    t_frame.initialize(q_frame.id, use_homography, inv_pose)
    t_frame.match[q_frame.id]["init_matches"] = matches

    return pose, True
      
def triangulate_points(q_frame: Frame, t_frame: Frame, scale: int):
    if debug:
        print(f"Triangulating points between frames {q_frame.id} & {t_frame.id}...")
    # Extract the Rotation and Translation arrays between the 2 frames
    T_qt = q_frame.match[t_frame.id]["T"] # [q->t]
    R_qt = T_qt[:3, :3]
    t_qt = T_qt[:3, 3].reshape(3,1)

    # ------------------------------------------------------------------------
    # 6. Triangulate 3D points
    # ------------------------------------------------------------------------

    # Extract inlier matches
    matches = q_frame.get_init_matches(t_frame.id)

    # Extract keypoint pixel coordinates and indices for both frames from the feature match
    q_kpt_pixels = np.float64([q_frame.keypoints[m.queryIdx].pt for m in matches])
    t_kpt_pixels = np.float64([t_frame.keypoints[m.trainIdx].pt for m in matches])

    # Triangulate
    q_points = triangulate(q_kpt_pixels, t_kpt_pixels, R_qt, t_qt) # (N, 3)
    if q_points is None or len(q_points) == 0:
        print("[initialize] Triangulation returned no 3D points.")
        return None, None, None, False

    # Transfer the points to the current coordinate frame [t->q]
    t_points = transform_points(q_points, T_qt) # (N, 3)

    # Scale the points
    q_points = scale * q_points
    t_points = scale * t_points

    # ------------------------------------------------------------------------
    # 7. Filter triangulated points for Z<0 and small triang. angle
    # ------------------------------------------------------------------------

    triang_mask = filter_triangulation_points(q_points, t_points, R_qt, t_qt)
    # If too few points or too small median angle, return None
    if triang_mask is None or triang_mask.sum() < MIN_NUM_TRIANG_POINTS:
        print("Discarding frame due to insufficient triangulation quality.")
        return None, None, None, False
            
    # Save the matches
    if debug:
        match_save_path = results_dir / "matches/initialization/4-triangulation" / f"{q_frame.id}_{t_frame.id}a.png"
        plot_matches(matches[~triang_mask], q_frame, t_frame, save_path=match_save_path)
        match_save_path = results_dir / "matches/initialization/4-triangulation" / f"{q_frame.id}_{t_frame.id}b.png"
        plot_matches(matches[triang_mask], q_frame, t_frame, save_path=match_save_path)

    matches = matches[triang_mask]
    q_points = q_points[triang_mask]
    t_points = t_points[triang_mask]

    # ------------------------------------------------------------------------
    # 8. Save the triangulated points and masks to the t_frame
    # ------------------------------------------------------------------------

    q_frame.match[t_frame.id]["points"] = q_points
    t_frame.match[q_frame.id]["points"] = t_points
    q_frame.match[t_frame.id]["init_matches"] = matches
    t_frame.match[q_frame.id]["init_matches"] = matches

    # Save the triangulated points keypoints and descriptors
    t_kpts = np.array([t_frame.keypoints[m.trainIdx] for m in matches])
    t_descriptors = np.uint8([t_frame.descriptors[m.trainIdx] for m in matches])

    # Return the initial pose and filtered points
    return t_points, t_kpts, t_descriptors, True
      
def triangulate(q_frame_pixels, t_frame_pixels, R, t):
    # Compute projection matrices for triangulation
    q_M = K @ np.eye(3,4)  # First camera at origin
    t_M = K @ np.hstack((R, t))  # Second camera at R, t

    # Triangulate points
    q_frame_points_4d_hom = cv2.triangulatePoints(q_M, t_M, q_frame_pixels.T, t_frame_pixels.T)

    # Convert homogeneous coordinates to 3D
    q_points_3d = q_frame_points_4d_hom[:3] / q_frame_points_4d_hom[3]

    return q_points_3d.T # (N, 3)
