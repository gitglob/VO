import numpy as np
import cv2
from src.others.frame import Frame
from src.others.linalg import transform_points
from src.others.visualize import plot_matches
from src.others.filtering import enforce_epipolar_constraint, filter_by_reprojection, filter_cheirality, filter_parallax
from src.others.epipolar_geometry import triangulate

from config import results_dir, SETTINGS, K, log


debug = SETTINGS["generic"]["debug"]
MIN_NUM_TRIANG_POINTS = SETTINGS["initialization"]["min_num_triang_points"]
REPROJECTION_THREHSOLD = SETTINGS["initialization"]["reprojection_threshold"]
MIN_PARALLAX = SETTINGS["initialization"]["min_parallax"]


def initialize_pose(matches: list[cv2.DMatch], q_frame: Frame, t_frame: Frame):
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

    Returns:
        Tuple[np.ndarray or None, bool]: 
            - The inverse 4x4 transformation matrix (q_frame -> frame) if successful, otherwise None.
            - A boolean indicating whether the initialization was successful.
    """
    if debug:
        log.info(f"[Initialization] Initializing the camera pose using frames {q_frame.id} & {t_frame.id}...")
    
    # ------------------------------------------------------------------------
    # 1. Get keypoint matches
    # ------------------------------------------------------------------------

    # Extract keypoint pixel coordinates and indices for both frames from the feature match
    q_kpt_pixels = np.float64([q_frame.keypoints[m.queryIdx].pt for m in matches])
    t_kpt_pixels = np.float64([t_frame.keypoints[m.trainIdx].pt for m in matches])

    # ------------------------------------------------------------------------
    # 2. Enforce Epipolar Constraint
    # ------------------------------------------------------------------------

    epipolar_constraint_mask, M, use_homography = enforce_epipolar_constraint(q_kpt_pixels, t_kpt_pixels)
    if epipolar_constraint_mask is None:
        log.warning("\t Failed to apply epipolar constraint..")
        return None, False

    # Save the matches
    if debug:
        match_save_path = results_dir / f"matches/initialization/1-epipolar_constraint" / f"{q_frame.id}_{t_frame.id}a.png"
        plot_matches(matches[~epipolar_constraint_mask], q_frame, t_frame, save_path=match_save_path)
        match_save_path = results_dir / f"matches/initialization/1-epipolar_constraint" / f"{q_frame.id}_{t_frame.id}b.png"
        plot_matches(matches[epipolar_constraint_mask], q_frame, t_frame, save_path=match_save_path)
    matches = matches[epipolar_constraint_mask]
    q_kpt_pixels = q_kpt_pixels[epipolar_constraint_mask]
    t_kpt_pixels = t_kpt_pixels[epipolar_constraint_mask]
        
    # ------------------------------------------------------------------------
    # 3. Recover pose (R, t) from Essential or Homography
    # ------------------------------------------------------------------------

    # Check if we will use homography
    R, t, reproj_mask = None, None, None
    if not use_homography:
        # Decompose Essential Matrix
        _, R, t, mask_pose = cv2.recoverPose(M, q_kpt_pixels, t_kpt_pixels, K)
        if R is None:
            return None, False
        
        # mask_pose indicates inliers used in cv2.recoverPose (1 for inliers, 0 for outliers)
        mask_pose = mask_pose.ravel().astype(bool)
        if debug:
            log.info(f"\t\t Pose Recovery filtered {epipolar_constraint_mask.sum() - mask_pose.sum()}/{epipolar_constraint_mask.sum()} matches!")
        matches = matches[mask_pose]        

        # Reprojection filter
        reproj_mask = filter_by_reprojection(
            matches, q_frame, t_frame,
            R, t,
            REPROJECTION_THREHSOLD,
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
            for j in range(len(t_kpt_pixels)):
                # Current frame pixel in camera coords
                p_t_cam = invK @ np.array([*t_kpt_pixels[j], 1.0])  

                # Depth for current pixel after transformation
                denom = np.dot(n_candidate, R_candidate @ p_t_cam + t_candidate)
                t_depth = np.dot(n_candidate, p_t_cam) / (denom + 1e-12)  # small eps for safety

                if t_depth <= 0:
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
            REPROJECTION_THREHSOLD,
            save_path=results_dir / f"matches/initialization/2-reprojection/"
        )

    # If we failed to recover R and t
    if R is None or t is None:
        log.warning("\t Failed to recover a valid pose from either E or H.")
        return None, False
            
    # Save the matches
    if debug:
        match_save_path = results_dir / "matches/initialization/3-reprojection" / f"{q_frame.id}_{t_frame.id}a.png"
        plot_matches(matches[~reproj_mask], q_frame, t_frame, save_path=match_save_path)
        match_save_path = results_dir / "matches/initialization/3-reprojection" / f"{q_frame.id}_{t_frame.id}b.png"
        plot_matches(matches[reproj_mask], q_frame, t_frame, save_path=match_save_path)

    matches = matches[reproj_mask]

    if debug:
        log.info(f"\t {reproj_mask.sum()} matches left!")

    # ------------------------------------------------------------------------
    # 4. Build the 4x4 Pose matrix
    # ------------------------------------------------------------------------
    # Extract the c1 to c2 pose
    T_q2t = np.eye(4)
    T_q2t[:3, :3] = R
    T_q2t[:3, 3] = t.flatten()

    # The translation should be a unit vector before scaling
    assert np.linalg.norm(T_q2t[:3, 3]) - 1 < 1e-6

    return matches, T_q2t, True
      
def triangulate_points(matches: list[cv2.DMatch], T_q2t: np.ndarray, q_frame: Frame, t_frame: Frame, scale: int):
    if debug:
        log.info(f"[Initialization] Triangulating points between frames {q_frame.id} & {t_frame.id}...")

    # ------------------------------------------------------------------------
    # 6. Triangulate 3D points
    # ------------------------------------------------------------------------

    # Extract keypoint pixel coordinates and indices for both frames from the feature match
    q_kpt_pixels = np.float64([q_frame.keypoints[m.queryIdx].pt for m in matches])
    t_kpt_pixels = np.float64([t_frame.keypoints[m.trainIdx].pt for m in matches])

    # Triangulate
    q_points = triangulate(q_kpt_pixels, t_kpt_pixels, T_q2t) # (N, 3)
    if q_points is None or len(q_points) == 0:
        log.warning("\t Triangulation returned no 3D points.")
        return None, None, None, False

    # Transfer the points to the current coordinate frame [t->q]
    t_points = transform_points(q_points, T_q2t) # (N, 3)

    # Scale the points
    q_points = scale * q_points
    t_points = scale * t_points

    # ------------------------------------------------------------------------
    # 7. Filter triangulated points for Z<0 and small triang. angle
    # ------------------------------------------------------------------------

    cheirality_mask = filter_cheirality(q_points, t_points)

    # If too few points or too small median angle, return None
    if cheirality_mask is None or cheirality_mask.sum() < MIN_NUM_TRIANG_POINTS:
        log.warning("\t Discarding frame after cheirality check.")
        return None, None, None, False
            
    # Save the matches
    if debug:
        match_save_path = results_dir / "matches/initialization/4-cheirality" / f"{q_frame.id}_{t_frame.id}a.png"
        plot_matches(matches[~cheirality_mask], q_frame, t_frame, save_path=match_save_path)
        match_save_path = results_dir / "matches/initialization/4-cheirality" / f"{q_frame.id}_{t_frame.id}b.png"
        plot_matches(matches[cheirality_mask], q_frame, t_frame, save_path=match_save_path)

    matches = matches[cheirality_mask]
    q_points = q_points[cheirality_mask]
    t_points = t_points[cheirality_mask]

    parallax_mask = filter_parallax(q_points, t_points, T_q2t, MIN_PARALLAX)

    # If too few points or too small median angle, return None
    if parallax_mask is None or parallax_mask.sum() < MIN_NUM_TRIANG_POINTS:
        log.warning("\t Discarding frame due to insufficient parallax.")
        return None, None, None, False
            
    # Save the matches
    if debug:
        match_save_path = results_dir / "matches/initialization/5-parallax" / f"{q_frame.id}_{t_frame.id}a.png"
        plot_matches(matches[~parallax_mask], q_frame, t_frame, save_path=match_save_path)
        match_save_path = results_dir / "matches/initialization/5-parallax" / f"{q_frame.id}_{t_frame.id}b.png"
        plot_matches(matches[parallax_mask], q_frame, t_frame, save_path=match_save_path)

    matches = matches[parallax_mask]
    q_points = q_points[parallax_mask]
    t_points = t_points[parallax_mask]
    w_points = transform_points(t_points, t_frame.pose)
    if np.any(np.isnan(w_points)):
        breakpoint()

    # ------------------------------------------------------------------------
    # 8. Save the triangulated points and masks to the t_frame
    # ------------------------------------------------------------------------

    # Save the triangulated points keypoints and descriptors
    q_kpts = np.array([q_frame.keypoints[m.queryIdx] for m in matches])
    t_kpts = np.array([t_frame.keypoints[m.trainIdx] for m in matches])
    q_descriptors = np.uint8([q_frame.descriptors[m.queryIdx] for m in matches])
    t_descriptors = np.uint8([t_frame.descriptors[m.trainIdx] for m in matches])

    # Return the initial pose and filtered points
    return w_points, q_kpts, t_kpts, q_descriptors, t_descriptors, True
      
