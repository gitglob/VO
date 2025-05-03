from pathlib import Path
import numpy as np
import cv2
import src.utils as utils
import src.visualization as vis
import src.utils as utils

from config import results_dir, SETTINGS, K, log


debug = SETTINGS["generic"]["debug"]
MIN_NUM_TRIANG_POINTS = SETTINGS["initialization"]["min_triang_points"]
REPROJECTION_THREHSOLD = SETTINGS["initialization"]["max_reprojection"]
MIN_PARALLAX = SETTINGS["initialization"]["min_parallax"]


def estimate_pose(matches: list[cv2.DMatch], q_frame: utils.Frame, t_frame: utils.Frame):
    """
    Initializes the camera pose by estimating the relative rotation and translation 
    between two consecutive frames using feature matches.

    This function computes the Essential and Homography matrices to determine 
    the best motion model. It then recovers the relative pose (rotation and 
    translation) using the Essential matrix if the motion is mostly translational 
    or the Homography matrix if the scene is planar. Finally, the pose is used 
    to initialize the frames and utils.triangulate 3D points.

    Args:
        q_frame (utils.Frame): The previous frame.
        t_frame (utils.Frame): The current frame.

    Returns:
        Tuple[np.ndarray or None, bool]: 
            - The inverse 4x4 transformation matrix (q_frame -> frame) if successful, otherwise None.
            - A boolean indicating whether the initialization was successful.
    """
    log.info(f"Initializing the camera pose using frames {q_frame.id} & {t_frame.id}...")
    
    # ------------------------------------------------------------------------
    # 1. Get keypoint matches
    # ------------------------------------------------------------------------

    # Extract keypoint pixel coordinates and indices for both frames from the feature match
    q_kpt_pixels = np.float64([q_frame.keypoints[m.queryIdx].pt for m in matches])
    t_kpt_pixels = np.float64([t_frame.keypoints[m.trainIdx].pt for m in matches])

    # ------------------------------------------------------------------------
    # 2. Enforce Epipolar Constraint
    # ------------------------------------------------------------------------

    ret = utils.enforce_epipolar_constraint(q_kpt_pixels, t_kpt_pixels)
    if ret is None:
        log.warning("\t Failed to apply epipolar constraint..")
        return None, False
    epipolar_constraint_mask, M, use_homography = ret
    log.info(f"\t Epipolar Constraint: Filtered {sum(~epipolar_constraint_mask)}/{len(q_kpt_pixels)} matches! (Using: {'Homography' if use_homography else 'Essential'}.)")

    # Save the matches
    if debug:
        match_save_path = results_dir / f"initialization/1-epipolar_constraint" / f"{q_frame.id}_{t_frame.id}a.png"
        vis.plot_matches(matches[~epipolar_constraint_mask], q_frame, t_frame, save_path=match_save_path)
        match_save_path = results_dir / f"initialization/1-epipolar_constraint" / f"{q_frame.id}_{t_frame.id}b.png"
        vis.plot_matches(matches[epipolar_constraint_mask], q_frame, t_frame, save_path=match_save_path)
    matches = matches[epipolar_constraint_mask]
    q_kpt_pixels = q_kpt_pixels[epipolar_constraint_mask]
    t_kpt_pixels = t_kpt_pixels[epipolar_constraint_mask]
        
    # ------------------------------------------------------------------------
    # 3. Recover pose (R, t) from Essential or Homography
    # ------------------------------------------------------------------------

    # Check if we will use homography
    R, t = None, None
    if not use_homography:
        # Decompose Essential Matrix
        _, R, t, mask_pose = cv2.recoverPose(M, q_kpt_pixels, t_kpt_pixels, K)
        if R is None:
            return None, False
        
        # mask_pose indicates inliers used in cv2.recoverPose (1 for inliers, 0 for outliers)
        mask_pose = mask_pose.ravel().astype(bool)
        log.info(f"\t Pose Recovery filtered {epipolar_constraint_mask.sum() - mask_pose.sum()}/{epipolar_constraint_mask.sum()} matches!")
        matches = matches[mask_pose]        
    else:
        # Decompose Homography Matrix
        num_solutions, Rs, Ts, Ns = cv2.decomposeHomographyMat(M, K)

        # desired “up” in camera coordinates (z‐axis)
        desired_normal = np.array([0, 0, 1.0])
        max_inliers = -1

        for i in range(num_solutions):
            R_i = Rs[i]
            t_i = Ts[i]
            n_i = Ns[i].ravel()

            # 1) normal must face forward
            if float(n_i @ desired_normal) < 0:
                continue

            # 2) count front‐points
            front = 0
            invK  = np.linalg.inv(K)
            for (u, v) in t_kpt_pixels:
                X = invK @ np.array([[u], [v], [1.0]])
                Xp = R_i @ X + t_i
                if Xp[2] > 0:
                    front += 1

            # 3) keep the best
            if front > max_inliers:
                max_inliers = front
                best_candidate_idx = i

        # Use the best solution
        R = Rs[best_candidate_idx]
        t = Ts[best_candidate_idx]

    # If we failed to recover R and t
    if R is None or t is None:
        log.warning("\t Failed to recover a valid pose from either E or H.")
        return None, False

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
      
def triangulate_points(matches: list[cv2.DMatch], T_q2t: np.ndarray, q_frame: utils.Frame, t_frame: utils.Frame, scale: int):
    log.info(f"Triangulating points between frames {q_frame.id} & {t_frame.id}...")

    # ------------------------------------------------------------------------
    # 6. Triangulate 3D points
    # ------------------------------------------------------------------------

    # Extract keypoint pixel coordinates and indices for both frames from the feature match
    q_kpt_pixels = np.float64([q_frame.keypoints[m.queryIdx].pt for m in matches])
    t_kpt_pixels = np.float64([t_frame.keypoints[m.trainIdx].pt for m in matches])

    # Triangulate
    q_points = utils.triangulate(q_kpt_pixels, t_kpt_pixels, T_q2t) # (N, 3)
    if q_points is None or len(q_points) == 0:
        log.warning("\t Triangulation returned no 3D points.")
        return None

    # Transfer the points to the current coordinate frame [t->q]
    t_points = utils.transform_points(q_points, T_q2t) # (N, 3)

    # Scale the points
    q_points = scale * q_points
    t_points = scale * t_points

    # ------------------------------------------------------------------------
    # 7. Filter triangulated points for Z<0
    # ------------------------------------------------------------------------

    cheirality_mask = utils.filter_cheirality(q_points, t_points)

    # If too few points or too small median angle, return None
    log.info(f"\t Cheirality check filtered {sum(~cheirality_mask)}/{len(q_points)} points!")
    if cheirality_mask.sum() < MIN_NUM_TRIANG_POINTS:
        log.warning("\t Discarding frame after cheirality check.")
        return None

    # Save the matches
    if debug:
        match_save_path = results_dir / "initialization/2-cheirality" / f"{q_frame.id}_{t_frame.id}-a.png"
        vis.plot_matches(matches[~cheirality_mask], q_frame, t_frame, save_path=match_save_path)
        match_save_path = results_dir / "initialization/2-cheirality" / f"{q_frame.id}_{t_frame.id}-b.png"
        vis.plot_matches(matches[cheirality_mask], q_frame, t_frame, save_path=match_save_path)

    matches = matches[cheirality_mask]
    q_points = q_points[cheirality_mask]
    t_points = t_points[cheirality_mask]
            
    # ------------------------------------------------------------------------
    # 7. Filter triangulated points for small parallax
    # ------------------------------------------------------------------------

    parallax_mask = utils.filter_parallax(q_points, t_points, T_q2t, MIN_PARALLAX)

    # If too few points or too small median angle, return None
    log.info(f"\t Parallax check filtered {sum(~parallax_mask)}/{len(q_points)} points!")
    if parallax_mask.sum() < MIN_NUM_TRIANG_POINTS:
        log.warning("\t Discarding frame due to insufficient parallax.")
        return None

    # Save the matches
    if debug:
        match_save_path = results_dir / "initialization/3-parallax" / f"{q_frame.id}_{t_frame.id}-a.png"
        vis.plot_matches(matches[~parallax_mask], q_frame, t_frame, save_path=match_save_path)
        match_save_path = results_dir / "initialization/3-parallax" / f"{q_frame.id}_{t_frame.id}-b.png"
        vis.plot_matches(matches[parallax_mask], q_frame, t_frame, save_path=match_save_path)

    matches = matches[parallax_mask]
    q_points = q_points[parallax_mask]

    # ------------------------------------------------------------------------
    # 7. Filter triangulated points by pixel reprojection
    # ------------------------------------------------------------------------

    # Reprojection error filter
    t_kpt_pixels = np.float64([t_frame.keypoints[m.trainIdx].pt for m in matches])
    reproj_mask, t_proj_pxs = utils.filter_by_reprojection(q_points, t_kpt_pixels, T_q2t, REPROJECTION_THREHSOLD
    )
    log.info(f"\t Reprojection filtered: {sum(~reproj_mask)}/{len(q_points)} matches!")
    if debug:
        save_path = results_dir / f"initialization/4-reprojection/{q_frame.id}_{t_frame.id}.png"
        s1 = Path(str(save_path.with_suffix("")) + "-a" + save_path.suffix)
        s2 = Path(str(save_path.with_suffix("")) + "-b" + save_path.suffix)
        vis.plot_reprojection(t_frame.img, t_kpt_pixels[~reproj_mask], t_proj_pxs[~reproj_mask], path=s1)
        vis.plot_reprojection(t_frame.img, t_kpt_pixels[reproj_mask], t_proj_pxs[reproj_mask], path=s2)

    # Save the matches
    if debug:
        match_save_path = results_dir / "initialization/5-reprojection" / f"{q_frame.id}_{t_frame.id}-a.png"
        vis.plot_matches(matches[~reproj_mask], q_frame, t_frame, save_path=match_save_path)
        match_save_path = results_dir / "initialization/5-reprojection" / f"{q_frame.id}_{t_frame.id}-b.png"
        vis.plot_matches(matches[reproj_mask], q_frame, t_frame, save_path=match_save_path)
        
    if reproj_mask.sum() < MIN_NUM_TRIANG_POINTS: 
        log.warning("\t Discarding frame due to high reprojection error.")
        return None          
        
    matches = matches[reproj_mask]
    q_points = q_points[reproj_mask]

    # Return the initial pose and filtered points
    w_points = utils.transform_points(q_points, q_frame.pose)

    return w_points, matches
      
