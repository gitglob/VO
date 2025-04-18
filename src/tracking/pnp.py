import numpy as np
import cv2
from src.local_mapping.local_map import Map
from src.initialization.initialization import triangulate
from src.others.frame import Frame
from src.others.linalg import transform_points
from src.others.visualize import plot_matches, plot_reprojection
from src.others.filtering import filter_cheirality, filter_parallax, enforce_epipolar_constraint
from src.others.filtering import filter_by_reprojection, filter_scale
from config import SETTINGS, results_dir, log


debug = SETTINGS["generic"]["debug"]
MIN_INLIERS = SETTINGS["PnP"]["min_inliers"]
W = SETTINGS["camera"]["width"]
H = SETTINGS["camera"]["height"]
MIN_NUM_MATCHES = SETTINGS["point_association"]["num_matches"]
HAMMING_THRESHOLD = SETTINGS["point_association"]["hamming_threshold"]


# Function to estimate the relative pose using solvePnP
def estimate_relative_pose(
    map: Map,
    t_frame: Frame,
    map_t_pairs: list,
    K: np.ndarray,
    dist_coeffs=None
):
    """
    Estimate the relative camera displacement using a 3D-2D PnP approach.

    Args:
        map_points_w (np.ndarray): 
            (N, 3) array of 3D map points in world coordinates
            that correspond to the 'map_idx' indices in map_t_pairs.
        t_frame (Frame): 
            The current t_frame containing keypoints and descriptors.
        map_t_pairs (list of (int, int)): 
            - map_idx is the index into map_points_w
            - frame_idx is the index of t_frame.keypoints
        K (np.ndarray): 
            (3, 3) camera intrinsic matrix.
        dist_coeffs:
            Distortion coefficients for the camera. Default = None.

    Returns:
        displacement (np.ndarray): 4×4 transformation matrix T_{cam_new <- cam_old}.
                                    i.e., the relative transform from the old camera t_frame
                                    to the new camera t_frame.

        If the function fails, returns (None, None).
    """
    map_points_w = map.get_frustum_points(t_frame)
    num_points = len(map_t_pairs)
    if debug:
        log.info(f"Estimating Map -> Frame #{t_frame.id} pose using {num_points}/{len(map_points_w)} map points...")

    # 1) Build 3D <-> 2D correspondences
    map_points = []
    image_pxs = []
    for (point_id, feature_idx) in map_t_pairs:
        map_points.append(map.points[point_id])  # 3D in world coords
        kp = t_frame.keypoints[feature_idx]
        image_pxs.append(kp.pt)                   # 2D pixel (u, v)

    map_points = np.array(map_points, dtype=object)   
    map_point_positions = np.array([p.pos for p in map_points], dtype=np.float64) # (M, 3)
    image_pxs = np.array(image_pxs, dtype=np.float64)     # (M, 2)

    # 2) solvePnPRansac to get rvec/tvec for world->new_cam
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        map_point_positions,
        image_pxs,
        cameraMatrix=K,
        distCoeffs=dist_coeffs,
        reprojectionError=SETTINGS["PnP"]["reprojection_threshold"],
        confidence=SETTINGS["PnP"]["confidence"],
        iterationsCount=SETTINGS["PnP"]["iterations"]
    )
    if not success or inliers is None:
        log.warning("\t solvePnP failed!")
        return None, None
    if len(inliers) < MIN_INLIERS:
        log.warning("\t solvePnP did not find enough inliers!")
        return None, None
    inliers = inliers.flatten()

    # Build an inliers mask
    inliers_mask = np.zeros(num_points, dtype=bool)
    inliers_mask[inliers] = True
    num_tracked_points = inliers_mask.sum()
    if debug:
        log.info(f"\t solvePnPRansac filtered {num_points - num_tracked_points}/{num_points} points.")
    
    # 3) Refine the pose using Levenberg-Marquardt on the inlier correspondences.
    rvec, tvec = cv2.solvePnPRefineLM(
        map_point_positions[inliers],
        image_pxs[inliers],
        K,
        dist_coeffs,
        rvec,
        tvec
    )

    # 4) Convert refined pose to a 4x4 transformation matrix.
    t_wc = tvec.flatten()
    R_wc, _ = cv2.Rodrigues(rvec)
    
    # Save the PnP tracked mask to the map
    tracking_mask = np.zeros(map.num_points, dtype=bool)
    for i, (map_idx, _) in enumerate(map_t_pairs): 
        tracking_mask[map_idx] = inliers_mask[i]
    map.set_tracking_mask(tracking_mask)

    # 5) Compute reprojection error
    ## Project the 3D points to 2D using the estimated pose
    projected_world_pxs, _ = cv2.projectPoints(map_point_positions, rvec, t_wc, dist_coeffs)
    projected_world_pxs = projected_world_pxs.squeeze()
    
    ## Calculate the per-point reprojection error (Euclidean distance)
    errors = np.sqrt(np.sum((image_pxs[inliers_mask] - projected_world_pxs[inliers_mask])**2, axis=1))
    
    ## Create a mask for points with error less than the threshold
    reproj_mask = errors < SETTINGS["PnP"]["reprojection_threshold"]
    if debug:
        log.info(f"\t Reprojection:",
              f"\n\t\t Median/Mean error ({np.median(errors):.2f}, {np.mean(errors):.2f})",
              f"\n\t\t Outliers {len(errors) - reproj_mask.sum()}/{len(errors)} points.")

    ## Visualization
    if debug:
        img_path = results_dir / f"matches/tracking/1-PnP_reprojection/map_{t_frame.id}a.png"
        plot_reprojection(t_frame.img, image_pxs[~inliers_mask], projected_world_pxs[~inliers_mask], path=img_path)
        img_path = results_dir / f"matches/tracking/1-PnP_reprojection/map_{t_frame.id}b.png"
        plot_reprojection(t_frame.img, image_pxs[inliers_mask], projected_world_pxs[inliers_mask], path=img_path)

    # 6) Construct T_{world->cam_new}
    T_wc = np.eye(4, dtype=np.float64)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = t_wc

    return T_wc, num_tracked_points

def triangulateNewPoints(q_frame: Frame, t_frame: Frame, map: Map):
    """
    Identifies and triangulates new 3D points from feature matches between two frames.

    This function extracts the relative pose between the current and reference frames.
    It then triangulates 3D points from feature correspondences, filters newly observed points, and updates the map.

    Args:
        q_frame (Frame): The previous reference t_frame.
        t_frame (Frame): The current t_frame containing keypoints, descriptors, and matches.
        map (Map): The global map storing existing 3D points.
        K (np.ndarray): The camera intrinsic matrix (3x3).

    Returns:
        Tuple[np.ndarray or None, np.ndarray or None, np.ndarray or None, bool]: 
            - The inverse 4x4 transformation matrix (q_frame -> t_frame) if successful, otherwise None.
            - A (N,3) array of newly triangulated 3D points if successful, otherwise None.
            - A (N,) array of associated point IDs if successful, otherwise None.
            - A boolean indicating whether triangulation was successful.
    """
    if debug:
        log.info(f"Triangulating New Points using keyframes {q_frame.id} & {t_frame.id}...")
    
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

    epipolar_constraint_mask, _, _ = enforce_epipolar_constraint(q_kpt_pixels, t_kpt_pixels)
    if epipolar_constraint_mask is None:
        log.warning("[Tracking] Failed to apply epipolar constraint..")
        return None, None, None, None, None, None, False
    
    # Save the matches
    if debug:
        match_save_path = results_dir / f"matches/mapping/1-epipolar_constraint" / f"{q_frame.id}_{t_frame.id}a.png"
        plot_matches(matches[~epipolar_constraint_mask], q_frame, t_frame, save_path=match_save_path)
        match_save_path = results_dir / f"matches/mapping/1-epipolar_constraint" / f"{q_frame.id}_{t_frame.id}b.png"
        plot_matches(matches[epipolar_constraint_mask], q_frame, t_frame, save_path=match_save_path)
    matches = matches[epipolar_constraint_mask]

    # Extract the q->t transformation
    # Extract the Rotation and Translation arrays between the 2 frames
    T_qt = q_frame.match[t_frame.id]["T"] # [q->t]
    R_qt = T_qt[:3, :3]
    t_qt = T_qt[:3, 3].reshape(3,1)
   
    # ------------------------------------------------------------------------
    # 3. Filter based on the reprojection of the PnP pose
    # ------------------------------------------------------------------------ 

    reproj_mask = filter_by_reprojection(
        matches, q_frame, t_frame,
        R_qt, t_qt,
        save_path= results_dir / f"matches/mapping/2-reprojection"
    )
    
    # Save the matches
    if debug:
        match_save_path = results_dir / f"matches/mapping/3-reprojection" / f"{q_frame.id}_{t_frame.id}a.png"
        plot_matches(matches[~reproj_mask], q_frame, t_frame, save_path=match_save_path)
        match_save_path = results_dir / f"matches/mapping/3-reprojection" / f"{q_frame.id}_{t_frame.id}b.png"
        plot_matches(matches[reproj_mask], q_frame, t_frame, save_path=match_save_path)
    matches = matches[reproj_mask]
    
    # ------------------------------------------------------------------------
    # 4. Triangulate the matched points using the PnP estimated pose
    # ------------------------------------------------------------------------

    # Extract the pixel coordinates of the points
    q_pxs = np.array([q_frame.keypoints[m.queryIdx].pt for m in matches])
    t_pxs = np.array([t_frame.keypoints[m.trainIdx].pt for m in matches])

    # Triangulate
    q_points = triangulate(q_pxs, t_pxs, R_qt, t_qt)
    if q_points is None or len(q_points) == 0:
        log.warning("Triangulation returned no 3D points.")
        return None, None, None, None, None, None, False

    # Transfer the points to the current coordinate frame [t->q]
    t_points = transform_points(q_points, T_qt) # (N, 3)

    # ------------------------------------------------------------------------
    # 5. Filter triangulated points for Z<0 and small triang. angle
    # ------------------------------------------------------------------------

    cheirality_mask = filter_cheirality(q_points, t_points)

    # If too few points or too small median angle, return None
    if cheirality_mask is None or cheirality_mask.sum():
        log.warning("Discarding frame after cheirality check.")
        return None, None, None, False
            
    # Save the matches
    if debug:
        match_save_path = results_dir / "matches/mapping/4-cheirality" / f"{q_frame.id}_{t_frame.id}a.png"
        plot_matches(matches[~cheirality_mask], q_frame, t_frame, save_path=match_save_path)
        match_save_path = results_dir / "matches/mapping/4-cheirality" / f"{q_frame.id}_{t_frame.id}b.png"
        plot_matches(matches[cheirality_mask], q_frame, t_frame, save_path=match_save_path)

    matches = matches[cheirality_mask]
    q_points = q_points[cheirality_mask]
    t_points = t_points[cheirality_mask]

    parallax_mask = filter_parallax(q_points, t_points, T_qt)
    q_points = q_points[parallax_mask]
    t_points = t_points[parallax_mask]

    # If too few points or too small median angle, return None
    if parallax_mask is None or parallax_mask.sum():
        log.warning("Discarding frame due to insufficient parallax.")
        return None, None, None, False
            
    # Save the matches
    if debug:
        match_save_path = results_dir / "matches/mapping/5-parallax" / f"{q_frame.id}_{t_frame.id}a.png"
        plot_matches(matches[~parallax_mask], q_frame, t_frame, save_path=match_save_path)
        match_save_path = results_dir / "matches/mapping/5-parallax" / f"{q_frame.id}_{t_frame.id}b.png"
        plot_matches(matches[parallax_mask], q_frame, t_frame, save_path=match_save_path)

    matches = matches[parallax_mask]
    q_points = q_points[parallax_mask]
    t_points = t_points[parallax_mask]

    # Extract the keypoints and descriptors of the valid triangulated 3d points
    q_kpts = np.array([q_frame.keypoints[m.queryIdx] for m in matches])
    t_kpts = np.array([t_frame.keypoints[m.trainIdx] for m in matches])

    # ------------------------------------------------------------------------
    # 6. Filter triangulated points based on scale
    # ------------------------------------------------------------------------

    q_scale_mask = filter_scale(q_points, q_kpts, q_frame.pose)
    t_scale_mask = filter_scale(t_points, t_kpts, t_frame.pose)
    scale_mask = q_scale_mask & t_scale_mask
    if scale_mask is None or scale_mask.sum() == 0:
        return None, None, None, None, None, None, False

    # Apply the scale mask to points, keypoints and descriptors
    matches = matches[scale_mask]
    q_points = q_points[q_scale_mask]
    t_points = t_points[t_scale_mask]
    q_kpts = q_kpts[t_scale_mask]
    t_kpts = t_kpts[t_scale_mask]

    # ------------------------------------------------------------------------
    # 7. Find 3D points that have/haven't been triangulated before
    # ------------------------------------------------------------------------

    # Extract the reference keypoint ids
    q_kpt_ids = np.array([q_frame.keypoints[m.queryIdx].class_id for m in matches])

    # Find which of the map keypoints don't intersect with the reference inlier keypoint IDs - these are new
    new_ids = np.setdiff1d(q_kpt_ids, map.point_ids)
    if len(new_ids) == 0:
        log.warning("No new points to triangulate.")
        return None, None, None, None, None, None, False
    if debug:
        log.warning(f"\t {len(new_ids)}/{len(q_kpt_ids)} points are new!")
    
    # Create a mask for the old/new triangulated points
    new_points_mask = np.isin(q_kpt_ids, new_ids)
    old_points_mask = ~new_points_mask

    # Apply the masks
    new_matches = matches[new_points_mask]
    old_matches = matches[old_points_mask]
    
    # Save the matches
    if debug:
        match_save_path = results_dir / f"matches/mapping/5-old_new" / f"{q_frame.id}_{t_frame.id}_old.png"
        plot_matches(old_matches, q_frame, t_frame, save_path=match_save_path)
        match_save_path = results_dir / f"matches/mapping/5-old_new" / f"{q_frame.id}_{t_frame.id}_new.png"
        plot_matches(new_matches, q_frame, t_frame, save_path=match_save_path)

    # ------------------------------------------------------------------------
    # 8. Save the old/new keypoints and descriptors
    # ------------------------------------------------------------------------

    q_old_points = q_points[old_points_mask]
    q_new_points = q_points[new_points_mask]

    t_old_points = t_points[old_points_mask]
    t_new_points = t_points[new_points_mask]

    t_old_kpts = t_kpts[old_points_mask]
    t_new_kpts = t_kpts[new_points_mask]

    t_old_descriptors = np.uint8([t_frame.descriptors[m.trainIdx] for m in old_matches])
    t_new_descriptors = np.uint8([t_frame.descriptors[m.trainIdx] for m in new_matches])

    # Return the newly triangulated points
    return t_old_points, t_old_kpts, t_old_descriptors, t_new_points, t_new_kpts, t_new_descriptors, True
