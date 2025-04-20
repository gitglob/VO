import numpy as np
import cv2
from src.local_mapping.local_map import Map
from src.others.frame import Frame
from src.others.visualize import plot_reprojection
from config import SETTINGS, results_dir, log


debug = SETTINGS["generic"]["debug"]


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
        reprojectionError=SETTINGS["tracking"]["PnP"]["reprojection_threshold"],
        confidence=SETTINGS["tracking"]["PnP"]["confidence"],
        iterationsCount=SETTINGS["tracking"]["PnP"]["iterations"]
    )
    if not success or inliers is None:
        log.warning("\t solvePnP failed!")
        return None, None
    if len(inliers) < SETTINGS["tracking"]["PnP"]["min_inliers"]:
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
    
    # Save the PnP tracked flags in the map points
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
