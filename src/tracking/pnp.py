import numpy as np
import cv2
import src.utils as utils
import src.visualization as vis
import src.globals as ctx
from config import SETTINGS, results_dir, log, K


debug = SETTINGS["generic"]["debug"]


# Function to estimate the relative pose using solvePnP
def estimate_relative_pose(t_frame: utils.Frame):
    """Estimate the map <-> camera displacement using a 3D-2D PnP approach."""
    t_map_pairs = t_frame.get_map_matches()
    num_matches = len(t_map_pairs)
    log.info(f"Estimating Map -> Frame #{t_frame.id} pose using {num_matches}/{ctx.map.num_points} map points...")

    # 1) Build 3D <-> 2D correspondences
    map_point_positions = []
    image_pxs = []
    for (feat, point) in t_map_pairs:
        map_point_positions.append(point.pos) # 3D in world coords
        image_pxs.append(feat.kpt.pt)         # 2D pixel (u, v)

    map_point_positions = np.array(map_point_positions, dtype=np.float64) # (M, 3)
    image_pxs = np.array(image_pxs, dtype=np.float64)     # (M, 2)

    # 2) solvePnPRansac to get rvec/tvec for world->new_cam
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        map_point_positions,
        image_pxs,
        cameraMatrix=K,
        distCoeffs=None,
        reprojectionError=SETTINGS["tracking"]["PnP"]["max_reprojection"],
        confidence=SETTINGS["tracking"]["PnP"]["confidence"],
        iterationsCount=SETTINGS["tracking"]["PnP"]["iterations"]
    )
    if not success or inliers is None:
        log.warning("\t solvePnP failed!")
        return False
    if len(inliers) < SETTINGS["tracking"]["PnP"]["min_inliers"]:
        log.warning("\t solvePnP did not find enough inliers!")
        return False
    inliers = inliers.flatten()

    # Build an inliers mask
    inliers_mask = np.zeros(num_matches, dtype=bool)
    inliers_mask[inliers] = True
    num_tracked_points = inliers_mask.sum()
    log.info(f"\t solvePnPRansac filtered {num_matches - num_tracked_points}/{num_matches} points.")

    # Based on the outliers, remove the observations from the points and the point matches from the features
    for i, (feat, point) in enumerate(t_map_pairs):
        if not inliers_mask[i]:
            feat.reset_mp_match()
            point.remove_observation(t_frame.id)
    
    # 3) Refine the pose using Levenberg-Marquardt on the inlier correspondences.
    rvec, tvec = cv2.solvePnPRefineLM(
        map_point_positions[inliers],
        image_pxs[inliers],
        cameraMatrix=K,
        distCoeffs=None,
        rvec=rvec,
        tvec=tvec
    )

    # 4) Convert refined pose to a 4x4 transformation matrix.
    t_wc = tvec.flatten()
    R_wc, _ = cv2.Rodrigues(rvec)
    
    # 5) Compute reprojection error
    ## Project the 3D points to 2D using the estimated pose
    projected_world_pxs, _ = cv2.projectPoints(objectPoints=map_point_positions, rvec=rvec, tvec=t_wc, cameraMatrix=K, distCoeffs=None)
    projected_world_pxs = projected_world_pxs.squeeze()
    
    ## Calculate the per-point reprojection error (Euclidean distance)
    errors = np.sqrt(np.sum((image_pxs[inliers_mask] - projected_world_pxs[inliers_mask])**2, axis=1))
    
    ## Create a mask for points with error less than the threshold
    reproj_mask = errors < SETTINGS["tracking"]["PnP"]["max_reprojection"]
    log.info(f"\t Reprojection:")
    log.info(f"\t\t Median/Mean error ({np.median(errors):.2f}, {np.mean(errors):.2f})")
    log.info(f"\t\t Outliers {len(errors) - reproj_mask.sum()}/{len(errors)} points.")

    ## Visualization
    if debug:
        img_path = results_dir / f"tracking/pnp/map_{t_frame.id}a.png"
        vis.plot_reprojection(t_frame.img, image_pxs[~inliers_mask], projected_world_pxs[~inliers_mask], path=img_path)
        img_path = results_dir / f"tracking/pnp/map_{t_frame.id}b.png"
        vis.plot_reprojection(t_frame.img, image_pxs[inliers_mask], projected_world_pxs[inliers_mask], path=img_path)

    # 6) Construct T_{world->cam_new}
    T_wc = np.eye(4, dtype=np.float64)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = t_wc
    T_c2w = utils.invert_transform(T_wc)

    # Set the pose to the current frame
    t_frame.set_pose(T_c2w)

    return True

