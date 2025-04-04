import numpy as np
import cv2
from src.others.local_map import Map, mapPoint
from src.frontend.initialization import triangulate
from src.others.frame import Frame
from src.others.utils import get_yaw, transform_points
from src.others.visualize import plot_matches, plot_reprojection, plot_pixels
from src.others.filtering import filter_triangulation_points, enforce_epipolar_constraint, filter_by_reprojection, filter_scale
from config import SETTINGS, results_dir


debug = SETTINGS["generic"]["debug"]
MIN_INLIERS = SETTINGS["PnP"]["min_inliers"]
W = SETTINGS["camera"]["width"]
H = SETTINGS["camera"]["height"]
MIN_NUM_MATCHES = SETTINGS["point_association"]["num_matches"]


def pointAssociation(map: Map, t_frame: Frame):
    """
    Matches the map points seen in the previous frame with the current frame.

    Args:
        map: The local map
        t_frame: The current t_frame, which has .keypoints and .descriptors
        T_wc: The predicted camera pose
        search_window: half-size of the bounding box (in pixels) around (u,v).

    Returns:
        pairs: (map_idx, frame_idx) indicating which map point matched which t_frame keypoint
    """
    # Extract the in view descriptors
    map_points = map.points_in_view
    map_descriptors = []
    map_descriptors_idx = []
    for i, p in enumerate(map_points):
        # Get the descriptors from every observation of a point
        for obs in p.observations:
            map_descriptors.append(obs["descriptor"])
            map_descriptors_idx.append(i)
    map_descriptors = np.array(map_descriptors)
    map_descriptors_idx = np.array(map_descriptors_idx)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # Match descriptors
    matches = bf.knnMatch(map_descriptors, t_frame.descriptors, k=2)
    if len(matches) < MIN_NUM_MATCHES:
        return []

    # Filter matches
    # Apply Lowe's ratio test to filter out false matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good_matches.append(m)
    if debug:
        print(f"\t Lowe's Test filtered {len(matches) - len(good_matches)}/{len(matches)} matches!")

    if len(good_matches) < MIN_NUM_MATCHES:
        return []
    
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

    if len(unique_matches) < MIN_NUM_MATCHES:
        return []
            
    # Save the matches
    if debug:
        match_save_path = results_dir / "matches/tracking/0-point_association" / f"map_{t_frame.id}_b.png"
        t_pxs = np.array([t_frame.keypoints[m.trainIdx].pt for m in unique_matches], dtype=np.float64)
        plot_pixels(t_frame.img, t_pxs, save_path=match_save_path)
    
    # Prepare results
    pairs = []  # list of (map_idx, frame_idx, best_dist)
    for m in unique_matches:
        pairs.append((map_descriptors_idx[m.queryIdx], m.trainIdx))

    if debug:
        print(f"\t Found {len(pairs)} Point Associations!")

    return pairs

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
    map_points_w = map.points_in_view
    num_points = len(map_t_pairs)
    print(f"Estimating Map -> Frame #{t_frame.id} pose using {num_points}/{len(map_points_w)} map points...")

    # 1) Build 3D <-> 2D correspondences
    map_points = []
    image_pxs = []
    for (map_idx, frame_idx) in map_t_pairs:
        map_points.append(map_points_w[map_idx])  # 3D in world coords
        kp = t_frame.keypoints[frame_idx]
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
        print("\t solvePnP failed!")
        return None, None
    if len(inliers) < MIN_INLIERS:
        print("\t solvePnP did not find enough inliers!")
        return None, None
    inliers = inliers.flatten()

    # Build an inliers mask
    inliers_mask = np.zeros(num_points, dtype=bool)
    inliers_mask[inliers] = True
    num_tracked_points = inliers_mask.sum()
    if debug:
        print(f"\t solvePnPRansac filtered {num_points - num_tracked_points}/{num_points} points.")
    
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
    
    # Invrease the match counter for all matched points
    for p in map_points[inliers_mask]:
        p.match_counter += 1

    # 5) Compute reprojection error
    ## Project the 3D points to 2D using the estimated pose
    projected_world_pxs, _ = cv2.projectPoints(map_point_positions, rvec, t_wc, K, dist_coeffs)
    projected_world_pxs = projected_world_pxs.squeeze()
    
    ## Calculate the per-point reprojection error (Euclidean distance)
    errors = np.sqrt(np.sum((image_pxs[inliers_mask] - projected_world_pxs[inliers_mask])**2, axis=1))
    
    ## Create a mask for points with error less than the threshold
    reproj_mask = errors < SETTINGS["PnP"]["reprojection_threshold"]
    if debug:
        print(f"\t Reprojection:",
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

def is_keyframe(T: np.ndarray, num_tracked_points: int):
    """ Determine if motion expressed by t, R is significant by comparing to tresholds. """
    tx = T[0, 3] # The x component points right
    ty = T[1, 3] # The y component points down
    tz = T[2, 3] # The z component points forward

    trans = np.sqrt(tx**2 + ty**2 + tz**2)
    angle = abs(get_yaw(T[:3, :3]))

    is_keyframe = num_tracked_points > SETTINGS["keyframe"]["num_tracked_points"]
    # is_keyframe = is_keyframe and (trans > SETTINGS["keyframe"]["distance"] or angle > SETTINGS["keyframe"]["angle"])
    
    if debug:
        print(f"\t Tracked points: {num_tracked_points}, dist: {trans:.3f}, angle: {angle:.3f}")
        if is_keyframe:
            print("\t\t Keyframe!")
        else:
            print("\t\t Not a keyframe!")

    return is_keyframe

def triangulateNewPoints(q_frame: Frame, t_frame: Frame, map: Map, K: np.ndarray):
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
        print(f"Triangulating New Points using keyframes {q_frame.id} & {t_frame.id}...")
    
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

    epipolar_constraint_mask, _, _ = enforce_epipolar_constraint(q_kpt_pixels, t_kpt_pixels, K)
    if epipolar_constraint_mask is None:
        print("[Tracking] Failed to apply epipolar constraint..")
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
        R_qt, t_qt, K,
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
    q_points = triangulate(q_pxs, t_pxs, R_qt, t_qt, K)
    if q_points is None or len(q_points) == 0:
        print("Triangulation returned no 3D points.")
        return None, None, None, None, None, None, False

    # Transfer the points to the current coordinate frame [t->q]
    t_points = transform_points(q_points, T_qt) # (N, 3)

    # ------------------------------------------------------------------------
    # 5. Filter triangulated points for Z<0 and small triang. angle
    # ------------------------------------------------------------------------

    triang_mask = filter_triangulation_points(q_points, t_points, R_qt, t_qt)
    if triang_mask is None or triang_mask.sum() == 0:
        return None, None, None, None, None, None, False
    
    # Save the matches
    if debug:
        match_save_path = results_dir / f"matches/mapping/4-triangulation" / f"{q_frame.id}_{t_frame.id}a.png"
        plot_matches(matches[~triang_mask], q_frame, t_frame, save_path=match_save_path)
        match_save_path = results_dir / f"matches/mapping/4-triangulation" / f"{q_frame.id}_{t_frame.id}b.png"
        plot_matches(matches[triang_mask], q_frame, t_frame, save_path=match_save_path)

    matches = matches[triang_mask]
    q_points = q_points[triang_mask]
    t_points = t_points[triang_mask]

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
        print("No new points to triangulate.")
        return None, None, None, None, None, None, False
    if debug:
        print(f"\t {len(new_ids)}/{len(q_kpt_ids)} points are new!")
    
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
    q_frame.match[t_frame.id]["points"] = q_new_points

    t_old_points = t_points[old_points_mask]
    t_new_points = t_points[new_points_mask]
    t_frame.match[q_frame.id]["points"] = t_new_points

    t_old_kpts = t_kpts[old_points_mask]
    t_new_kpts = t_kpts[new_points_mask]

    t_old_descriptors = np.uint8([t_frame.descriptors[m.trainIdx] for m in old_matches])
    t_new_descriptors = np.uint8([t_frame.descriptors[m.trainIdx] for m in new_matches])

    # Return the newly triangulated points
    return t_old_points, t_old_kpts, t_old_descriptors, t_new_points, t_new_kpts, t_new_descriptors, True
