import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import src.utils as utils
import src.globals as ctx
import src.visualization as vis
from .bow import query_recognition_candidate
from config import SETTINGS, log, K, results_dir


DEBUG = SETTINGS["generic"]["debug"]


def detect_candidates(frame: utils.Frame) -> set[utils.Frame]:
    """Find suitable candidates for loop closure using the convisibility graph.""" 
    if DEBUG:
        log.info(f"\t Searching for loop closing candidates with frame {frame.id}")
    if frame.bow_hist is None:
        log.warning("\t No BoW descriptor computed for the current frame.")
        return None

    # Find neighbors that share at least 30 points
    neighbor_frame_ids = ctx.cgraph.get_connected_frames_with_min_w(frame.id, 30)

    # Iterate over all the neighbor frames
    min_score = np.inf
    for other_kf_id in neighbor_frame_ids:
        other_kf = ctx.map.keyframes[other_kf_id]

        # Compare the histograms of the 2 frames
        # Use cosine similarity: higher score indicates greater similarity.
        score = cosine_similarity(frame.bow_hist, other_kf.bow_hist)[0][0]

        # Find the minimum score 
        if score < min_score:
            min_score = score

    # Query the recognition database
    candidates = query_recognition_candidate(frame)
    if candidates is None:
        return None

    # Keep only candidates that have better similarity than the minimum score of the neighbors, and aren't neighbors
    good_candidates = [(kf_id, score) for kf_id, score in candidates if (score >= min_score) and (kf_id not in neighbor_frame_ids)]
    if len(good_candidates) == 0:
        log.warning("\t No candidates found!")
        return None
    candidate_kfs = [ctx.map.keyframes[kf_id] for kf_id, _ in good_candidates]
    
    # Find the best candidate
    best_candidate = max(good_candidates, key=lambda x: x[1])
    if DEBUG:
        log.info(f"\t Loop closure best candidate: {best_candidate[0]}, score: {best_candidate[1]:.2f}")
    best_candidate_kf = ctx.map.keyframes[best_candidate[0]]

    return candidate_kfs
    
def frame_search(q_frame: utils.Frame, t_frame: utils.Frame, use_epipolar_constraint: bool = False):
    """
    Matches the map points seen in a previous frame with the current frame.

    Args:
        map: The local map
        t_frame: The current t_frame, which has .keypoints and .descriptors
        T_wc: The predicted camera pose
        search_window: half-size of the bounding box (in pixels) around (u,v).

    Returns:
        pairs: (map_idx, frame_idx) indicating which map point matched which t_frame keypoint
    """
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # Match descriptors
    points, q_features = q_frame.get_map_points_and_features()
    q_descriptors = [f.desc for f in q_features]
    matches = bf.knnMatch(q_descriptors, t_frame.descriptors, k=2)
    if len(matches) < 10:
        return []

    # Filter matches
    # Apply Lowe's ratio test to filter out false matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good_matches.append(m)
    if DEBUG:
        log.info(f"\t Lowe's Test filtered {len(matches) - len(good_matches)}/{len(matches)} matches!")

    if len(good_matches) < 10:
        return []
    
    # Next, ensure uniqueness by keeping only the best match per train descriptor.
    unique_matches = {}
    for m in good_matches:
        # If this train descriptor is not seen yet, or if the current match is better, update.
        if m.trainIdx not in unique_matches or m.distance < unique_matches[m.trainIdx].distance:
            unique_matches[m.trainIdx] = m

    # Convert the dictionary values to a list of unique matches
    unique_matches = list(unique_matches.values())
    if DEBUG:
        log.info(f"\t Uniqueness filtered {len(good_matches) - len(unique_matches)}/{len(good_matches)} matches!")

    if len(unique_matches) < 10:
        return []
    
    # Finally, filter using the epipolar constraint
    if use_epipolar_constraint:
        q_pixels = np.array([q_features[m.queryIdx].kpt.pt for m in unique_matches], dtype=np.float64)
        t_pixels = np.array([t_frame.keypoints[m.trainIdx].pt for m in unique_matches], dtype=np.float64)

        epipolar_constraint_mask, _, _ = utils.enforce_epipolar_constraint(q_pixels, t_pixels)
        if epipolar_constraint_mask is None:
            log.warning("Failed to apply epipolar constraint..")
            return []
        unique_matches = np.array(unique_matches)[epipolar_constraint_mask].tolist()
    
    # Prepare results
    cv2_matches = []
    m: cv2.DMatch
    for m in unique_matches:
        q_feat = q_features[m.queryIdx]
        point = points[m.queryIdx]
        t_feat = t_frame.features[m.trainIdx]

        t_feat.match_map_point(point, m.distance)
        point.observe(ctx.map._kf_counter, t_frame.id, t_feat.kpt, t_feat.desc)
        ctx.cgraph.add_observation(t_frame.id, point.id)

        cv2_matches.append(cv2.DMatch(q_feat.idx, t_feat.idx, m.distance))
    ctx.cgraph.update_edges()

    # Save the matches
    if DEBUG:
        save_path=results_dir / "loop/matches" / f"{q_frame.id}_{t_frame.id}.png"
        vis.plot_matches(cv2_matches, q_frame, t_frame, save_path = save_path)

    if DEBUG:
        log.info(f"\t Found {len(cv2_matches)} Point Associations!")
    return len(cv2_matches)

def estimate_relative_pose(q_frame: utils.Frame, t_frame: utils.Frame):
    """Estimate the map <-> camera displacement using a 3D-2D PnP approach."""
    q_t_map_pairs = t_frame.get_map_matches_with(q_frame.id)
    num_matches = len(q_t_map_pairs)
    if DEBUG:
        log.info(f"Estimating Map -> Frame #{t_frame.id} pose using {num_matches}/{ctx.map.num_points} map points...")

    # 1) Build 3D <-> 2D correspondences
    q_point_positions = []
    t_img_pxs = []
    T_w2q = utils.invert_transform(q_frame.pose)
    for (feat, map_point) in q_t_map_pairs:
        q_point_pos = T_w2q[:3, :3] @ map_point.pos + T_w2q[:3, 3]
        q_point_positions.append(q_point_pos) # 3D in q_frame coordinates
        t_img_pxs.append(feat.kpt.pt)         # 2D pixel (u, v)

    q_point_positions = np.array(q_point_positions, dtype=np.float64) # (M, 3)
    t_img_pxs = np.array(t_img_pxs, dtype=np.float64)     # (M, 2)

    # 2) solvePnPRansac to get rvec/tvec for world->t_frame
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        q_point_positions,
        t_img_pxs,
        cameraMatrix=K,
        distCoeffs=None,
        reprojectionError=SETTINGS["tracking"]["PnP"]["reprojection_threshold"],
        confidence=SETTINGS["tracking"]["PnP"]["confidence"],
        iterationsCount=SETTINGS["tracking"]["PnP"]["iterations"]
    )
    if not success or inliers is None:
        log.warning("\t solvePnP failed!")
        return False
    if len(inliers) < SETTINGS["loop"]["pnp_inliers"]:
        log.warning("\t solvePnP did not find enough inliers!")
        return False
    inliers = inliers.flatten()

    # Build an inliers mask
    inliers_mask = np.zeros(num_matches, dtype=bool)
    inliers_mask[inliers] = True
    num_tracked_points = inliers_mask.sum()
    if DEBUG:
        log.info(f"\t solvePnPRansac filtered {num_matches - num_tracked_points}/{num_matches} points.")
    
    # 3) Refine the pose using Levenberg-Marquardt on the inlier correspondences.
    rvec, tvec = cv2.solvePnPRefineLM(
        q_point_positions[inliers],
        t_img_pxs[inliers],
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
    projected_world_pxs, _ = cv2.projectPoints(objectPoints=q_point_positions, rvec=rvec, tvec=t_wc, cameraMatrix=K, distCoeffs=None)
    projected_world_pxs = projected_world_pxs.squeeze()
    
    ## Calculate the per-point reprojection error (Euclidean distance)
    errors = np.sqrt(np.sum((t_img_pxs[inliers_mask] - projected_world_pxs[inliers_mask])**2, axis=1))
    
    ## Create a mask for points with error less than the threshold
    reproj_mask = errors < SETTINGS["tracking"]["PnP"]["reprojection_threshold"]
    if DEBUG:
        log.info(f"\t Reprojection:")
        log.info(f"\t\t Median/Mean error ({np.median(errors):.2f}, {np.mean(errors):.2f})")
        log.info(f"\t\t Outliers {len(errors) - reproj_mask.sum()}/{len(errors)} points.")

    ## Visualization
    if DEBUG:
        img_path = results_dir / f"loop/pnp/{q_frame.id}_{t_frame.id}a.png"
        vis.plot_reprojection(t_frame.img, t_img_pxs[~inliers_mask], projected_world_pxs[~inliers_mask], path=img_path)
        img_path = results_dir / f"loop/pnp/{q_frame.id}_{t_frame.id}b.png"
        vis.plot_reprojection(t_frame.img, t_img_pxs[inliers_mask], projected_world_pxs[inliers_mask], path=img_path)

    # 6) Construct T_{q_frame->t_frame}
    T_q2t = np.eye(4, dtype=np.float64)
    T_q2t[:3, :3] = R_wc
    T_q2t[:3, 3] = t_wc

    return T_q2t
