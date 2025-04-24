import numpy as np

from src.others.data import Dataset
from src.others.frame import Frame
from src.others.visualize import plot_trajectory, plot_ground_truth, plot_trajectory_3d, plot_BA, plot_BA2d
from src.others.linalg import invert_transform
from src.others.utils import save_image, delete_subdirectories
from src.others.scale import estimate_depth_scale, validate_scale

from src.initialization.feature_matching import matchFeatures
from src.initialization.initialization import initialize_pose, triangulate_points

from src.tracking.pnp import estimate_relative_pose
from src.tracking.point_association import constant_velocity_model, localPointAssociation
from src.tracking.point_association import mapPointAssociation, bowPointAssociation

from src.place_recognition.bow import load_vocabulary, query_recognition_candidate

from src.local_mapping.keyframe import is_keyframe
from src.local_mapping.map import Map

from src.backend.local_ba import localBA
from src.backend.global_ba import globalBA
from src.backend.pose_optimization import poseBA
from src.backend.single_pose_optimization import singlePoseBA
from src.backend.convisibility_graph import ConvisibilityGraph


from config import main_dir, data_dir, scene, results_dir, SETTINGS, log
log.info(f"\t\tUsing dataset: `{scene}` ...")


"""
Important notes:
- Both the Kitti and OpenCV camera frames follow the same convention:
    - x: right
    - y: down
    - z: forward
"""


debug = SETTINGS["generic"]["debug"]
MIN_ASSOCIATIONS = SETTINGS["point_association"]["num_matches"]
SEARCH_WINDOW_SIZE = SETTINGS["point_association"]["search_window"]


def main():
    use_dist = False
    cleanup = True

    # Clean previous results
    if cleanup:
        delete_subdirectories(results_dir)

    # Read the data
    data = Dataset(data_dir, scene, use_dist)

    # Read the vocabulary and initialize the BoW database
    vocab = load_vocabulary("cv2") # Basically contains 1000 descriptors
    bow_db: dict[list] = {} # contains visual_word_id -> keyframe_that_sees_it dicts
    for i in range(len(vocab)): bow_db[i] = []

    # Plot the ground truth trajectory
    gt = data.ground_truth()
    plot_ground_truth(gt)

    # Initialize the local map
    map = Map()

    # Initialize the convisibility graph
    cgraph = ConvisibilityGraph()

    # Run the main VO loop
    i = -1
    is_initialized = False
    while not data.finished():
        # Advance the iteration
        i+=1
        log.info("")
        log.info("")
        log.info(f"\t Iteration: {i} / {data.length()}")

        # Capture new image frame (current_frame)
        t, img, gt_pose = data.get()

        # Create a frame and extract its ORB features
        t_frame = Frame(i, img)
        t_frame.set_gt(gt_pose)
        t_frame.set_time(t)

        # Iteration #0
        if t_frame.id == 0:
            log.info("")
            log.info("~~~~First Frame~~~~")
            assert np.all(np.eye(4) - gt_pose < 1e-6)

            # Bookkeping
            t_frame.set_pose(gt_pose)
            cgraph.add_first_keyframe(t_frame)
            if debug:
                save_image(t_frame.img, results_dir / "keyframes" / f"{i}_bw.png")
            q_frame = t_frame
        else:                    
            # ########### Initialization ###########
            if not is_initialized:
                log.info("")
                log.info("~~~~Initialization~~~~")
                # Feature matching
                matches = matchFeatures(q_frame, t_frame) # (N) : N < M
                if matches is None:
                    log.info("Feature matching failed!")
                    continue

                # Extract the initial pose using the Essential or Homography matrix (2d-2d)
                matches, T_q2t, is_initialized = initialize_pose(matches, q_frame, t_frame)
                if not is_initialized:
                    log.info("Pose initialization failed!")
                    continue
                T_t2q = invert_transform(T_q2t)

                # Calculate the next pose with scale ambiguity
                T_q2w = q_frame.pose
                T_t2w_unscaled = T_q2w @ T_t2q

                # Estimate the depth scale
                scale = estimate_depth_scale([T_q2w, T_t2w_unscaled], [q_frame.gt, gt_pose])
            
                # Remove scale ambiguity
                T_t2q[:3, 3] *= scale

                # Apply the scale to the pose and validate it
                T_t2w = T_q2w @ T_t2q
                log.info(f"\t RMSE: {np.linalg.norm(gt_pose[:3, 3] - T_t2w[:3, 3]):.2f}")

                # Set the pose in the current frame
                t_frame.set_pose(T_t2w)

                # Triangulate the 3D points using the initial pose
                w_points, q_kpts, t_kpts, q_descriptors, t_descriptors, is_initialized = triangulate_points(matches, T_q2t, q_frame, t_frame, scale)
                if not is_initialized:
                    log.info("Triangulation failed!")
                    continue

                # Push the keyframes and triangulated points to the map
                map.add_keyframe(q_frame)
                map.add_keyframe(t_frame)
                map.add_init_points(w_points, 
                                    q_frame, q_kpts, q_descriptors, 
                                    t_frame, t_kpts, t_descriptors)

                # Add the keyframe to the convisibility graph
                cgraph.add_init_keyframe(t_frame)

                # Validate the scale
                validate_scale([q_frame.pose, t_frame.pose], [q_frame.gt, t_frame.gt])

                # Perform Bundle Adjustment
                prev_pts = map.point_positions().copy()
                ba = globalBA(map, verbose=debug)
                ba.optimize()

                # plot_BA(prev_pts, map.point_positions())
                plot_BA2d(prev_pts, map.point_positions(), i)
                plot_trajectory(map, i)

                tracking_success = True
                q_frame = t_frame
                    
            # ########### Tracking ###########
            else:
                log.info("")
                log.info("~~~~Tracking~~~~")
                if tracking_success:
                    # ########### Track from Previous Frame ###########
                    log.info("Using constant velocity model...")
                    # Predict the new pose based on a constant velocity model
                    T_w2t = constant_velocity_model(t, map.keyframes)
                    T_t2w = invert_transform(T_w2t)
                    t_frame.set_pose(T_t2w)

                    # Match these map points with the current frame
                    num_matches = localPointAssociation(map, q_frame, t_frame, cgraph, theta=15)
                    if num_matches < MIN_ASSOCIATIONS:
                        log.warning(f"Scale-based Point association failed! Only {num_matches} matches found!")
                        num_matches = localPointAssociation(map, q_frame, t_frame, cgraph, search_window=SEARCH_WINDOW_SIZE)
                        if num_matches < MIN_ASSOCIATIONS:
                            log.warning(f"Window-based Point association failed! Only {num_matches} matches found!")
                            map.remove_observation(t_frame.id)
                            tracking_success = False
                            continue

                    # Bookkeeping
                    map.add_keyframe(t_frame)
                    tracking_success = True

                    # Perform pose optimization
                    ba = singlePoseBA(map, t_frame, verbose=debug)
                    ba.optimize()
                else:
                    # ########### Relocalization ###########
                    log.info("Performing Relocalization!")

                    # Compute the BOW representation of the keyframe
                    t_frame.compute_bow(vocab, bow_db)

                    # Find keyframe candidates from the BoW database for global relocalization
                    kf_candidate_ids = query_recognition_candidate(t_frame, bow_db)

                    # Iterate over all candidates
                    log.info(f"Iterating over {len(kf_candidate_ids)} keyframe candidates!")
                    for j, kf_id in enumerate(kf_candidate_ids):
                        # Extract the candidate keyframe
                        cand_frame = map.get_keyframe(kf_id)
                        # Perform point association of its map points with the current frame
                        num_matches = bowPointAssociation(map, cand_frame, t_frame, cgraph)
                        # Estimate the new world pose using PnP (3d-2d)
                        T_w2t, num_tracked_points = estimate_relative_pose(map, t_frame)
                        T_t2w = invert_transform(T_w2t)
                        if T_w2t is not None:
                            log.info(f"Candidate {j}, keyframe {kf_id}: solvePnP success!")

                            # Perform pose optimization
                            ba = poseBA(map, verbose=debug)
                            ba.optimize()

                            # Temporarily set the candidate frame as keyframe
                            map.add_keyframe(cand_frame)
                            t_frame.set_pose(T_t2w)
                            tracking_success = True
                            break
                    
                    # If global relocalization failed, we need to restart initialization
                    if T_w2t is None:
                        tracking_success = False
                        is_initialized = False
                        map.remove_observation(t_frame.id)
                        continue
                
                # ########### Track Local Map ###########
                log.info("")
                log.info("~~~~Local Mapping~~~~")

                # Set the visible mask
                map.view(t_frame)

                # Extract a local map from the map
                local_map = cgraph.create_local_map(t_frame, map)
                
                # Project the local map to the frame and search more correspondances
                mapPointAssociation(local_map, map, t_frame)
                
                # Set the found mask
                map.tracked(t_frame)
                
                # Optimize the camera pose with all the map points found in the frame
                ba = singlePoseBA(map, t_frame, verbose=debug)
                ba.optimize()
    
                # ########### New Keyframe Decision ###########
                log.info("Checking for Keyframe...")
                # Check if this t_frame is a keyframe
                T_w2q = invert_transform(q_frame.pose)
                T_t2q = T_w2q @ T_t2w
                if not is_keyframe(t_frame, map.keyframes, local_map):
                    map.remove_keyframe(t_frame.id)
                    t_frame.reset_pose()
                    continue

                # Set the pose in the current frame
                log.info(f"\t RMSE: {np.linalg.norm(gt_pose[:3, 3] - T_t2w[:3, 3]):.2f}")

                # Save the keyframe
                if debug:
                    save_image(t_frame.img, results_dir / "keyframes" / f"{i}_bw.png")
    
                # ########### New Map Point Creation ###########
                log.info("Creating New Map Points...")

                # Add frame to graph
                cgraph.add_track_keyframe(t_frame)

                # Clean up map points that are not seen anymore
                map.cull_points(cgraph)

                # Create new map points
                map.create_track_points(cgraph, t_frame, map.keyframes, bow_db)

                # Perform Bundle Adjustment
                ba = localBA(t_frame, map, cgraph, verbose=debug)
                ba.optimize()
                plot_trajectory(map, i)

                # Clean up redundant frames
                map.cull_keyframes(t_frame, cgraph)

                q_frame = t_frame            

    # Perform one final optimization
    ba.finalize()

    # Save final map and trajectory
    # plot_trajectory(map, i)
    plot_trajectory_3d(map.keyframes)

if __name__ == "__main__":
    main()