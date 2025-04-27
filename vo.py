import numpy as np

import src.utils as utils
import src.visualization as vis
import src.initialization as init
import src.tracking as track
import src.place_recognition as pr
import src.local_mapping as mapping
import src.backend as backend
import src.globals as ctx

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
        utils.delete_subdirectories(results_dir)

    # Read the data
    data = utils.Dataset(data_dir, scene, use_dist)

    # Plot the ground truth trajectory
    gt = data.ground_truth()
    vis.plot_ground_truth(gt)

    ## Setup Globals
    # Read the vocabulary and initialize the BoW database
    ctx.vocab = pr.load_vocabulary("cv2").astype(np.uint8) # Basically contains 1000 descriptors
    ctx.bow_db = {} # contains visual_word_id -> keyframe_that_sees_it dicts
    for i in range(len(ctx.vocab)): ctx.bow_db[i] = []

    # Initialize map and convisibility graph
    ctx.map = mapping.Map()
    ctx.cgraph = backend.ConvisibilityGraph()

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
        t_frame = utils.Frame(i, img)
        t_frame.set_gt(gt_pose)
        t_frame.set_time(t)

        # Iteration #0
        if t_frame.id == 0:
            log.info("")
            log.info("~~~~First utils.Frame~~~~")
            assert np.all(np.eye(4) - gt_pose < 1e-6)

            # Bookkeping
            t_frame.set_pose(gt_pose)
            ctx.cgraph.add_first_keyframe(t_frame)
            if debug:
                utils.save_image(t_frame.img, results_dir / "keyframes" / f"{i}_bw.png")
            q_frame = t_frame
        else:                    
            # ########### Initialization ###########
            if not is_initialized:
                log.info("")
                log.info("~~~~Initialization~~~~")
                # Feature matching
                matches = init.matchFeatures(q_frame, t_frame) # (N) : N < M
                if matches is None:
                    log.info("Feature matching failed!")
                    continue

                # Extract the initial pose using the Essential or Homography matrix (2d-2d)
                matches, T_q2t, is_initialized = init.estimate_pose(matches, q_frame, t_frame)
                if not is_initialized:
                    log.info("Pose initialization failed!")
                    continue
                T_t2q = utils.invert_transform(T_q2t)

                # Calculate the next pose with scale ambiguity
                T_q2w = q_frame.pose
                T_t2w_unscaled = T_q2w @ T_t2q

                # Estimate the depth scale
                scale = utils.estimate_depth_scale([T_q2w, T_t2w_unscaled], [q_frame.gt, gt_pose])
            
                # Remove scale ambiguity
                T_t2q[:3, 3] *= scale

                # Apply the scale to the pose and validate it
                T_t2w = T_q2w @ T_t2q
                log.info(f"\t RMSE: {np.linalg.norm(gt_pose[:3, 3] - T_t2w[:3, 3]):.2f}")

                # Set the pose in the current frame
                t_frame.set_pose(T_t2w)

                # Triangulate the 3D points using the initial pose
                w_points, q_kpts, t_kpts, q_descriptors, t_descriptors, is_initialized = init.triangulate_points(matches, T_q2t, q_frame, t_frame, scale)
                if not is_initialized:
                    log.info("Triangulation failed!")
                    continue

                # Push the keyframes and triangulated points to the map
                ctx.map.add_keyframe(q_frame)
                ctx.map.add_keyframe(t_frame)
                ctx.map.add_init_points(w_points, 
                                    q_frame, q_kpts, q_descriptors, 
                                    t_frame, t_kpts, t_descriptors)

                # Add the keyframe to the convisibility graph
                ctx.cgraph.add_init_keyframe(t_frame)

                # Compute the BOW representation for both keyframes
                q_frame.compute_bow()
                t_frame.compute_bow()

                # Validate the scale
                utils.validate_scale([q_frame.pose, t_frame.pose], [q_frame.gt, t_frame.gt])

                # Perform Bundle Adjustment
                ba = backend.globalBA(verbose=debug)
                ba.optimize()

                # plot_BA()
                vis.plot_BA2d(i)
                vis.plot_trajectory(i)

                tracking_success = True
                q_frame = t_frame
                    
            # ########### Tracking ###########
            else:
                log.info("")
                log.info("~~~~Tracking~~~~")
                if tracking_success:
                    # ########### Track from Previous utils.Frame ###########
                    log.info("Using constant velocity model...")
                    # Predict the new pose based on a constant velocity model
                    T_w2t = track.constant_velocity_model(t, ctx.map.keyframes)
                    T_t2w = utils.invert_transform(T_w2t)
                    t_frame.set_pose(T_t2w)

                    # Match these map points with the current frame
                    num_matches = track.localPointAssociation(q_frame, t_frame, theta=15)
                    if num_matches < MIN_ASSOCIATIONS:
                        log.warning(f"Scale-based Point association failed! Only {num_matches} matches found!")
                        num_matches = track.localPointAssociation(q_frame, t_frame, search_window=SEARCH_WINDOW_SIZE)
                        if num_matches < MIN_ASSOCIATIONS:
                            log.warning(f"Window-based Point association failed! Only {num_matches} matches found!")
                            ctx.map.remove_observation(t_frame.id)
                            tracking_success = False
                            continue

                    # Bookkeeping
                    ctx.map.add_keyframe(t_frame)
                    tracking_success = True

                    # Compute the BOW representation of the keyframe
                    t_frame.compute_bow()

                    # Perform pose optimization
                    ba = backend.singlePoseBA(t_frame, verbose=debug)
                    ba.optimize()
                else:
                    # ########### Relocalization ###########
                    log.info("Performing Relocalization!")

                    # Compute the BOW representation of the keyframe
                    t_frame.compute_bow()

                    # Find keyframe candidates from the BoW database for global relocalization
                    kf_candidate_ids = pr.query_recognition_candidate(t_frame)

                    # Iterate over all candidates
                    log.info(f"Iterating over {len(kf_candidate_ids)} keyframe candidates!")
                    for j, kf_id in enumerate(kf_candidate_ids):
                        # Extract the candidate keyframe
                        cand_frame = ctx.map.get_keyframe(kf_id)
                        # Perform point association of its map points with the current frame
                        num_matches = track.bowPointAssociation(cand_frame, t_frame)
                        # Estimate the new world pose using PnP (3d-2d)
                        T_w2t, num_tracked_points = track.estimate_relative_pose(t_frame)
                        T_t2w = utils.invert_transform(T_w2t)
                        if T_w2t is not None:
                            log.info(f"Candidate {j}, keyframe {kf_id}: solvePnP success!")

                            # Perform pose optimization
                            ba = backend.poseBA(verbose=debug)
                            ba.optimize()

                            # Temporarily set the candidate frame as keyframe
                            ctx.map.add_keyframe(cand_frame)
                            t_frame.set_pose(T_t2w)
                            t_frame.relocalization = True
                            tracking_success = True
                            break
                    
                    # If global relocalization failed, we need to restart initialization
                    if T_w2t is None:
                        tracking_success = False
                        is_initialized = False
                        ctx.map.remove_observation(t_frame.id)
                        continue
                
                # ########### Track Local mapping.Map ###########
                log.info("")
                log.info("~~~~Local Mapping~~~~")

                # Set the visible mask
                ctx.map.view(t_frame)

                # Extract a local map from the map
                ctx.local_map = ctx.cgraph.create_local_map(t_frame)
                
                # Project the local map to the frame and search more correspondances
                track.mapPointAssociation(t_frame)
                
                # Set the found mask
                ctx.map.tracked(t_frame)
                
                # Optimize the camera pose with all the map points found in the frame
                ba = backend.singlePoseBA(t_frame, verbose=debug)
                ba.optimize()
    
                # ########### New Keyframe Decision ###########
                log.info("Checking for Keyframe...")
                # Check if this t_frame is a keyframe
                T_w2q = utils.invert_transform(q_frame.pose)
                T_t2q = T_w2q @ T_t2w
                if not mapping.is_keyframe(t_frame):
                    ctx.map.remove_keyframe(t_frame.id)
                    t_frame.reset_pose()
                    continue

                # Set the pose in the current frame
                log.info(f"\t RMSE: {np.linalg.norm(gt_pose[:3, 3] - T_t2w[:3, 3]):.2f}")

                # Save the keyframe
                if debug:
                    utils.save_image(t_frame.img, results_dir / "keyframes" / f"{i}_bw.png")
    
                # ########### New mapping.Map Point Creation ###########
                log.info("Creating New mapping.Map Points...")

                # Add frame to graph
                ctx.cgraph.add_track_keyframe(t_frame)

                # # Compute the BOW representation of the keyframe
                # t_frame.compute_bow() # This here is only useful if there was no relocalization

                # Clean up map points that are not seen anymore
                ctx.map.cull_points()

                # Create new map points
                ctx.map.create_track_points(t_frame)

                # Perform Bundle Adjustment
                ba = backend.localBA(t_frame, verbose=debug)
                ba.optimize()
                vis.plot_trajectory(i)

                # Clean up redundant frames
                ctx.map.cull_keyframes(t_frame)

                q_frame = t_frame            

    # Perform one final optimization
    ba.finalize()

    # Save final map and trajectory
    # vis.plot_trajectory(i)
    vis.plot_trajectory_3d()

if __name__ == "__main__":
    main()