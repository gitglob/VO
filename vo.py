import numpy as np

from src.others.data import Dataset
from src.others.frame import Frame
from src.others.visualize import plot_trajectory, plot_ground_truth, plot_trajectory_3d
from src.others.utils import save_image, delete_subdirectories, transform_points, invert_transform

from src.frontend.feature_matching import matchFeatures
from src.frontend.initialization import initialize_pose, triangulate_points
from src.frontend.tracking import estimate_relative_pose, is_keyframe, triangulateNewPoints
from src.frontend.point_association import constant_velocity_model, localPointAssociation
from src.frontend.point_association import mapPointAssociation, bowPointAssociation
from src.frontend.scale import estimate_depth_scale, validate_scale

from src.place_recognition.bow import load_vocabulary, query_recognition_candidate

from src.others.local_map import Map
from src.backend.g2o.ba import BA
from src.backend.g2o.pose_optimization import poseBA
from src.backend.convisibility_graph import ConvisibilityGraph


from config import main_dir, data_dir, scene, results_dir, SETTINGS
print(f"\t\tUsing dataset: `{scene}` ...")


"""
Important notes:
- Both the Kitti and OpenCV camera frames follow the same convention:
    - x: right
    - y: down
    - z: forward
"""


debug = SETTINGS["generic"]["debug"]
BA_FREQ = SETTINGS["ba"]["frequency"]
MIN_ASSOCIATIONS = SETTINGS["point_association"]["num_matches"]
SEARCH_WINDOW_SIZE = SETTINGS["point_association"]["search_window"]


def main():
    use_dist = False
    cleanup = True
    use_loop_closures = False

    # Clean previous results
    if cleanup:
        delete_subdirectories(results_dir)

    # Read the data
    data = Dataset(data_dir, scene, use_dist)

    # Read the vocabulary and initialize the BoW database
    if use_loop_closures:
        vocab = load_vocabulary("dbow")
        bow_db: list[dict] = [] # contains visual_word_id -> keyframe_that_sees_it dicts

    # Plot the ground truth trajectory
    gt = data.ground_truth()
    plot_ground_truth(gt)

    # Get the camera matrix
    K = data.get_intrinsics()

    # Initialize the local map
    map = Map()

    # Initialize the convisibility graph
    cgraph = ConvisibilityGraph()

    # Initialize the graph, the keyframes, and the bow list
    keyframes = {} # keyframe_id -> keyframe 
    frames = []
    gt_poses = []
    times = []

    # Run the main VO loop
    i = -1
    is_initialized = False
    while not data.finished():
        # Advance the iteration
        i+=1
        print(f"\n\tIteration: {i} / {data.length()}")

        # Capture new image frame (current_frame)
        t, img, gt_pose = data.get()

        # Create a frame and extract its ORB features
        t_frame = Frame(i, img, is_initialized)
        frames.append(t_frame)

        # Iteration #0
        if t_frame.id == 0:
            print("First iteration)")
            pose = gt_pose
            assert np.all(np.eye(4) - pose < 1e-6)

            # Bookkeping
            times.append(t)
            gt_poses.append(gt_pose)
            t_frame.set_pose(pose)
            t_frame.set_keyframe(True)
            keyframes[i] = t_frame
            cgraph.add_keyframe(t_frame, map)
            if debug:
                save_image(t_frame.img, results_dir / "keyframes" / f"{i}_bw.png")
                
            plot_trajectory(keyframes, gt_poses, i)
            q_frame = t_frame
        else:                    
            # ########### Initialization ###########
            if not is_initialized:
                print("Initialization)")
                num_tracking_fails = 0

                # Feature matching
                matches = matchFeatures(q_frame, t_frame, "initialization/0-raw") # (N) : N < M

                # Check if there are enough matches
                if len(matches) < SETTINGS["matches"]["min"]:
                    print("Not enough matches!")
                    continue

                # Extract the initial pose using the Essential or Homography matrix (2d-2d)
                T_qt, is_initialized = initialize_pose(q_frame, t_frame)
                if not is_initialized:
                    print("Pose initialization failed!")
                    continue
                assert np.linalg.norm(T_qt[:3, 3]) - 1 < 1e-6
                T_t2q = invert_transform(T_qt)

                # Calculate the next pose with scale ambiguity
                T_q2w = q_frame.pose
                T_t2w_unscaled = T_q2w @ T_t2q

                # Estimate the depth scale
                scale = estimate_depth_scale([T_q2w, T_t2w_unscaled], [gt_poses[-1], gt_pose])
            
                # Remove scale ambiguity
                T_t2q[:3, 3] *= scale
                # if not is_keyframe(T_t2q):
                #     is_initialized = False
                #     continue

                # Apply the scale to the pose and validate it
                T_t2w = T_q2w @ T_t2q

                # Set the pose in the current frame
                t_frame.set_pose(T_t2w)
                t_frame.set_keyframe(True)

                # Add the keyframe to the convisibility graph
                cgraph.add_keyframe(t_frame, map)

                # Triangulate the 3D points using the initial pose
                t_points, t_kpts, t_descriptors, is_initialized = triangulate_points(q_frame, t_frame, scale)
                if not is_initialized:
                    print("Triangulation failed!")
                    continue

                # Transfer the points to the world frame
                points_w = transform_points(t_points, T_t2w)

                # Push the triangulated points to the map
                map.add_points(t_frame, points_w, t_kpts, t_descriptors)

                # Bookkeping
                times.append(t)
                gt_poses.append(gt_pose)
                keyframes[i] = t_frame
                cgraph.add_keyframe(t_frame, map)

                # Validate the scale
                validate_scale([q_frame.pose, t_frame.pose], [gt_poses[-1], gt_pose])
               
                # Visualize the current state of the map and trajectory
                plot_trajectory(keyframes, gt_poses, i)

                # Perform Bundle Adjustment
                ba = BA(K, verbose=debug)
                ba.add_frames(keyframes)
                ba.add_observations(map)
                _, opt_poses, landmark_ids, landmark_poses, ba_success = ba.optimize()

                if ba_success:
                    map.update_landmarks(landmark_ids, landmark_poses)
                    plot_trajectory(keyframes, gt_poses, i, ba_poses=opt_poses)
                tracking_success = True
                q_frame = t_frame
                    
            # ########### Tracking ###########
            else:
                print("Tracking)")    
                if tracking_success:
                    # ########### Track from Previous Frame ###########
                    # Predict the new pose based on a constant velocity model
                    T_w2t = constant_velocity_model(t, times, keyframes)

                    # Match these map points with the current frame
                    map_t_pairs = localPointAssociation(map, t_frame, T_w2t, theta=15)
                    if len(map_t_pairs) < MIN_ASSOCIATIONS:
                        print(f"Scale-based Point association failed! Only {len(map_t_pairs)} matches found!")
                        map_t_pairs = localPointAssociation(map, t_frame, T_w2t, search_window=SEARCH_WINDOW_SIZE)
                        if len(map_t_pairs) < MIN_ASSOCIATIONS:
                            print(f"Window-based Point association failed! Only {len(map_t_pairs)} matches found!")
                            is_initialized = False
                            tracking_success = False
                            continue

                    # Perform pose optimization
                    ba = BA(K, verbose=debug)
                    ba.add_frames(keyframes)
                    ba.add_observations(map)
                    ba.optimize()

                    # Bookkeeping
                    keyframes[i] = cand_frame
                    tracking_success = True
                else:
                    # ########### Relocalization ###########
                    print("Performing Relocalization!")

                    # Compute the BOW representation of the keyframe
                    t_frame.compute_bow(vocab, bow_db)

                    # Find keyframe candidates from the BoW database for global relocalization
                    kf_candidate_ids = query_recognition_candidate(t_frame, bow_db)

                    # Iterate over all candidates
                    print(f"Iterating over {len(kf_candidate_ids)} keyframe candidates!")
                    for j, kf_id in enumerate(kf_candidate_ids):
                        # Extract the candidate keyframe
                        cand_frame = keyframes[kf_id]
                        # Perform point association of its map points with the current frame
                        map_t_pairs = bowPointAssociation(map, cand_frame, t_frame, cgraph)
                        # Estimate the new world pose using PnP (3d-2d)
                        T_w2t, num_tracked_points = estimate_relative_pose(map, t_frame, map_t_pairs)
                        if T_w2t is not None:
                            print(f"Candidate {j}, keyframe {kf_id}: solvePnP success!")

                            # Temporarily set the candidate frame as keyframe
                            keyframes[i] = cand_frame

                            # Perform pose optimization
                            ba = poseBA(verbose=debug)
                            ba.add_frames(keyframes)
                            ba.add_observations(map)
                            ba.optimize()

                            tracking_success = True
                            break
                    
                    # If global relocalization failed, we need to restart initialization
                    if T_w2t is None:
                        tracking_success = False
                        is_initialized = False
                        continue
                
                # ########### Track Local Map ###########
                print("Tracking local map...")

                # Extract a local map from the map
                local_map = cgraph.create_local_map()

                # Project the local map to the frame and search more correspondances
                map_t_pairs = mapPointAssociation(map_t_pairs, local_map, t_frame)

                # Optimize the camera pose with all the map points found in the frame
                ba = poseBA(verbose=debug)
                ba.add_frames(keyframes)
                ba.add_observations(map)
                ba.optimize()
    
                # Check if this t_frame is a keyframe
                T_w2q = invert_transform(q_frame.pose)
                T_t2w = invert_transform(T_w2t)
                T_t2q = T_w2q @ T_t2w
                if not is_keyframe(T_t2q, num_tracked_points):
                    del keyframes[i]
                    continue

                # Set the pose in the current frame
                t_frame.set_pose(T_t2w)
                t_frame.set_keyframe(True)

                # Add the keyframe to the convisibility graph
                cgraph.add_keyframe(t_frame, map)

                # Update the map observation and match counters
                map.update_counters()

                # Save the keyframe
                if debug:
                    save_image(t_frame.img, results_dir / "keyframes" / f"{i}_bw.png")

                # Do feature matching with the previous keyframe
                q_frame = list(keyframes.values())[-2]
                matches = matchFeatures(q_frame, t_frame, "mapping/0-raw")
                q_frame.match[t_frame.id]["T"] = T_qt
                t_frame.match[q_frame.id]["T"] = T_t2q

                # Find new keypoints and triangulate them
                (t_old_points, old_kpts, old_descriptors, 
                 t_new_points, new_kpts, new_descriptors, 
                 new_points_success) = triangulateNewPoints(q_frame, t_frame, map)
                if new_points_success:
                    # Transfer the new points to the world frame
                    w_new_points = transform_points(t_new_points, T_t2w)

                    # Add the triangulated points to the local map
                    map.add_points(t_frame, w_new_points, new_kpts, new_descriptors)

                    # Update the old triangulated points
                    map.update_points(t_frame.id, old_kpts, old_descriptors)

                # Bookkeping
                times.append(t)
                gt_poses.append(gt_pose)
                cgraph.add_keyframe(t_frame, map)

                # Plot trajectory
                plot_trajectory(keyframes, gt_poses, i)

                # Optimize the keyframe poses using BA
                if (len(keyframes)-1) % BA_FREQ == 0:
                    # Perform Bundle Adjustment
                    ba = BA(K, verbose=debug)
                    ba.add_frames(keyframes)
                    ba.add_observations(map)
                    landmark_ids, landmark_poses, ba_success = ba.optimize()

                    if ba_success:
                        map.update_landmarks(landmark_ids, landmark_poses)
                        plot_trajectory(keyframes, gt_poses, i, ba_poses=opt_poses)

                # Clean up map points that are not seen anymore
                map.cull()

                q_frame = t_frame            

    # Perform one final optimization
    ba.finalize()

    # Save final map and trajectory
    plot_trajectory(keyframes, gt_poses, i, ba_poses=opt_poses)
    plot_trajectory_3d(keyframes)

if __name__ == "__main__":
    main()