import numpy as np

from src.others.data import Dataset
from src.others.frame import Frame
from src.others.visualize import plot_trajectory, plot_ground_truth, plot_trajectory_3d
from src.others.utils import save_image, delete_subdirectories, transform_points, invert_transform

from src.frontend.feature_tracking import matchFeatures
from src.frontend.initialization import initialize_pose, triangulate_points
from src.frontend.tracking import estimate_relative_pose, is_keyframe, predictPose, pointAssociation, triangulateNewPoints
from src.frontend.scale import estimate_depth_scale, validate_scale

from src.others.local_map import Map
from src.backend.full_ba import BA


from config import main_dir, data_dir, scene, results_dir, debug, SETTINGS
print(f"\t\tUsing dataset: `{scene}` ...")


"""
Important notes:
- Both the Kitti and OpenCV camera frames follow the same convention:
    - x: right
    - y: down
    - z: forward
"""


def main():
    use_dist = False
    cleanup = True

    # Clean previous results
    if cleanup:
        delete_subdirectories(results_dir)

    # Read the data
    data = Dataset(data_dir, scene, use_dist)

    # Plot the ground truth trajectory
    gt = data.ground_truth()
    plot_ground_truth(gt)

    # Get the camera matrix
    K = data.get_intrinsics()

    # Initialize the BA optimizer
    ba = BA(K)

    # Initialize the graph, the keyframes, and the bow list
    frames = []
    keyframes = []
    poses = []
    gt_poses = []

    # Run the main VO loop
    i = -1
    is_initialized = False
    while not data.finished():
        # Advance the iteration
        i+=1
        # if i%50 == 0:
        print(f"\n\tIteration: {i} / {data.length()}")

        # Capture new image frame (current_frame)
        t, img, gt_pose = data.get()

        # Create a frame and extract its ORB features
        t_frame = Frame(i, img)
        frames.append(t_frame)

        # Iteration #0
        if t_frame.id == 0:
            print("First iteration)")
            pose = gt_pose
            assert np.all(np.eye(4) - pose < 1e-6)

            ba.add_first_pose(pose, t_frame.id)
            gt_poses.append(gt_pose)
            poses.append(pose)
            t_frame.set_pose(pose)
            t_frame.set_keyframe(True)
            keyframes.append(t_frame)
            if debug:
                save_image(t_frame.img, results_dir / "keyframes" / f"{i}_rgb.png")
                
            plot_trajectory(poses, gt_poses, i)
        else:                    
            # ########### Initialization ###########
            if not is_initialized:
                print("Initialization)")
                # Extract the last keyframe
                q_frame = keyframes[-1]

                # Feature matching
                matches = matchFeatures(q_frame, t_frame, K, "0-raw") # (N) : N < M

                # Check if there are enough matches
                if len(matches) < SETTINGS["matches"]["min"]:
                    print("Not enough matches!")
                    continue

                # Extract the initial pose using the Essential or Homography matrix (2d-2d)
                T_qt, is_initialized = initialize_pose(q_frame, t_frame, K)
                if not is_initialized:
                    print("Pose initialization failed!")
                    continue
                assert np.linalg.norm(T_qt[:3, 3]) - 1 < 1e-6
                T_tq = invert_transform(T_qt)

                # Calculate the next pose with scale ambiguity
                unscaled_pose = poses[-1] @ T_tq

                # Estimate the depth scale
                scale = estimate_depth_scale([poses[-1], unscaled_pose], [gt_poses[-1], gt_pose])
            
                # Remove scale ambiguity
                T_tq[:3, 3] *= scale

                # Apply the scale to the pose and validate it
                scaled_pose = poses[-1] @ T_tq

                # Triangulate the 3D points using the initial pose
                t_points, t_kpts, t_descriptors, is_initialized = triangulate_points(q_frame, t_frame, K, scale)
                if not is_initialized:
                    print("Triangulation failed!")
                    continue

                # Transfer the points to the world frame
                points_w = transform_points(t_points, scaled_pose)

                # Add pose and landmarks to the optimizer
                ba.add_pose(scaled_pose, t_frame.id)
                ba.add_landmarks(points_w, t_kpts)
                ba.add_observations(t_frame.id, t_kpts)

                # Save the pose and t_frame information
                gt_poses.append(gt_pose)
                poses.append(scaled_pose)
                t_frame.set_pose(scaled_pose)
                t_frame.set_keyframe(True)
                keyframes.append(t_frame)

                # Validate the scale
                validate_scale([poses[-1], scaled_pose], [gt_poses[-1], gt_pose])
               
                # Visualize the current state of the map and trajectory
                plot_trajectory(poses, gt_poses, i)

                # Save the keyframe
                if debug:
                    save_image(t_frame.img, results_dir / "keyframes" / f"{i}_bw.png")

                # Create a local map and push the triangulated points
                map = Map(q_frame.id)
                map.add_points(points_w, t_kpts, t_descriptors, scaled_pose)
            # ########### Tracking ###########
            else:
                print("Tracking)")
                q_frame = keyframes[-1]
                    
                # Predict next pose using constant velocity
                pred_pose = predictPose(poses)
                T_wp = invert_transform(pred_pose)
    
                # Find the map points that can be seen in the predicted robot's pose
                map.view(T_wp, K, pred=True)
                if map.num_points_in_view < SETTINGS["keyframe"]["num_tracked_points"]:
                    print(f"Not enough points in view ({map.num_points_in_view}).")
                    is_initialized = False
                    continue

                # Search the map point of the new frame in a window around the projected location
                associations_found = False
                search_window = SETTINGS["guided_search"]["search_window"]
                while not associations_found:
                    map_kpt_dist_pairs = pointAssociation(map, t_frame, T_wp, search_window)
                    if len(map_kpt_dist_pairs) > SETTINGS["guided_search"]["num_associations"]:
                        associations_found = True
                    else:
                        # Keep increasing the search window if associations are not found
                        search_window *= 2
                        print(f"Not enough point associations ({len(map_kpt_dist_pairs)}).",
                              f"Increasing window size to {search_window}!")
                        if search_window > 2000:
                            break
                if not associations_found:
                    print("Point association failed!")
                    is_initialized = False
                    continue

                # Estimate the new world pose using PnP (3d-2d)
                T_wt, num_tracked_points = estimate_relative_pose(map, t_frame, map_kpt_dist_pairs, K)
                if T_wt is None:
                    print(f"Warning: solvePnP failed!")
                    is_initialized = False
                    continue
                T_tw = invert_transform(T_wt)
    
                # Find the map points that can be seen in the actual robot's pose
                map.view(T_wt, K, pred=True)
            
                # Check if this t_frame is a keyframe
                T_wq = invert_transform(poses[-1])
                T_tq = T_wq @ T_tw
                if not is_keyframe(T_tq, num_tracked_points):
                    continue

                # Add the new pose to the optimizer
                ba.add_pose(T_tw, t_frame.id)
                
                # Save the T_tw and t_frame information
                gt_poses.append(gt_pose)
                poses.append(T_tw)
                t_frame.set_pose(T_tw)
                t_frame.set_keyframe(True)

                # Save the keyframe
                keyframes.append(t_frame)
                if debug:
                    save_image(t_frame.img, results_dir / "keyframes" / f"{i}_rgb.png")

                # Do feature matching with the previous keyframe
                q_frame = keyframes[-2]
                matches = matchFeatures(q_frame, t_frame, K, "5-raw")
                q_frame.match[t_frame.id]["T"] = T_qt
                t_frame.match[q_frame.id]["T"] = T_tq

                # Find new keypoints and triangulate them
                t_points, t_kpts, t_descriptors, new_points_success = triangulateNewPoints(q_frame, t_frame, map, K)
                if new_points_success:
                    # Transfer the points to the world frame
                    points_w = transform_points(t_points, T_tw)

                    # Add the triangulated points to the local map
                    map.add_points(points_w, t_kpts, t_descriptors, T_tw)

                    # Add landmarks and observations to the optimizer
                    ba.add_landmarks(points_w, t_kpts)
                    ba.add_observations(t_frame.id, t_kpts)

                # Clean up map points that are not seen anymore
                map.cull()

                # Optimizer the poses using BA
                plot_trajectory(poses, gt_poses, i)
                poses, _ = ba.optimize()
                plot_trajectory(poses, gt_poses, i, save_path=results_dir / "trajectory_ba")

    # Save final map and trajectory
    plot_trajectory(poses, gt_poses, i)
    plot_trajectory_3d(poses)

if __name__ == "__main__":
    main()