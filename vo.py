import numpy as np

from src.others.data import Dataset
from src.others.frame import Frame
from src.others.visualize import plot_trajectory, plot_ground_truth, plot_trajectory_3d
from src.others.utils import save_image, delete_subdirectories, transform_points, invert_transform

from src.frontend.feature_matching import matchFeatures
from src.frontend.initialization import initialize_pose, triangulate_points
from src.frontend.tracking import estimate_relative_pose, is_keyframe, pointAssociation, triangulateNewPoints
from src.frontend.scale import estimate_depth_scale, validate_scale

from src.others.local_map import Map
# from src.backend.g2o.ba import BA
from src.backend.gtsam.ba import BA


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
    opt_freq = SETTINGS["ba"]["frequency"]

    # Initialize the local map
    map = Map()

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

            ba.add_pose(pose, t_frame.id, fixed=True)
            gt_poses.append(gt_pose)
            poses.append(pose)
            t_frame.set_pose(pose)
            t_frame.set_keyframe(True)
            keyframes.append(t_frame)
            if debug:
                save_image(t_frame.img, results_dir / "keyframes" / f"{i}_bw.png")
                
            plot_trajectory(poses, gt_poses, i)
        else:                    
            # ########### Initialization ###########
            if not is_initialized:
                print("Initialization)")
                # Extract the last keyframe
                q_frame = keyframes[-1]

                # Feature matching
                matches = matchFeatures(q_frame, t_frame, K, "initialization/0-raw") # (N) : N < M

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
                T_tw = poses[-1] @ T_tq

                # Triangulate the 3D points using the initial pose
                t_points, t_kpts, t_descriptors, is_initialized = triangulate_points(q_frame, t_frame, K, scale)
                if not is_initialized:
                    print("Triangulation failed!")
                    continue

                # Transfer the points to the world frame
                points_w = transform_points(t_points, T_tw)

                # Add pose and landmarks to the optimizer
                ba.add_pose(T_tw, t_frame.id)
                ba.add_observations(t_frame.id, points_w, t_kpts)

                # Save the pose and t_frame information
                gt_poses.append(gt_pose)
                poses.append(T_tw)
                t_frame.set_pose(T_tw)
                t_frame.set_keyframe(True)
                keyframes.append(t_frame)

                # Validate the scale
                validate_scale([poses[-1], T_tw], [gt_poses[-1], gt_pose])
               
                # Visualize the current state of the map and trajectory
                plot_trajectory(poses, gt_poses, i)

                # Save the keyframe
                if debug:
                    save_image(t_frame.img, results_dir / "keyframes" / f"{i}_bw.png")

                # Push the triangulated points to the map
                map.add_points(t_frame.id, points_w, t_kpts, t_descriptors, T_tw)

                # Optimizer the poses using BA
                poses, landmark_ids, landmark_poses = ba.optimize()
                map.update_landmarks(landmark_ids, landmark_poses)
                plot_trajectory(poses, gt_poses, i, ba=True)
            # ########### Tracking ###########
            else:
                print("Tracking)")
                q_frame = keyframes[-1]
    
                # Find the map points that can be seen in the previous frame
                T_wq = invert_transform(poses[-1])
                map.view(T_wq, K)
                if map.num_points_in_view < SETTINGS["keyframe"]["num_tracked_points"]:
                    print(f"Not enough points in view ({map.num_points_in_view}).")
                    is_initialized = False
                    continue

                # Match these map points with the current frame
                map_t_pairs = pointAssociation(map, t_frame)
                if len(map_t_pairs) < SETTINGS["point_association"]["num_matches"]:
                    print("Point association failed!")
                    is_initialized = False
                    continue

                # Estimate the new world pose using PnP (3d-2d)
                T_wt, num_tracked_points = estimate_relative_pose(map, t_frame, map_t_pairs, K)
                if T_wt is None:
                    print(f"Warning: solvePnP failed!")
                    is_initialized = False
                    continue
                T_tw = invert_transform(T_wt)
    
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
                    save_image(t_frame.img, results_dir / "keyframes" / f"{i}_bw.png")

                # Do feature matching with the previous keyframe
                q_frame = keyframes[-2]
                matches = matchFeatures(q_frame, t_frame, K, "mapping/0-raw")
                q_frame.match[t_frame.id]["T"] = T_qt
                t_frame.match[q_frame.id]["T"] = T_tq

                # Find new keypoints and triangulate them
                (t_old_points, old_kpts, old_descriptors, 
                 t_new_points, new_kpts, new_descriptors, 
                 new_points_success) = triangulateNewPoints(q_frame, t_frame, map, K)
                if new_points_success:
                    # Transfer the new points to the world frame
                    w_new_points = transform_points(t_new_points, T_tw)
                    w_old_points = transform_points(t_old_points, T_tw)

                    # Add the triangulated points to the local map
                    map.add_points(t_frame.id, w_new_points, new_kpts, new_descriptors, T_tw)

                    # Update the old triangulated points
                    map.update_points(t_frame.id, old_kpts, old_descriptors, T_tw)

                    # Add landmarks and observations to the optimizer
                    ba.add_observations(t_frame.id, w_old_points, old_kpts)
                    ba.add_observations(t_frame.id, w_new_points, new_kpts)

                # Optimizer the poses using BA
                plot_trajectory(poses, gt_poses, i)
                poses, landmark_ids, landmark_poses = ba.optimize()
                map.update_landmarks(landmark_ids, landmark_poses)
                plot_trajectory(poses, gt_poses, i, ba=True)

                # Clean up map points that are not seen anymore
                removed_landmarks = map.cull()
                ba.cull(removed_landmarks)

    # Perform one final optimization
    poses = ba.finalize()

    # Save final map and trajectory
    plot_trajectory(poses, gt_poses, i)
    plot_trajectory_3d(poses)

if __name__ == "__main__":
    main()