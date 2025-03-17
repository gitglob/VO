import numpy as np

from src.data import Dataset

from src.frame import Frame

from src.frontend.feature_tracking import match_features
from src.frontend.initialization import initialize_pose, triangulate_points
from src.frontend.tracking import estimate_relative_pose, is_keyframe, predict_pose_constant_velocity, guided_descriptor_search, get_new_triangulated_points

from src.frontend.scale import estimate_depth_scale, validate_scale

from src.backend.local_map import Map
from src.backend import optimization

from src.visualize import plot_trajectory, plot_ground_truth, plot_trajectory_3d
from src.utils import save_image, delete_subdirectories, transform_points, invert_transform, get_yaw

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

    # Initialize the graph, the keyframes, and the bow list
    frames = []
    keyframes = []
    poses = []
    gt_poses = []

    # Run the main VO loop
    i = -1
    is_initialized = False
    is_scale_initialized = False
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
            pose = gt_pose
            assert np.all(np.eye(4) - pose < 1e-6)

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
                # Extract the last keyframe
                q_frame = keyframes[-1]

                # Feature matching
                matches = match_features(q_frame, t_frame, K, "0-raw") # (N) : N < M

                # Check if there are enough matches
                if len(matches) < SETTINGS["matches"]["min"]:
                    print("Not enough matches!")
                    continue

                # Extract the initial pose using the Essential or Homography matrix (2d-2d)
                T_qt, is_initialized = initialize_pose(q_frame, t_frame, K)
                # if not is_initialized:
                #     print("Pose initialization failed!")
                #     continue
                # assert np.linalg.norm(T_tq[:3, 3]) - 1 < 1e-6
                # T_tq = inevert_transform(T_qt)
                T_tw = gt_pose
                T_qw = gt_poses[-1]
                T_wq = invert_transform(T_qw)
                T_qt = T_wq @ T_tw
                T_tq = invert_transform(T_qt)
                q_frame.match[t_frame.id]["T"] = T_qt
                t_frame.match[q_frame.id]["T"] = T_tq

                # Calculate the next pose with scale ambiguity
                unscaled_pose = poses[-1] @ T_tq

                # Estimate the depth scale
                scale = estimate_depth_scale([poses[-1], unscaled_pose], [gt_poses[-1], gt_pose])
            
                # Remove scale ambiguity
                T_tq[:3, 3] *= scale

                # Apply the scale to the pose and validate it
                # scaled_pose = poses[-1] @ T_tq
                scaled_pose = gt_pose

                # Triangulate the 3D points using the initial pose
                t_points, t_point_ids, is_initialized = triangulate_points(q_frame, t_frame, K, scale)
                if not is_initialized:
                    print("Triangulation failed!")
                    continue

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

                # Transfer the points to the world frame
                points_w = transform_points(t_points, scaled_pose)

                # Create a local map and push the triangulated points
                map = Map(q_frame.id)
                map.add_initialization_points(points_w, t_point_ids, q_frame, t_frame)
            # ########### Tracking ###########
            else:
                q_frame = keyframes[-1]
                    
                # Predict next pose using constant velocity
                # pred_pose = predict_pose_constant_velocity(poses)
                pred_pose = gt_pose
    
                # Find the map points that can be seen in the predicted robot's pose
                pred_map_points_w, pred_map_pixels, pred_map_descriptors = map.get_points_in_view(pred_pose, K)
                # Check if enough map points are still visible
                if pred_map_points_w is None or len(pred_map_points_w) < 6:
                    print(f"Not enough points in view ({len(pred_map_points_w)})")
                    is_initialized = False
                    continue

                # Compare the map point’s descriptor to the descriptors of the new frame in a small search window around the projected location
                map_kpt_dist_pairs = guided_descriptor_search(pred_map_pixels, pred_map_descriptors, t_frame)
                # Check if enough descriptor matches were found
                if len(map_kpt_dist_pairs) < 6:
                    print(f"Not enough guided descriptor matches ({len(map_kpt_dist_pairs)})")
                    is_initialized = False
                    continue

                # Estimate the new world pose using PnP (3d-2d)
                T_wt = estimate_relative_pose(pred_map_points_w, t_frame, map_kpt_dist_pairs, K)
                if T_wt is None:
                    print(f"Warning: solvePnP failed!")
                    is_initialized = False
                    continue
                T_tw = invert_transform(T_wt)
            
                # Check if this t_frame is a keyframe
                T_qw = poses[-1]
                T_wq = invert_transform(T_qw)
                T_tq = T_wq @ T_tw
                T_qt = invert_transform(T_tq)
                if not is_keyframe(T_tq):
                    continue

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
                matches = match_features(q_frame, t_frame, K, "4-tracking")
                q_frame.match[t_frame.id]["T"] = T_qt
                t_frame.match[q_frame.id]["T"] = T_tq

                # Get inliers by Epipolar constraint
                t_points, point_ids, new_points_success = get_new_triangulated_points(q_frame, t_frame, map, K)
                if not new_points_success:
                    print("Searching for new triangulation points failed!")
                    continue

                # Transfer the points to the world frame
                points_w = transform_points(t_points, T_tw)

                # Create a local map and push the triangulated points
                map.add_tracking_points(points_w, point_ids, q_frame, t_frame)

                # Clean up map points that are not seen anymore
                map.cleanup(T_wt, K)
                
                # Visualize the current state of the map and trajectory
                plot_trajectory(poses, gt_poses, i)

    # Save final map and trajectory
    plot_trajectory(poses, gt_poses, i)
    plot_trajectory_3d(poses)

if __name__ == "__main__":
    main()