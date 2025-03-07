from src.data import Dataset

from src.frame import Frame

from src.frontend.feature_tracking import match_features
from src.frontend.initialization import initialize_pose, triangulate_points
from src.frontend.tracking import estimate_relative_pose, is_keyframe, predict_pose_constant_velocity, guided_descriptor_search, get_new_triangulated_points

from src.backend.local_map import Map
from src.backend import optimization

from src.visualize import plot_vo_trajectory, plot_ground_truth
from src.visualize import plot_2d_trajectory, plot_ground_truth_2d, plot_ground_truth_6dof, plot_trajectory_components
from src.utils import save_image, delete_subdirectories, transform_points, invert_transform

from config import main_dir, data_dir, scene, results_dir
print(f"\t\tUsing dataset: `{scene}` ...")


"""
Important notes:
- Both the Kitti and OpenCV camera frames follow the same convention:
    - x: right
    - y: down
    - z: forward
"""


def main():
    debug = True
    use_dist = False
    cleanup = True

    # Clean previous results
    if cleanup:
        delete_subdirectories(results_dir)

    # Read the data
    data = Dataset(data_dir, scene, use_dist)

    # Plot the ground truth trajectory
    gt = data.ground_truth()
    plot_ground_truth(gt, save_path=results_dir / "ground_truth/3d.png")
    plot_ground_truth_2d(gt, save_path=results_dir / "ground_truth/2d.png")
    plot_ground_truth_6dof(gt, save_path=results_dir / "ground_truth/6dof.png")

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
    while not data.finished():
        # Advance the iteration
        i+=1
        # if i%50 == 0:
        print(f"\tIteration: {i} / {data.length()}")

        # Capture new image frame (current_frame)
        t, img, gt_pose = data.get(debug)
        gt_poses.append(gt_pose)

        # Create a frame and extract its ORB features
        frame = Frame(i, img, debug)
        frames.append(frame)

        # Iteration #0
        if frame.id == 0:
            pose = gt_pose
            poses.append(pose)
            frame.set_pose(pose)
            frame.set_keyframe(True)
            keyframes.append(frame)
            if debug:
                save_image(frame.img, results_dir / "keyframes" / f"{i}_rgb.png")
            RMSE = 0
                
            # Visualize the current state of the map and trajectory
            plot_2d_trajectory(poses, gt_poses, save_path=results_dir / "vo" / f"{i}_a.png")
            plot_trajectory_components(poses, gt_poses, RMSE, save_path=results_dir / "vo" / f"{i}_b.png")

            # There is nothing left to do in the first iteration
            continue
        else:                    
            # ########### Initialization ###########
            if not is_initialized:
                # Extract the last frame, keyframe
                ref_frame = frames[-2]

                # Feature matching
                matches = match_features(frame, ref_frame, K, "0-raw", debug) # (N) : N < M

                # Check if there are enough matches
                if len(matches) < 20:
                    print("Not enough matches!")
                    continue

                # Extract the initial pose using the Essential or Homography matrix (2d-2d)
                init_pose, is_initialized = initialize_pose(frame, ref_frame, K, debug)
                if not is_initialized:
                    print("Pose initialization failed!")
                    continue

                # Save the initial pose
                pose = poses[-1] @ init_pose
                poses.append(pose)
                frame.set_pose(pose)
                
                # Visualize the current state of the map and trajectory
                plot_2d_trajectory(poses, gt_poses, save_path=results_dir / "vo" / f"{i}_a.png")
                plot_trajectory_components(poses, gt_poses, RMSE, save_path=results_dir / "vo" / f"{i}_b.png")

                # Triangulate the 3D points using the initial pose
                points_c, point_ids, is_initialized = triangulate_points(frame, ref_frame, K, debug)
                if not is_initialized:
                    print("Triangulation failed!")
                    continue

                # Transfer the points to the world frame
                points_w = transform_points(points_c, invert_transform(pose))

                # Create a local map and push the triangulated points
                map = Map(ref_frame.id)
                map.add_initialization_points(points_w, point_ids, frame, ref_frame)

                # Go to next iteration as we don't want to do 3d-2d PnP on the same image that we initialized
                continue

            # ########### Tracking ###########
            if is_initialized:
                ref_keyframe = keyframes[-1]
                    
                # Predict next pose using constant velocity
                pose_pred = predict_pose_constant_velocity(poses)
    
                # Find the map points that can be seen in the previous frame
                map_points, map_points_2d, map_descriptors = map.get_points_in_view(pose_pred, K)

                # Project the map points to the new predicted camera frame
                # map_points_pred = transform_points(map_points, pose_pred)

                # Compare the map point’s descriptor to the descriptors of the new frame in a small search window around the projected location
                guided_matches = guided_descriptor_search(map_points_2d, map_descriptors, frame)

                # Estimate the camera displacement using PnP (3d-2d)
                displacement, RMSE = estimate_relative_pose(map_points, frame, guided_matches, pose, K, debug)
                if displacement is None:
                    print(f"Warning: solvePnP failed!")
                    is_initialized = False
                    continue
            
                # Check if this frame is a keyframe     
                if not is_keyframe(displacement, debug=debug):
                    continue

                # Save the keyframe
                frame.set_keyframe(True)
                keyframes.append(frame)
                if debug:
                    save_image(frame.img, results_dir / "keyframes" / f"{i}_rgb.png")

                # Calculate the new pose
                pose = poses[-1] @ displacement 
                poses.append(pose)
                frame.set_pose(pose)
                    
                # Do feature matching with the previous keyframe
                ref_frame = keyframes[-2]
                matches = match_features(frame, ref_frame, K, "4-tracking", debug)

                # Get inliers by Epipolar constraint
                triangulation_pose, points_c, point_ids, triangulation_success = get_new_triangulated_points(frame, ref_keyframe, map, K)
                if not triangulation_success:
                    print("Triangulation failed!")
                    continue

                # Transfer the points to the world frame
                points_w = transform_points(points_c, invert_transform(triangulation_pose))

                # Create a local map and push the triangulated points
                map.add_tracking_points(points_w, point_ids, frame, ref_frame)

                # Clean up map points that are not seen anymore
                map.cleanup(frame.pose, K)
                
                # Visualize the current state of the map and trajectory
                plot_2d_trajectory(poses, gt_poses, save_path=results_dir / "vo" / f"{i}_a.png")
                plot_trajectory_components(poses, gt_poses, RMSE, save_path=results_dir / "vo" / f"{i}_b.png")

    # Save final map and trajectory
    final_traj_save_path = results_dir / "vo" / "final_trajectory.png"
    plot_vo_trajectory(poses, final_traj_save_path, show_plot=True)

if __name__ == "__main__":
    main()