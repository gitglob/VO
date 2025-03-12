from pathlib import Path
from src.data import Dataset
from src.frame import Frame
from src.frontend.feature_tracking import match_features
from src.frontend.pose_estimation import estimate_relative_pose, is_keyframe
from src.visualize import plot_trajectory, plot_ground_truth, plot_trajectory_3d
from src.utils import save_depth, save_image, delete_subdirectories
from config import data_dir, main_dir, scene, results_dir


def main():
    print(f"\t\tUsing dataset: `{scene}` ...")
    debug = False
    use_dist = False
    cleanup = True
    log_period = 20

    # Clean previous results
    if cleanup:
        delete_subdirectories(results_dir)

    # Read the data
    dataset_dir = data_dir / scene
    data = Dataset(dataset_dir, use_dist)

    # Plot the ground truth trajectory
    gt = data.ground_truth()
    plot_ground_truth(gt)

    # Get the camera matrix
    K, dist_coeffs = data.get_intrinsics()
    # Initialize the graph, the keyframes, and the bow list
    frames = []
    keyframes = []
    poses = []
    gt_poses = []

    # Run the main VO loop
    i = -1
    while not data.finished():
        # Advance the iteration
        i+=1
        if i%log_period == 0:
            print(f"\n\tIteration: {i} / {data.length()}")

        # Capture new image frame (current_frame)
        type, ts, img, depth, gt_pose = data.get()
        if debug:
            depth_save_path = results_dir / "depth" / f"{i}_d"
            save_depth(depth, depth_save_path)
            rgb_save_path = results_dir / "img" / f"{i}_rgb.png"
            save_image(img, rgb_save_path)

        # Create a frame 
        frame = Frame(i, img, depth)
        frames.append(frame)

        # The first frame is de facto a keyframe
        if i == 0:
            poses.append(gt_pose)
            gt_poses.append(gt_pose)
            frame.set_pose(gt_pose)
            keyframes.append(frame)
            if debug:
                save_image(img, results_dir / "keyframes" / f"{i}_rgb.png")

            plot_trajectory(poses, gt_poses, i)
        else:
            # Feature matching
            matches = match_features(keyframes[-1], frame, K, debug=debug) # (N) : N < M

            # Check if there are enough matches
            if len(matches) < 20:
                print("Not enough matches!")
                continue

            # Estimate the relative pose (odometry) between the current frame and the last keyframe
            T, error = estimate_relative_pose(keyframes[-1], frame, K, debug=True) # (4, 4)
            if T is None:
                print(f"Warning: solvePnP failed!")
                continue
           
            # Check if this frame is a keyframe (significant motion or lack of feature matches)
            if is_keyframe(T, debug=debug):
                # Calculate the new pose
                pose = poses[-1] @ T # (4, 4)

                # Save info
                poses.append(pose)
                gt_poses.append(gt_pose)
                frame.set_pose(pose)
                frame.set_keyframe(True)
                keyframes.append(frame)

                # Save keyframe
                if debug:
                    save_image(frame.img, results_dir / "keyframes" / f"{i}_rgb.png")
        
            # Visualize the current state of the map and trajectory
            if i%log_period == 0:
                plot_trajectory(poses, gt_poses, i)

    # Save final map and trajectory
    plot_trajectory(poses, gt_poses, i)
    plot_trajectory_3d(poses)

if __name__ == "__main__":
    main()