from src.data import Dataset
from src.frame import Frame
from src.feature_tracking import match_features
from src.pose_estimation import estimate_relative_pose, is_keyframe
from src.visualize import plot_trajectory, plot_ground_truth, plot_trajectory_3d
from src.utils import save_depth, save_image, delete_subdirectories
from config import data_dir, main_dir, scene, results_dir, debug


def main():
    print(f"\t\tUsing dataset: `{scene}` ...")
    use_dist = False
    cleanup = True

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

    # Initialize bookkeping lists
    frames = []
    keyframes = []
    poses = []
    gt_poses = []
    times = []

    # Run the main VO loop
    i = -1
    while not data.finished():
        # Advance the iteration
        i+=1
        if debug:
            print(f"\n\tIteration: {i} / {data.length()}")

        # Capture new image frame (current_frame)
        type, ts, img, depth, gt_pose = data.get()

        # Create a frame 
        frame = Frame(i, img, depth)
        frames.append(frame)

        # The first frame is de facto a keyframe
        if i == 0:
            poses.append(gt_pose)
            gt_poses.append(gt_pose)
            frame.set_pose(gt_pose)
            keyframes.append(frame)
            times.append(ts)

            save_image(img, results_dir / "keyframes" / f"{i}_bw.png")
            save_depth(depth, results_dir / "depth" / f"{i}_d")
            plot_trajectory(poses, gt_poses, i)
        else:
            # Feature matching
            matches = match_features(keyframes[-1], frame, K) # (N) : N < M

            # Check if there are enough matches
            if len(matches) < 20:
                print(f"Not enough matches: ({len(matches)})!")
                continue

            # Estimate the relative pose (odometry) between the current frame and the last keyframe
            T_q2t = estimate_relative_pose(matches, keyframes[-1], frame, K) # (4, 4)
            if T_q2t is None:
                continue
           
            # Check if this frame is a keyframe (significant motion or lack of feature matches)
            if is_keyframe(T_q2t, debug=debug):
                # Calculate the new pose
                T_w2q = poses[-1]
                T_w2t = T_w2q @ T_q2t # (4, 4)

                # Save info
                poses.append(T_w2t)
                gt_poses.append(gt_pose)
                frame.set_pose(T_w2t)
                frame.set_keyframe(True)
                keyframes.append(frame)
                times.append(ts)

                # Save plots
                save_image(img, results_dir / "keyframes" / f"{i}_bw.png")
                save_depth(depth, results_dir / "depth" / f"{i}_d")
                plot_trajectory(poses, gt_poses, i)

    # Save final map and trajectory
    plot_trajectory(poses, gt_poses, i)
    plot_trajectory_3d(poses, gt_poses)

if __name__ == "__main__":
    main()