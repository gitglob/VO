from src.data import Dataset
from src.frame import Frame
from src.pose_estimation import estimate_relative_pose, check_velocity, is_keyframe
from src.visualize import plot_ground_truth, plot_icp, plot_trajectory, plot_trajectory_3d
from src.utils import save_image, delete_subdirectories, save_depth, invert_transform
from config import data_dir, scene, results_dir


def main():
    print(f"\t\tUsing dataset: `{scene}` ...")
    debug = True
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

    # Initialize the graph, the keyframes, and the bow list
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

            save_image(img, results_dir / "keyframes" / f"{i}_rgb.png")
            save_depth(depth, results_dir / "depth" / f"{i}_d")
            plot_trajectory(poses, gt_poses, i)
        else:
            # Estimate time since last keyframe
            dt = ts - times[-1]

            # Estimate the relative pose (odometry) between the current frame and the last keyframe
            T = estimate_relative_pose(keyframes[-1], frame, debug=debug)
            if T is None:
                print(f"Warning: ICP failed!")
                continue

            # Check if the velocity is within acceptable limits
            # success = check_velocity(T, dt)
            # if not success:
            #     continue
           
            # Check if this frame is a keyframe (significant motion or lack of feature matches)
            if is_keyframe(T, debug=debug):
                # Calculate the new pose
                pose = poses[-1] @ invert_transform(T) # (4, 4)

                # Save info
                poses.append(pose)
                gt_poses.append(gt_pose)
                frame.set_pose(pose)
                frame.set_keyframe(True)
                keyframes.append(frame)
                times.append(ts)

                # Save plots
                save_image(img, results_dir / "keyframes" / f"{i}_rgb.png")
                save_depth(depth, results_dir / "depth" / f"{i}_d")
                plot_trajectory(poses, gt_poses, i)
                plot_icp(keyframes[-2].pcd_down, frame.pcd_down, transform=T)
        
    # Save final map and trajectory
    plot_trajectory(poses, gt_poses, i)
    plot_trajectory_3d(poses)

if __name__ == "__main__":
    main()