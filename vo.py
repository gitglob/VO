from pathlib import Path
import numpy as np
from src.data import Dataset
from src.frame import Frame
from src.frontend import extract_features, match_features, estimate_relative_pose, is_significant_motion
from src.visualize import plot_matches, plot_vo_trajectory, plot_ground_truth
from src.visualize import plot_keypoints, plot_2d_trajectory, plot_ground_truth_2d, plot_trajectory_components
from src.utils import save_image, delete_subdirectories
main_dir = Path(__file__).parent


def main():
    debug = False
    use_dist = False
    cleanup = True
    scene = "rgbd_dataset_freiburg2_pioneer_360"
    print(f"\t\tUsing dataset: `{scene}` ...")

    # Clean previous results
    if cleanup:
        delete_subdirectories(main_dir / "results" / scene)

    # Read the data
    data_dir = main_dir / "data" / scene
    data = Dataset(data_dir, use_dist)

    # Plot the ground truth trajectory
    gt = data.ground_truth()
    gt_save_path = main_dir / "results" / scene / "ground_truth.png"
    gt_save_path_2d = main_dir / "results" / scene / "ground_truth_2d.png"
    plot_ground_truth(gt, save_path=gt_save_path)
    plot_ground_truth_2d(gt, save_path=gt_save_path_2d)

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
        if i%50 == 0:
            print(f"\tIteration: {i} / {data.length()}")

        # Capture new image frame (current_frame)
        type, ts, img, depth, gt_pose = data.get()
        if debug:
            depth_save_path = main_dir / "results" / scene / "depth" / f"{i}_d.png"
            save_image(depth, depth_save_path)
            rgb_save_path = main_dir / "results" / scene / "img" / f"{i}_rgb.png"
            save_image(img, rgb_save_path)
        
        # Feature Extraction
        keypoints, descriptors = extract_features(img) # (M), (M, 32)
        if debug:
            kpts_save_path = main_dir / "results" / scene / "keypoints" / f"{i}_kpts.png"
            plot_keypoints(img, keypoints, kpts_save_path)

        # Create a frame 
        frame = Frame(i, img, depth, keypoints, descriptors)
        frames.append(frame)

        # The first frame is de facto a keyframe
        if i == 0:
            pose = gt_pose
            error = 0
            frame.set_pose(pose)
            poses.append(pose)
            gt_poses.append(gt_pose)
            keyframes.append(frame)
            keyframe_save_path = main_dir / "results" / scene / "keyframes" / f"{i}_rgb.png"
            if debug:
                save_image(img, keyframe_save_path)

        # Check if this is the very first image, so that we can perform VO
        if i != 0:
            # Extract the last keyframe
            prev_keyframe = keyframes[-1]

            # Feature matching
            matches = match_features(frame, prev_keyframe, debug) # (N) : N < M
            if debug:
                match_save_path = main_dir / "results" / scene / "matches" / f"{frame.id}_{prev_keyframe.id}.png"
                plot_matches(frame.img, frame.keypoints, 
                         prev_keyframe.img, prev_keyframe.keypoints, 
                         matches, match_save_path)

            # Estimate the relative pose (odometry) between the current frame and the last keyframe
            displacement, error = estimate_relative_pose(matches, 
                                                  frame.keypoints, 
                                                  prev_keyframe.keypoints,  
                                                  prev_keyframe.depth, 
                                                  K, dist_coeffs,
                                                  debug) # (4, 4)
            if displacement is None:
                print(f"Warning: solvePnP failed!")
                continue
           
            # Check if this frame is a keyframe (significant motion or lack of feature matches)
            if not is_significant_motion(displacement, debug=debug):
                continue

            # Save keyframe
            if debug:
                keyframe_save_dir = main_dir / "results" / scene / "keyframes"
                save_image(frame.img, keyframe_save_dir / f"{i}_rgb.png")
                plot_matches(frame.img, frame.keypoints, 
                         prev_keyframe.img, prev_keyframe.keypoints, 
                         matches, keyframe_save_dir / f"{frame.id}_{prev_keyframe.id}.png")
                
            # Calculate the new pose
            pose = prev_keyframe.pose @ displacement # (4, 4)
            frame.set_pose(pose)
            
            # Make the current pose, depth, img, descriptors and keypoints the previous ones
            poses.append(pose)
            gt_poses.append(gt_pose)
            keyframes.append(frame)
        
        # Visualize the current state of the map and trajectory
        traj2d_save_path = main_dir / "results" / scene / "vo" / f"{i}.png"
        plot_2d_trajectory(poses, gt_poses, save_path=traj2d_save_path, ground_truth=True, limits=False)
        traj_comp_save_path = main_dir / "results" / scene / "vo_debug" / f"{i}.png"
        plot_trajectory_components(poses, gt_poses, error, save_path=traj_comp_save_path)

    # Save final map and trajectory
    final_traj_save_path = main_dir / "results" / scene / "vo" / "final_trajectory.png"
    plot_vo_trajectory(poses, final_traj_save_path, show_plot=True)

if __name__ == "__main__":
    main()