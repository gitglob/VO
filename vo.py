from pathlib import Path
from src.data import Dataset
from src.frame import Frame
from src.frontend import extract_features, match_features, estimate_relative_pose, is_significant_motion, initialize, compute_relative_scale
from src.visualize import plot_matches, plot_vo_trajectory, plot_ground_truth
from src.visualize import plot_keypoints, plot_2d_trajectory, plot_ground_truth_2d, plot_trajectory_components
from src.utils import save_image, delete_subdirectories
main_dir = Path(__file__).parent
data_dir = Path.home() / "Documents" / "data" / "VO"


def main():
    debug = True
    use_dist = False
    cleanup = True
    scene = "sequence_37"
    print(f"\t\tUsing dataset: `{scene}` ...")
    results_dir = main_dir / "results" / scene / "2d_2d"

    # Clean previous results
    if cleanup:
        delete_subdirectories(results_dir)

    # Read the data
    dataset_dir = data_dir / scene
    data = Dataset(dataset_dir, use_dist)

    # Plot the ground truth trajectory
    gt = data.ground_truth()
    gt_save_path = results_dir / "ground_truth.png"
    gt_save_path_2d = results_dir / "ground_truth_2d.png"
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
    pose_initialized = False
    scale_computed = False
    while not data.finished():
        # Advance the iteration
        i+=1
        if i%50 == 0:
            print(f"\tIteration: {i} / {data.length()}")

        # Capture new image frame (current_frame)
        ts, img, gt_pose = data.get()
        if debug:
            rgb_save_path = results_dir / "img" / f"{i}_rgb.png"
            save_image(img, rgb_save_path)
        
        # Feature Extraction
        keypoints, descriptors = extract_features(img) # (M), (M, 32)
        if debug:
            kpts_save_path = results_dir / "keypoints" / f"{i}_kpts.png"
            plot_keypoints(img, keypoints, kpts_save_path)

        # Create a frame 
        frame = Frame(i, img, keypoints, descriptors)
        frames.append(frame)

        # The very first frame is the reference frame
        if frame.id == 0:
            print(f"{i}: Taking reference frame...")
            prev_keyframe = frame
            pose = gt_pose
            error = 0
            frame.set_pose(pose)

        # After the first frame, we perform feature matching
        else:
            # Extract the last keyframe
            prev_keyframe = keyframes[-1]

            # Feature matching
            matches = match_features(prev_keyframe, frame, debug) # (N) : N < M
            
            # If pose has not been initialized, we need to initialize the 3d points using the Essential Matrix and Triangulation
            if not pose_initialized:
                print(f"{i}: Trying to initialize pose...")
                # Etract the initial pose using the Essential or Homography matrix (2d-2d)
                pose, success = initialize(prev_keyframe, frame, matches, K)
                if success:
                    print("Estimated pose through triangulation successfully!")
                    pose_initialized = True

                # If this is not the 2nd frame, we also compute the relative scale
                if len(keyframes) > 1:              
                    print(f"{i}: Trying to compute scale...")
                    # Use the previous and current matches and frames to compute the relative scale
                    pre_prev_keyframe = keyframes[-2]
                    scale_factor, success = compute_relative_scale(pre_prev_keyframe, prev_keyframe, frame)
                    if success:
                        print("Scale computed successfully!")
                        scale_computed = True

                    # Scale the pose
                    pose[:3, 3] = pose[:3, 3]*scale_factor
                
                frame.set_pose(pose)
                error = 0

            # If scale has been initialized, we can calculate VO using PnP
            elif scale_computed:
                # Estimate the relative pose using PnP (3d-2d)
                displacement, error = estimate_relative_pose(prev_keyframe, 
                                                    frame, 
                                                    K, dist_coeffs,
                                                    debug) # (4, 4)
                if displacement is None:
                    print(f"Warning: solvePnP failed!")
                    continue
            
                # Check if this frame is a keyframe (significant motion or lack of feature matches)
                if not is_significant_motion(displacement, debug=debug):
                    continue
                    
                # Calculate the new pose
                pose = prev_keyframe.pose @ displacement # (4, 4)
                frame.set_pose(pose)
            
            # Save the matches
            if debug:
                match_save_path = main_dir / "results" / scene / "matches" / f"{frame.id}_{prev_keyframe.id}.png"
                plot_matches(prev_keyframe.img, prev_keyframe.keypoints,
                            frame.img, frame.keypoints,
                            matches, match_save_path)

        # Save the keyframe
        if debug:
            keyframe_save_dir = results_dir / "keyframes"
            save_image(frame.img, keyframe_save_dir / f"{i}_rgb.png")
                
        # Keep the poses, ground truth, and keyframes
        poses.append(pose)
        gt_poses.append(gt_pose)
        keyframes.append(frame)
        
        # Visualize the current state of the map and trajectory
        traj2d_save_path = results_dir / "vo" / f"{i}.png"
        plot_2d_trajectory(poses, gt_poses, save_path=traj2d_save_path, ground_truth=True, limits=False)
        traj_comp_save_path = results_dir / "vo_debug" / f"{i}.png"
        plot_trajectory_components(poses, gt_poses, error, save_path=traj_comp_save_path)

    # Save final map and trajectory
    final_traj_save_path = results_dir / "vo" / "final_trajectory.png"
    plot_vo_trajectory(poses, final_traj_save_path, show_plot=True)

if __name__ == "__main__":
    main()