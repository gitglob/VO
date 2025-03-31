import numpy as np

from src.data import Dataset
from src.frame import Frame
from src.feature_matching import match_features
from src.pose_estimation import estimate_pose, is_keyframe
from src.scale_estimation import estimate_depth_scale, validate_scale
from src.visualize import plot_trajectory, plot_ground_truth, plot_trajectory_3d
from src.utils import save_image, delete_subdirectories

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
    debug = False
    use_dist = False
    cleanup = True
    use_keyframes = True
    log_period = 100

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
    while not data.finished():
        # Advance the iteration
        i+=1
        if debug:
            print(f"\n\n\tIteration: {i} / {data.length()}")
        if i%log_period == 0:
            print(f"\n\tIteration: {i} / {data.length()}")

        # Capture new image frame (current_frame)
        t, img, gt_pose = data.get(debug)

        # Create a frame and extract its ORB features
        frame = Frame(i, img, debug)
        frames.append(frame)

        # Iteration #0
        if frame.id == 0:
            pose = gt_pose
            gt_poses.append(gt_pose)
            poses.append(pose)
            frame.set_pose(pose)
            frame.set_keyframe(True)
            keyframes.append(frame)
            if debug:
                save_image(frame.img, results_dir / "keyframes" / f"{i}_bw.png")

            plot_trajectory(poses, gt_poses, i)
        else:                    
            # Feature matching
            matches = match_features(keyframes[-1], frame, debug) # (N) : N < M

            # Check if there are enough matches
            if len(matches) < 20:
                print("Not enough matches!")
                continue

            # Extract the initial pose using the Essential or Homography matrix (2d-2d)
            T, is_initialized = estimate_pose(keyframes[-1], frame, K, debug)
            if not is_initialized:
                print("Pose estimation failed!")
                continue
            assert np.linalg.norm(T[:3, 3]) - 1 < 1e-6

            # Calculate the next pose with scale ambiguity
            unscaled_pose = poses[-1] @ T

            # Estimate the depth scale
            scale = estimate_depth_scale([poses[-1], unscaled_pose], [gt_poses[-1], gt_pose], debug=debug)
        
            # Remove scale ambiguity
            T[:3, 3] *= scale

            # Apply the scale to the pose and validate it
            scaled_pose = poses[-1] @ T
            validate_scale([poses[-1], scaled_pose], [gt_poses[-1], gt_pose])

            # Check if the frame is a keyframe
            if use_keyframes and is_keyframe(T, debug=debug):
                # Save the poses
                gt_poses.append(gt_pose)
                poses.append(scaled_pose)
                
                # Save the frame info
                frame.set_pose(scaled_pose)
                frame.set_keyframe(True)

                # Save the keyframe
                keyframes.append(frame)
                if debug:
                    save_image(frame.img, results_dir / "keyframes" / f"{i}_bw.png")

            # Visualize the current state of the map and trajectory
            if i%log_period == 0 or debug:
                plot_trajectory(poses, gt_poses, i)

    # Save final map and trajectory
    plot_trajectory(poses, gt_poses, i, save_path=results_dir / "vo")
    plot_trajectory_3d(poses, gt_poses)

if __name__ == "__main__":
    main()