from typing import List
import numpy as np
import gtsam


def optimize_poses(initial_poses: List[np.ndarray], between_noise_sigma=0.1):
    """
    Perform Bundle Adjustment on a sequence of camera poses using GTSAM.
    
    Parameters:
        initial_poses (list): List of 4x4 numpy arrays representing camera poses.
        between_noise_sigma (float): Standard deviation for the between factor noise.
        
    Returns:
        optimized_poses (list): List of 4x4 numpy arrays representing the optimized poses.
    """
    # Create a factor graph and an initial estimate container.
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    
    # Define noise models.
    # Use a very small noise for the prior to fix the first pose.
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6]*6))
    # Noise for the between factors (rotation and translation).
    between_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([between_noise_sigma]*6))
    
    # Insert poses and add factors.
    for i, pose in enumerate(initial_poses):
        # Convert the numpy 4x4 matrix to a gtsam.Pose3 object.
        gtsam_pose = gtsam.Pose3(pose)
        initial_estimate.insert(i, gtsam_pose)
        
        if i == 0:
            # Add a prior factor for the first pose to fix the scale/coordinate system.
            graph.add(gtsam.PriorFactorPose3(i, gtsam_pose, prior_noise))
        else:
            # Add a between factor between consecutive poses.
            # We assume the relative transformation between pose i-1 and i is valid.
            prev_pose = gtsam.Pose3(initial_poses[i-1])
            relative_pose = prev_pose.between(gtsam_pose)
            graph.add(gtsam.BetweenFactorPose3(i-1, i, relative_pose, between_noise))
    
    # Optimize the factor graph using Levenberg-Marquardt.
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    
    # Extract the optimized poses as a list of 4x4 numpy matrices.
    optimized_poses = []
    for i in range(len(initial_poses)):
        optimized_pose = result.atPose3(i)
        optimized_poses.append(optimized_pose.matrix())
    
    return optimized_poses
