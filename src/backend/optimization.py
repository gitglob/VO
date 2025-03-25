from typing import List
import numpy as np
import gtsam


class Optimizer():
    def __init__(self):
        # Create a factor graph and an initial estimate container.
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()

        # Define noise models.
        # Use a very small noise for the prior to fix the first pose.
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6]*6))
        # Noise for the between factors (rotation and translation).
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 1, 1, 1]))

        # Pose counter
        self.pose_counter = 0

    def add_first_pose(self, pose: np.ndarray):
        # Convert the numpy 4x4 matrix to a gtsam.Pose3 object.
        gtsam_pose = gtsam.Pose3(pose)
        self.initial.insert(self.pose_counter, gtsam_pose)
        # Add a prior factor for the first pose to fix the scale/coordinate system.
        self.graph.add(gtsam.PriorFactorPose3(self.pose_counter, gtsam_pose, self.prior_noise))

        self.pose_counter += 1

    def add_pose(self, T: np.ndarray, pose: np.ndarray):
        # Convert the numpy 4x4 matrices to a gtsam.Pose3 objects
        pose = gtsam.Pose3(T)
        T = gtsam.Pose3(pose)

        # Insert the new absolute pose into the initial estimates.
        self.initial.insert(self.pose_counter, pose)
        
        # Add a between factor with the relative transformation as the measurement.
        self.graph.add(gtsam.BetweenFactorPose3(self.pose_counter - 1, self.pose_counter,
                                                T, self.odometry_noise))
        self.pose_counter += 1

    def optimize(self) -> List[np.ndarray]:
        # Optimize the factor graph using Levenberg-Marquardt.
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial, params)
        result = optimizer.optimize()

        # Extract the optimized poses as a list of 4x4 numpy matrices.
        optimized_poses = []
        for i in range(self.pose_counter):
            optimized_pose = result.atPose3(i)
            optimized_poses.append(optimized_pose.matrix())

        return optimized_poses