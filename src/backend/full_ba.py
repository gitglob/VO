import gtsam
import numpy as np
import cv2
from config import SETTINGS


MEASUREMENT_SIGMA = SETTINGS["ba"]["measurement_noise"]
PRIOR_SIGMA = SETTINGS["ba"]["first_pose_noise"]


class BA:
    def __init__(self, K: np.ndarray):
        """
        Initializes the BA class.
        
        Args:
            K: The camera intrinsics matrix.
        """
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()

        # Initialize calibration
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        self.calibration = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)

        # Noise models
        self.measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, MEASUREMENT_SIGMA)
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([PRIOR_SIGMA]*6))
        
        # Counters for poses and landmarks keys.
        self.poses = []
        self.landmarks = []

    def add_first_pose(self, pose: np.ndarray, kf_idx: int):
        """Add the first pose and anchor it with a prior factor."""
        pose = gtsam.Pose3(pose)
        pose_key = gtsam.symbol('x', kf_idx)

        self.initial_estimates.insert(pose_key, pose)
        self.graph.add(gtsam.PriorFactorPose3(pose_key, pose, self.prior_noise))
        
        self.poses.append(kf_idx)

    def add_pose(self, pose: np.ndarray, kf_idx: int):
        """Add a subsequent pose."""
        pose = gtsam.Pose3(pose)
        pose_key = gtsam.symbol('x', kf_idx)

        self.initial_estimates.insert(pose_key, pose)
        
        self.poses.append(kf_idx)

    def add_landmarks(self, landmarks: np.ndarray, landmark_kpts: np.ndarray):
        """Add landmarks.
        
        Args:
            landmarks: A (N, 3) array with landmark positions.
            landmark_kpts: The corresponding keypoints.
        """
        for i in range(len(landmarks)):
            pt = landmarks[i]
            pt = gtsam.Point3(pt)

            l_idx = landmark_kpts[i].class_id
            if l_idx in self.landmarks:
                continue
            landmark_key = gtsam.symbol('l', l_idx)

            self.initial_estimates.insert(landmark_key, pt)
            
            self.landmarks.append(l_idx)

    def add_observations(self, pose_idx: int, observations: np.ndarray[cv2.KeyPoint]):
        """
        Add reprojection observations to the BA problem.
        
        Args:
            pose_idx:     index of the keyframe (corresponding to symbol 'x')
            observations: list of keypoints
        """
        pose_key = gtsam.symbol('x', pose_idx)

        # Iterate over all observations
        for kpt in observations:
            # Extract the keypoint id and its pixel location
            l_idx = kpt.class_id
            u, v = kpt.pt

            # Add the observation factor to the graph
            landmark_key = gtsam.symbol('l', l_idx)
            factor = gtsam.GenericProjectionFactorCal3_S2(
                gtsam.Point2(u, v),
                self.measurement_noise,
                pose_key,
                landmark_key,
                self.calibration
            )
            self.graph.add(factor)

    def optimize(self, params=None):
        """
        Optimize the factor graph using Levenberg-Marquardt.
        
        Args:
            params: (Optional) gtsam.LevenbergMarquardtParams. If None, default parameters are used.
            
        Returns:
            result: A gtsam.Values object containing the optimized poses and landmarks.
        """
        if params is None:
            params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimates, params)
        result = optimizer.optimize()

        # Extract the optimized poses as a (N, 4, 4) array.
        optimized_poses = np.empty((len(self.poses), 4, 4))
        for i, p_idx in enumerate(self.poses):
            pose_key = gtsam.symbol('x', p_idx)
            optimized_pose = result.atPose3(pose_key)
            optimized_poses[i] = optimized_pose.matrix()

        # Extract the optimized landmark positions as a (N, 3) array.
        optimized_landmarks = np.empty((len(self.landmarks), 3))
        for i, l_idx in enumerate(self.landmarks):
            landmark_key = gtsam.symbol('l', l_idx)
            optimized_landmarks[i] = result.atPoint3(landmark_key)

        return list(optimized_poses), optimized_landmarks
