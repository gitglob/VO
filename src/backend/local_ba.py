import gtsam
import numpy as np

class PointsOnlyBA:
    def __init__(self, calibration, measurement_sigma=1.0, pose_fix_sigma=1e-9):
        """
        Initializes the BA object that optimizes only landmarks while keeping camera poses fixed.

        Args:
            calibration: A gtsam.Cal3_S2 camera calibration object.
            measurement_sigma: Standard deviation (in pixels) for the reprojection measurement noise.
            pose_fix_sigma: Standard deviation for fixing camera poses (should be set very low).
        """
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()
        self.calibration = calibration

        # Noise models:
        self.measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, measurement_sigma)
        # For camera poses, we add a prior factor with near-zero noise to fix them.
        self.pose_fix_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([pose_fix_sigma]*6))
        
        self.pose_count = 0
        self.landmark_count = 0

    def add_pose(self, pose):
        """
        Add a camera pose (gtsam.Pose3) and fix it with a near-zero noise prior factor.
        These poses will remain fixed during optimization.
        """
        pose_key = gtsam.symbol('x', self.pose_count)
        self.initial_estimates.insert(pose_key, pose)
        self.graph.add(gtsam.PriorFactorPose3(pose_key, pose, self.pose_fix_noise))
        self.pose_count += 1

    def add_landmarks(self, landmarks):
        """
        Add landmarks to the BA problem.

        Args:
            landmarks: A list of landmarks. Each landmark can be a numpy array of shape (3,)
                       or a gtsam.Point3.

        Returns:
            A list of gtsam.Symbol keys corresponding to the inserted landmarks.
        """
        landmark_keys = []
        for pt in landmarks:
            landmark_key = gtsam.symbol('l', self.landmark_count)
            if isinstance(pt, np.ndarray):
                pt = gtsam.Point3(pt[0], pt[1], pt[2])
            self.initial_estimates.insert(landmark_key, pt)
            landmark_keys.append(landmark_key)
            self.landmark_count += 1
        return landmark_keys

    def add_observations(self, observations):
        """
        Add reprojection observations linking camera poses and landmarks.

        Args:
            observations: A list of tuples (pose_idx, landmark_idx, (u, v)) where:
                - pose_idx: index of the camera pose (corresponds to symbol 'x')
                - landmark_idx: index of the landmark (corresponds to symbol 'l')
                - (u, v): observed image pixel coordinates.
        """
        for (pose_idx, landmark_idx, uv) in observations:
            pose_key = gtsam.symbol('x', pose_idx)
            landmark_key = gtsam.symbol('l', landmark_idx)
            factor = gtsam.GenericProjectionFactorCal3_S2(
                gtsam.Point2(uv[0], uv[1]),
                self.measurement_noise,
                pose_key,
                landmark_key,
                self.calibration
            )
            self.graph.add(factor)

    def optimize(self, params=None):
        """
        Optimize the factor graph using Levenberg–Marquardt.
        Only the landmarks will be adjusted; camera poses remain fixed.

        Args:
            params: (Optional) gtsam.LevenbergMarquardtParams. Uses defaults if None.

        Returns:
            result: A gtsam.Values object containing the optimized landmarks (and unchanged camera poses).
        """
        if params is None:
            params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimates, params)
        result = optimizer.optimize()
        return result

# ============================================================
# Example usage:
# ============================================================
if __name__ == "__main__":
    # Define camera calibration parameters.
    fx, fy = 525.0, 525.0
    cx, cy = 319.5, 239.5
    calibration = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)

    # Create an instance of PointsOnlyBA.
    points_ba = PointsOnlyBA(calibration, measurement_sigma=1.0, pose_fix_sigma=1e-9)

    # Add fixed camera poses.
    # For example, add two camera poses.
    pose1 = gtsam.Pose3()  # Identity pose.
    points_ba.add_pose(pose1)

    # Add a second pose simulating a small motion.
    pose2 = gtsam.Pose3(gtsam.Rot3.YawPitchRoll(0.05, 0.0, 0.0), gtsam.Point3(0.1, 0, 0))
    points_ba.add_pose(pose2)

    # Add landmarks (3D points) as
