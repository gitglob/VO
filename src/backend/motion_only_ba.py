import gtsam
import numpy as np

class MotionOnlyBA:
    def __init__(self, calibration, measurement_sigma=1.0, prior_sigma=0.1, landmark_fix_sigma=1e-9):
        """
        Initializes the motion-only BA class.

        Args:
            calibration: A gtsam.Cal3_S2 camera calibration object.
            measurement_sigma: Standard deviation for the reprojection measurement noise (in pixels).
            prior_sigma: Standard deviation for the prior noise on the first pose.
            landmark_fix_sigma: A very small standard deviation to fix landmarks.
        """
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()
        self.calibration = calibration

        # Noise models.
        self.measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, measurement_sigma)
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([prior_sigma]*6))
        # Landmark fix noise is set very small so that landmarks do not change.
        self.landmark_fix_noise = gtsam.noiseModel.Isotropic.Sigma(3, landmark_fix_sigma)

        self.pose_count = 0
        self.landmark_count = 0

    def add_first_pose(self, pose):
        """
        Adds the first camera pose (gtsam.Pose3) and anchors it with a prior factor.

        Args:
            pose: A gtsam.Pose3 object.
        """
        pose_key = gtsam.symbol('x', self.pose_count)
        self.initial_estimates.insert(pose_key, pose)
        self.graph.add(gtsam.PriorFactorPose3(pose_key, pose, self.prior_noise))
        self.pose_count += 1

    def add_pose(self, pose):
        """
        Adds a subsequent camera pose.

        Args:
            pose: A gtsam.Pose3 object.
        """
        pose_key = gtsam.symbol('x', self.pose_count)
        self.initial_estimates.insert(pose_key, pose)
        self.pose_count += 1

    def add_landmarks(self, landmarks):
        """
        Adds landmarks and fixes them via a prior factor with near-zero noise.

        Args:
            landmarks: A list of landmarks. Each can be a numpy array of shape (3,)
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
            # Fix the landmark by adding a prior factor with a very small noise.
            self.graph.add(gtsam.PriorFactorPoint3(landmark_key, pt, self.landmark_fix_noise))
            landmark_keys.append(landmark_key)
            self.landmark_count += 1
        return landmark_keys

    def add_observations(self, observations):
        """
        Adds reprojection observations to the graph.

        Args:
            observations: A list of tuples (pose_idx, landmark_idx, (u, v)) representing an observation.
                          - pose_idx: index of the pose (corresponding to symbol 'x')
                          - landmark_idx: index of the landmark (corresponding to symbol 'l')
                          - (u, v): the observed pixel coordinates.
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
        Optimizes the factor graph using Levenberg–Marquardt.
        Only the camera poses will be adjusted; landmarks remain fixed.

        Args:
            params: (Optional) gtsam.LevenbergMarquardtParams object. Uses default if None.

        Returns:
            result: A gtsam.Values object with the optimized poses.
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
    
    # Create an instance of MotionOnlyBA.
    motion_ba = MotionOnlyBA(calibration, measurement_sigma=1.0, prior_sigma=0.1, landmark_fix_sigma=1e-9)
    
    # Add the first pose (e.g., the identity pose).
    first_pose = gtsam.Pose3()  # Identity pose.
    motion_ba.add_first_pose(first_pose)
    
    # Add a second pose (simulate a small motion).
    second_pose = gtsam.Pose3(
        gtsam.Rot3.YawPitchRoll(0.05, 0.02, -0.01),
        gtsam.Point3(0.1, 0.0, 0.0)
    )
    motion_ba.add_pose(second_pose)
    
    # Add some landmarks (assumed to be known and fixed).
    landmarks = [
        np.array([1.0, 0.5, 10.0]),
        np.array([1.2, -0.3, 9.5]),
        np.array([0.8, 0.8, 10.5])
    ]
    motion_ba.add_landmarks(landmarks)
    
    # Add observations: each observation is (pose_idx, landmark_idx, (u, v)).
    # Here we assume that both poses observe all the landmarks.
    observations = [
        (0, 0, (320.0, 240.0)),
        (0, 1, (315.0, 235.0)),
        (0, 2, (325.0, 245.0)),
        (1, 0, (322.0, 242.0)),
        (1, 1, (317.0, 237.0)),
        (1, 2, (327.0, 247.0))
    ]
    motion_ba.add_observations(observations)
    
    # Optimize the factor graph (only poses will change).
    result = motion_ba.optimize()
    
    # Retrieve and print optimized poses.
    print("Optimized Poses:")
    for i in range(motion_ba.pose_count):
        pose_key = gtsam.symbol('x', i)
        opt_pose = result.atPose3(pose_key)
        print(f"Pose {i}: {opt_pose}")
