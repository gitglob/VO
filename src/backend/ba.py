import gtsam
import numpy as np
import cv2
from config import SETTINGS
from gtsam.symbol_shorthand import X, L 


POSE_PRIOR_SIGMA = float(SETTINGS["ba"]["first_pose_noise"])
ODOMETRY_TRANS_SIGMA = float(SETTINGS["ba"]["odometry_trans_noise"])
ODOMETRY_ROT_SIGMA = np.deg2rad(float(SETTINGS["ba"]["odometry_rot_noise"]))

MEASUREMENT_PRIOR_SIGMA = float(SETTINGS["ba"]["first_measurement_noise"])
MEASUREMENT_SIGMA = float(SETTINGS["ba"]["measurement_noise"])


class BA:
    def __init__(self, K: np.ndarray, verbose=False):
        """
        Initializes the BA class with iSAM2.
        
        Args:
            K: The camera intrinsics matrix.
        """
        # Initialize iSAM2
        self.isam = gtsam.ISAM2()
        
        # These will collect new factors and estimates to be added incrementally.
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()

        # Initialize calibration
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        self.calibration = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)

        # Measurement and odometry noise
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([ODOMETRY_TRANS_SIGMA]*3 + [ODOMETRY_ROT_SIGMA]*3)
        )
        self.measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, MEASUREMENT_SIGMA)
        
        # Prior noise
        self.prior_pose_noise = gtsam.noiseModel.Isotropic.Sigma(6, POSE_PRIOR_SIGMA)
        self.prior_landmark_noise = gtsam.noiseModel.Isotropic.Sigma(3, MEASUREMENT_PRIOR_SIGMA)
        
        # Counters for poses and landmarks keys.
        self.poses = []
        self.landmarks = []
        self._last_pose = None

        # Debugging info
        self.verbose = verbose

        # Flag for first initialization
        self.initialized = False

    def add_first_pose(self, pose: np.ndarray, node_idx: int):
        """Add the first pose and anchor it with a prior factor."""
        pose = gtsam.Pose3(pose)

        self.initial_estimates.insert(X(node_idx), pose)
        self.graph.add(gtsam.PriorFactorPose3(X(node_idx), pose, self.prior_pose_noise))
        if self.verbose:
            self.print_x_nodes_in_graph()
            print(f"Added fixed pose X({node_idx})")
        
        self.poses.append(node_idx)
        self._last_pose = pose

    def add_odometry(self, T: np.ndarray, node_idx: int):
        """
        Add an odometry measurement to the graph.

        Args:
            pose (np.ndarray): The 4x4 transformation matrix representing the pose.
            information (np.ndarray): The 6x6 information matrix representing the measurement uncertainty.
        """
        # Extract the Rotation and Translation components from the transformation matrix
        T = gtsam.Pose3(T)

        # Retrieve the new pose
        new_pose = self._last_pose.compose(T).matrix()
        
        # Add a BetweenFactor (edge) between the last pose and the new pose
        edge = gtsam.BetweenFactorPose3(X(self.poses[-1]), 
                                        X(node_idx), 
                                        T, 
                                        self.odometry_noise)
        self.graph.add(edge)
        if self.verbose:
            self.print_x_nodes_in_graph()
            print(f"Added edge X({self.poses[-1]}) ~ X({node_idx})")

        # Insert the new pose into initial estimates
        self._add_pose(new_pose, node_idx)

    def _add_pose(self, pose: np.ndarray, node_idx: int):
        """Add a subsequent pose."""
        pose = gtsam.Pose3(pose)

        self.initial_estimates.insert(X(node_idx), pose)
        if self.verbose:
            print(f"Added pose X({node_idx})")
        
        self.poses.append(node_idx)
        self._last_pose = pose

    def add_landmarks(self, landmarks: np.ndarray, landmark_kpts: np.ndarray):
        """Add landmarks.
        
        Args:
            landmarks: A (N, 3) array with landmark positions.
            landmark_kpts: The corresponding keypoints.
        """
        # Add initial guesses to all observed landmarks
        for i in range(len(landmarks)):
            # We only want to initialize the first landmark observations
            l_idx = landmark_kpts[i].class_id
            if l_idx in self.landmarks:
                continue

            pt = landmarks[i]
            pt = gtsam.Point3(pt)

            self.initial_estimates.insert(L(l_idx), pt)
            self.graph.add(gtsam.PriorFactorPoint3(L(l_idx), pt, self.prior_landmark_noise))
            # if self.verbose:
            #     print(f"Added landmark L({l_idx})")
            self.landmarks.append(l_idx)

    def add_observations(self, pose_idx: int, observations: np.ndarray[cv2.KeyPoint]):
        """
        Add reprojection observations to the BA problem.
        
        Args:
            pose_idx:     index of the keyframe (corresponding to symbol 'x')
            observations: list of keypoints
        """
        # Iterate over all observations
        for kpt in observations:
            # Extract the keypoint id and its pixel location
            l_idx = kpt.class_id
            u, v = kpt.pt

            # Add the observation factor to the graph
            factor = gtsam.GenericProjectionFactorCal3_S2(
                gtsam.Point2(u, v),
                self.measurement_noise,
                X(pose_idx),
                L(l_idx),
                self.calibration
            )
            self.graph.add(factor)

    def optimize(self):
        """ Optimize the graph and return the optimized robot poses and landmark positions"""
        # Do a full optimize the first time
        if not self.initialized:
            batchOptimizer = gtsam.LevenbergMarquardtOptimizer(
                self.graph, self.initial_estimates)
            self.initial_estimates = batchOptimizer.optimize()
            self.initialized = True

        # Update iSAM
        self.isam.update(self.graph, self.initial_estimates)
        self.estimate = self.isam.calculateEstimate()
        self._lastPose = self.estimate.atPose3(X(self.poses[-1]))

        # Get the optimized robot poses and landmark positions
        optimized_poses = self.get_poses()
        optimized_landmark_poses = self.get_landmarks()

        # Prepare a new graph
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()

        return optimized_poses, self.landmarks, optimized_landmark_poses
    
    def finalize(self):
        """Returns the final estimate"""
        self.estimate = self.isam.calculateEstimate()

        # Get the optimized robot poses and landmark positions
        optimized_poses = self.get_poses()

        return optimized_poses

    def get_poses(self):
        """
        Return the trajectory as an array of 3D coordinates.

        Returns:
            np.ndarray: An Mx3 array with the x, y, z pose coordinates.
        """
        # Extract poses
        poses = np.empty((len(self.poses), 4, 4))

        # Iterate over all nodes
        for i, p_idx in enumerate(self.poses):
            # Get the pose for each node
            pose = self.estimate.atPose3(X(p_idx))
            # Extract translation
            R = pose.rotation()
            t = pose.translation()
            # Append the coordinates to the poses
            poses[i] = np.eye(4)
            poses[i, :3, :3] = R.matrix()
            poses[i, :3, 3] = t
    
        return list(poses)

    def get_landmarks(self):
        """
        Return the positions of the landmarks as an array of 3D coordinates and their covariances.

        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: 
                - An Nx3 array with the x, y, z coordinates of the landmarks.
                - A list of covariance matrices for each landmark.
        """        
        # Initialize array to store landmark positions
        positions = np.empty((len(self.landmarks), 3))

        # Iterate over all landmarks
        for i, landmark_id in enumerate(self.landmarks):
            # Get the position for each landmark
            landmark_position = self.estimate.atPoint3(L(landmark_id))
            # Store the position in the array
            positions[i] = landmark_position
        
        return positions

    def print_x_nodes_in_graph(self):
        x_nodes = []
        x_node_names = []
        for i in range(self.graph.size()):
            factor = self.graph.at(i)
            keys = factor.keys()
            for key in keys:
                symbol = gtsam.Symbol(key)
                if symbol.__str__()[0] == 'x':  # Assuming 'x' is the character for pose variables
                    x_nodes.append(symbol.key())
                    x_node_names.append(symbol.__str__()[:-1])
        
        print("\tx nodes in graph:", sorted(x_node_names))
    
