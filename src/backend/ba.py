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
        self.cam = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)

        # Measurement and odometry noise
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([ODOMETRY_TRANS_SIGMA]*3 + [ODOMETRY_ROT_SIGMA]*3)
        )
        self.measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, MEASUREMENT_SIGMA)
        
        # Prior noise
        self.prior_pose_noise = gtsam.noiseModel.Isotropic.Sigma(6, POSE_PRIOR_SIGMA)
        self.prior_landmark_noise = gtsam.noiseModel.Isotropic.Sigma(3, MEASUREMENT_PRIOR_SIGMA)
        
        # Counters for poses and landmarks keys.
        self.pose_ids = []
        self.landmark_ids = []
        self.obs_buffer = {}
        self._last_pose = None

        # Debugging info
        self.verbose = verbose

    # NOT USED
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
        edge = gtsam.BetweenFactorPose3(X(self.pose_ids[-1]), 
                                        X(node_idx), 
                                        T, 
                                        self.odometry_noise)
        self.graph.add(edge)
        if self.verbose:
            self.print_x_nodes_in_graph()
            print(f"Added edge X({self.pose_ids[-1]}) ~ X({node_idx})")

        # Insert the new pose into initial estimates
        self._add_pose(new_pose, node_idx)

    def add_pose(self, node_idx: int, pose: np.ndarray, fixed=False):
        """Add a subsequent pose."""
        pose = gtsam.Pose3(pose)

        self.initial_estimates.insert(X(node_idx), pose)
        if fixed or len(self.pose_ids) == 0:
            factor = gtsam.PriorFactorPose3(X(node_idx), pose, self.prior_pose_noise)
            self.graph.add(factor)

        if self.verbose:
            print(f"Added pose X({node_idx})")
        
        self.pose_ids.append(node_idx)
        self._last_pose = pose

    def add_landmark(self, l_idx: int, pos: np.ndarray):
        """Add landmarks.
        
        Args:
            landmarks: A (N, 3) array with landmark positions.
            landmark_kpts: The corresponding keypoints.
        """
        pt = gtsam.Point3(pos)

        # Add a prior on the first landmark to set the scale
        if len(self.landmark_ids) == 0:
            factor = gtsam.PriorFactorPoint3(L(l_idx), pt, self.prior_landmark_noise)
            self.graph.add(factor)
            if self.verbose:
                print(f"Added landmark L({l_idx})")

        # We only want to initialize the first landmark observations
        if l_idx in self.landmark_ids:
            return

        # Add the landmark to the graph
        self.initial_estimates.insert(L(l_idx), pt)
        self.landmark_ids.append(l_idx)

    def add_observations(self, pose_idx: int, points: np.ndarray, observations: np.ndarray[cv2.KeyPoint]):
        """
        Add reprojection observations to the BA problem.
        
        Args:
            pose_idx:     index of the keyframe (corresponding to symbol 'x')
            observations: list of keypoints
        """
        if self.verbose:
            print(f"Adding {len(observations)} observations to pose X({pose_idx})")

        # Iterate over all observations
        for i in range(len(observations)):
            # Extract the keypoint id and the corresponding 3D point
            kpt = observations[i]
            l_idx = kpt.class_id
            pt = points[i]

            # Possibly initialize the landmark
            self.add_landmark(l_idx, pt)

            # Extract the keypoint pixel coordinates
            u, v = kpt.pt

            # Add the observation factor to the graph
            factor = gtsam.GenericProjectionFactorCal3_S2(
                gtsam.Point2(u, v),
                self.measurement_noise,
                X(pose_idx),
                L(l_idx),
                self.cam
            )

            # If that landmark has been observed before, add it to the graph 
            if l_idx == 60701:
                pass
            if l_idx in self.obs_buffer.keys():
                # Get the number of observations of that landmark
                num_obs = self.obs_buffer[l_idx]["count"]

                # If there is only one, add the first factor to the graph
                if num_obs == 1:
                    first_factor = self.obs_buffer[l_idx]["first_factor"]
                    self.graph.add(first_factor)

                # Add the current factor too
                self.graph.add(factor)

                # Increase the factor observation counter
                self.obs_buffer[l_idx]["count"] += 1
            else:
                # If it is the first observation, add the factor to a buffer
                self.obs_buffer[l_idx] = {
                    "first_factor": factor,
                    "count": 1
                }

    def optimize(self):
        """ Optimize the graph and return the optimized robot poses and landmark positions"""
        # Update iSAM
        self.isam.update(self.graph, self.initial_estimates)
        self._estimate = self.isam.calculateEstimate()
        self._lastPose = self._estimate.atPose3(X(self.pose_ids[-1]))

        # Get the optimized robot poses and landmark positions
        optimized_poses = self.get_poses()
        optimized_landmark_poses = self.get_landmarks()

        # Prepare a new graph
        self.graph.resize(0)
        self.initial_estimates.clear()
        self.obs_buffer = {}
        # self.landmark_ids = []
        # self.pose_ids = []

        return optimized_poses, self.landmark_ids, optimized_landmark_poses
    
    def finalize(self):
        """Returns the final estimate"""
        self._estimate = self.isam.calculateEstimate()

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
        poses = np.empty((len(self.pose_ids), 4, 4))

        # Iterate over all nodes
        for i, p_idx in enumerate(self.pose_ids):
            # Get the pose for each node
            pose = self._estimate.atPose3(X(p_idx))
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
        positions = np.empty((len(self.landmark_ids), 3))

        # Iterate over all landmarks
        for i, l_id in enumerate(self.landmark_ids):
            # Get the position for each landmark
            landmark_position = self._estimate.atPoint3(L(l_id))
            # Store the position in the array
            positions[i] = landmark_position
        
        return positions

    def cull(self, landmark_ids_to_remove: list[int]):
        """
        Remove landmark estimates for the specified landmark IDs.
        
        This function removes the landmark IDs from the internal landmark list and
        erases any corresponding initial estimates. Note that factors already added to
        the graph remain in the iSAM2 instance (GTSAM does not support direct removal of factors),
        so this simply prevents culled landmarks from being used in future operations.
        
        Args:
            landmark_ids_to_remove (list[int]): List of landmark IDs to remove.
        """
        for l_id in landmark_ids_to_remove:
            # Remove the landmark id from the landmarks list if present.
            if l_id in self.landmark_ids:
                self.landmark_ids.remove(l_id)
                # Create the corresponding key using the symbol shorthand.
                key = L(l_id)
                # Erase from initial estimates if it exists.
                if self.initial_estimates.exists(key):
                    self.initial_estimates.erase(key)
                if self.verbose:
                    print(f"Culled landmark L({l_id})")
                
    def print_x_nodes_in_graph(self):
        x_nodes = []
        x_node_names = []
        l_nodes = []
        l_node_names = []
        for i in range(self.graph.size()):
            factor = self.graph.at(i)
            keys = factor.keys()
            for key in keys:
                symbol = gtsam.Symbol(key)
                # Pose variables are X
                if symbol.__str__()[0] == 'x':
                    x_nodes.append(symbol.key())
                    x_node_names.append(symbol.__str__()[:-1])
                # Landmark variables are L
                elif symbol.__str__()[0] == 'l':
                    l_nodes.append(symbol.key())
                    l_node_names.append(symbol.__str__()[:-1])
        
        print("\tx nodes in graph:", sorted(x_node_names))
        print("\tl nodes in graph:", sorted(l_node_names))
    
    def get_factors_for_variable(self, variable_key):
        """
        Returns a list of factors in the graph that reference the given variable key.

        Args:
            variable_key: The key (e.g., X(7)) to search for in the factor graph.
        
        Returns:
            List of factors that reference the variable.
        """
        factors_for_variable = []
        for i in range(self.graph.size()):
            factor = self.graph.at(i)
            # factor.keys() returns a list of keys for this factor.
            for key in factor.keys():
                if key == variable_key:
                    factors_for_variable.append(factor)
                    break  # No need to check further keys in this factor.
        return factors_for_variable

    def print_poses_for_landmark(self, landmark_key: int):
        """
        For a given landmark id, print all the poses (i.e. the pose keys)
        that are associated with that landmark via projection factors.
        
        Args:
            landmark_id (int): The id of the landmark to search for.
        """
        landmark_symbol = gtsam.Symbol(landmark_key)
        related_pose_keys = set()
        
        # Iterate over all factors in the graph.
        for i in range(self.graph.size()):
            factor = self.graph.at(i)
            # We check if the factor is a projection factor.
            if isinstance(factor, gtsam.GenericProjectionFactorCal3_S2):
                keys = factor.keys()
                # If this factor references the given landmark...
                if landmark_key in keys:
                    # ...find the key that corresponds to the pose
                    for key in keys:
                        if key != landmark_key:
                            symbol = gtsam.Symbol(key)
                            if symbol.__str__()[0] == 'x':
                                related_pose_keys.add(key)
                                
        # Print all the collected pose keys.
        for pk in related_pose_keys:
            pose_symbol = gtsam.Symbol(pk)
            print(f"Landmark {landmark_symbol} is observed in pose {pose_symbol}")



