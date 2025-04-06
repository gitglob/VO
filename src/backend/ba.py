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
        self.new_pose_ids = []
        self.new_landmark_ids = []
        self.obs_buffer = {}
        self._scale_set = False
        self._frame_anchored = False

        # Debugging info
        self.verbose = verbose

    def add_pose(self, node_idx: int, pose: np.ndarray, fixed=False):
        """Add a subsequent pose."""
        pose3 = gtsam.Pose3(pose)

        # Check if this is the first pose to anchor the graph
        if not self._frame_anchored or fixed:
            factor = gtsam.PriorFactorPose3(X(node_idx), pose3, self.prior_pose_noise)
            self.graph.add(factor)
            self._frame_anchored = True
            if self.verbose:
                print(f"Anchoring pose X({node_idx})...")

        self.initial_estimates.insert(X(node_idx), pose3)

        # Store the pose ids that have passed
        self.new_pose_ids.append(node_idx)

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
            u, v = kpt.pt
            l_idx = kpt.class_id
            pt = points[i]
            point3 = gtsam.Point3(pt)

            # Add a prior on the first landmark to set the scale
            if not self._scale_set:
                factor = gtsam.PriorFactorPoint3(L(l_idx), point3, self.prior_landmark_noise)
                self.graph.add(factor)
                self._scale_set = True
                if self.verbose:
                    print(f"Anchored landmark L({l_idx})")

            # Create the projection factor
            factor = gtsam.GenericProjectionFactorCal3_S2(
                gtsam.Point2(u, v),
                self.measurement_noise,
                X(pose_idx),
                L(l_idx),
                self.cam
            )

            # Add the observation factor to the buffer
            if l_idx not in self.obs_buffer.keys():
                self.obs_buffer[l_idx] = {
                    "factor": factor,
                    "estimate": point3,
                    "num_observations": 1
                }
            else:
                # Check if this is the 2nd observation of this landmark
                if self.obs_buffer[l_idx]["num_observations"] == 1:
                    # Add the very first estimate and factor to the graph
                    point3 = self.obs_buffer[l_idx]["estimate"]
                    if not self.isam.valueExists(L(l_idx)) and not self.initial_estimates.exists(L(l_idx)):
                        self.initial_estimates.insert(L(l_idx), point3)
                    first_factor = self.obs_buffer[l_idx]["factor"]
                    self.graph.add(first_factor)

                # Add the current factor
                self.graph.add(factor)

                # Increase the observation count
                self.obs_buffer[l_idx]["num_observations"] += 1

            # Save the landmark id that have passed
            if l_idx not in self.new_landmark_ids:
                self.new_landmark_ids.append(l_idx)

    def optimize(self):
        """ Optimize the graph and return the optimized robot poses and landmark positions"""
        # Update iSAM
        self.isam.update(self.graph, self.initial_estimates)
        estimate = self.isam.calculateEstimate()

        # Get the optimized robot poses and landmark positions
        opt_pose_ids, opt_poses, opt_l_ids, opt_l_pos = self.get_poses_and_landmarks(estimate)

        # Prepare a new graph
        self.graph.resize(0)
        self.initial_estimates.clear()
        self.obs_buffer = {}
        self.new_landmark_ids = []
        self.new_pose_ids = []

        return opt_pose_ids, list(opt_poses), opt_l_ids, list(opt_l_pos)
    
    def finalize(self):
        """Returns the final estimate"""
        self._estimate = self.isam.calculateEstimate()

        # Get the optimized robot poses and landmark positions
        optimized_poses = self.get_poses()

        return optimized_poses

    def get_poses_and_landmarks(self, estimate):
        poses = {}
        landmarks = {}
        
        # Iterate over all keys in the gtsam.Values estimate.
        for key in estimate.keys():
            symbol = gtsam.Symbol(key)
            idx = symbol.index()
            
            # Extract the pose
            if symbol.string()[0] == 'x':
                pose = estimate.atPose3(key)
                poses[idx] = pose.matrix()
            # Extract the landmark
            elif symbol.string()[0] == 'l':
                point = estimate.atPoint3(key)
                landmarks[idx] = point
        
        # Sort the results by id if desired.
        pose_ids = sorted(poses.keys())
        landmark_ids = sorted(landmarks.keys())
        
        pose_array = np.array([poses[i] for i in pose_ids])
        landmark_array = np.array([landmarks[i] for i in landmark_ids])
        
        return pose_ids, pose_array, landmark_ids, landmark_array

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
                if symbol.string()[0] == 'x':
                    x_nodes.append(symbol.key())
                    x_node_names.append(symbol.__str__()[:-1])
                # Landmark variables are L
                elif symbol.string()[0] == 'l':
                    l_nodes.append(symbol.key())
                    l_node_names.append(symbol.__str__()[:-1])
        
        print("\tx nodes in graph:", sorted(x_node_names))
        print("\tl nodes in graph:", sorted(l_node_names))
    
    def is_key_in_graph(self, key) -> bool:
        for i in range(self.graph.size()):
            factor = self.graph.at(i)
            if key in factor.keys():
                return True
        return False

    def get_landmark_symbol_in_factor(self, factor) -> gtsam.gtsam.Symbol:
        for key in factor.keys():
            symbol = gtsam.Symbol(key)
            if symbol.string()[0] == 'l':
                return symbol
        return None

    def get_pose_symbol_in_factor(self, factor) -> gtsam.gtsam.Symbol:
        for key in factor.keys():
            symbol = gtsam.Symbol(key)
            if symbol.string()[0] == 'x':
                return symbol
        return None

    def get_factors_for_variable(self, variable_key: int) -> int:
        """
        Returns a list of factors in the graph that reference the given variable key.

        Args:
            variable_key: The key (e.g., X(7)) to search for in the factor graph.
        
        Returns:
            List of factors that reference the variable.
        """
        var_symbol = gtsam.Symbol(variable_key)
        factors_for_variable = []
        for i in range(self.graph.size()):
            factor = self.graph.at(i)
            # factor.keys() returns a list of keys for this factor.
            for key in factor.keys():
                if key == variable_key:
                    factors_for_variable.append(factor)
                    break  # No need to check further keys in this factor.

        # Print the number of landmarks observed in the pose.
        print(f"Variable {var_symbol} is referenced in {len(factors_for_variable)} factors")

        return factors_for_variable 
    
    def get_landmarks_for_pose(self, pose_key: int) -> int:
        """
        For a given pose id, print all the landmarks (i.e. the landmark keys)
        that are associated with that pose via projection factors.
        
        Args:
            pose_key (int): The id of the pose to search for.

        Returns:
            int: The number of landmarks associated with the pose.
        """
        pose_symbol = gtsam.Symbol(pose_key)
        related_landmark_keys = set()
        
        # Iterate over all factors in the graph.
        for i in range(self.graph.size()):
            factor = self.graph.at(i)
            # Check if the factor is a projection factor.
            if isinstance(factor, gtsam.GenericProjectionFactorCal3_S2):
                keys = factor.keys()
                # If this factor references the given pose...
                if pose_key in keys:
                    # ...find the key that corresponds to the landmark.
                    for key in keys:
                        if key != pose_key:
                            symbol = gtsam.Symbol(key)
                            # landmark variables start with the letter 'l'
                            if symbol.string()[0] == 'l':
                                related_landmark_keys.add(key)

        # Print the number of landmarks observed in the pose.
        print(f"Pose {pose_symbol} observes {len(related_landmark_keys)} landmarks")

        # # Print all the collected landmark keys.
        # for lk in related_landmark_keys:
        #     landmark_symbol = gtsam.Symbol(lk)
        #     print(f"Pose {pose_symbol} observes landmark {landmark_symbol}")

        return len(related_landmark_keys)

    def get_poses_for_landmark(self, landmark_key: int) -> int:
        """
        For a given landmark id, print all the poses (i.e. the pose keys)
        that are associated with that landmark via projection factors.
        
        Args:
            landmark_id (int): The id of the landmark to search for.

        Returns:
            int: The number of poses associated with the landmark.
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
                            if symbol.string()[0] == 'x':
                                related_pose_keys.add(key)

        # Print all the collected pose keys.
        for pk in related_pose_keys:
            pose_symbol = gtsam.Symbol(pk)
            print(f"Landmark {landmark_symbol} is observed in pose {pose_symbol}")

        return len(related_pose_keys)



