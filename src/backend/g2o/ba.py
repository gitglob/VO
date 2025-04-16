from typing import Literal
import numpy as np
import g2o
from config import SETTINGS
from src.backend.convisibility_graph import ConvisibilityGraph
from src.local_mapping.local_map import Map, mapPoint
from src.others.frame import Frame

# Set parameters from the config
MEASUREMENT_SIGMA = float(SETTINGS["ba"]["measurement_noise"])
NUM_OBSERVATIONS = int(SETTINGS["ba"]["num_observations"])


def X(idx: int):
    return 2 * idx
def X_inv(idx: int):
    return idx / 2

def L(idx: int):
    return 2 * idx + 1
def L_inv(idx: int):
    return (idx - 1) / 2

# -----------------------------------------------------------------------------
# Bundle Adjustment class using g2o.
# -----------------------------------------------------------------------------
class BA:
    def __init__(self, K: np.ndarray, verbose=False, type: Literal["full", "pose"]="full"):
        """
        Initializes BA_g2o with a g2o optimizer and camera intrinsics.
        
        Args:
            K: Camera intrinsics matrix.
            verbose: If True, print debug information.
        """
        self.verbose = verbose

        # Set up the g2o optimizer.
        self.optimizer = g2o.SparseOptimizer()
        # Create the linear solver and block solver for SE3 (poses)
        linear_solver = g2o.LinearSolverEigenSE3()
        solver = g2o.BlockSolverSE3(linear_solver)
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        self.optimizer.set_algorithm(algorithm)

        # Set up camera parameters.
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        self.cam = g2o.CameraParameters(fx, [cx, cy], 0.0)
        self.cam.set_id(0)
        self.optimizer.add_parameter(self.cam)

        # Landmark observation information matrix
        self.measurement_sigma = MEASUREMENT_SIGMA
        self.measurement_information = np.eye(2) * (1.0 / (self.measurement_sigma ** 2))

        # Buffers and tracking dictionaries.
        self.obs_buffer = {}        # Buffer for observation edges per landmark id.

        # The keyframes to optimize
        self.frames: dict[int, Frame] = None

    def add_frames(self, frames: dict[int, Frame]):
        """
        Add a pose (4x4 transformation matrix) as a VertexSE3Expmap.
        The first pose is fixed to anchor the graph.
        """
        self.frames = frames

        if self.verbose:
            print(f"Adding {len(frames)} poses...")

        # Iterate over all poses
        for i, f in enumerate(frames):
            p_id = f.id
            p = f.pose

            # Convert the 4x4 pose matrix into an SE3Quat.
            R = p[:3, :3]
            t = p[:3, 3]
            se3 = g2o.SE3Quat(R, t)
            vertex = g2o.VertexSE3Expmap()
            vertex.set_id(X(p_id))
            vertex.set_estimate(se3)

            # We optimize both poses and landmarks, but fix the first pose
            vertex.set_fixed(True)
            if self.verbose:
                print(f"Anchoring pose x({p_id})...")
            
            # Add the vertex to the graph
            self.optimizer.add_vertex(vertex)

    def add_observations(self, map: Map):
        """Add landmarks as vertices and reprojection observations as edges."""
        if self.verbose:
            print(f"Adding {map.num_points} landmarks...")

        # Iterate over all map points
        for i, pt in enumerate(map.points_arr):
            pos = pt.pos      # 3D position of landmark
            l_idx = pt.id     # landmark id

            # Create landmark vertex
            v_landmark = g2o.VertexPointXYZ()
            v_landmark.set_id(L(l_idx))
            v_landmark.set_estimate(pos)
            v_landmark.set_marginalized(True)
                    
            # Add landmark vertex
            self.optimizer.add_vertex(v_landmark)

            # Iterate over all map point observations
            for obs in pt.observations:
                pose_idx = obs["kf_id"]     # id of keyframe that observed the landmark
                kpt = obs["keypoint"]       # keypoint of the observation
                u, v = kpt.pt               # pixels of the keypoint

                # Create the reprojection edge.
                edge = g2o.EdgeProjectXYZ2UV()
                # In g2o, convention is vertex 0 = landmark, vertex 1 = pose.
                edge.set_vertex(0, v_landmark)
                edge.set_vertex(1, self.optimizer.vertex(X(pose_idx)))
                edge.set_measurement([u, v])
                edge.set_information(self.measurement_information)
                # Link the camera parameters (parameter id 0)
                edge.set_parameter_id(0, 0)

                # Add observation edge
                self.optimizer.add_edge(edge)

            # # Buffer the edge until enough observations are accumulated.
            # if l_idx not in self.obs_buffer:
            #     self.obs_buffer[l_idx] = {
            #         "edge_list": [edge],
            #         "estimate": pos,
            #         "num_observations": 1
            #     }
            # else:
            #     self.obs_buffer[l_idx]["num_observations"] += 1
            #     if self.obs_buffer[l_idx]["num_observations"] < NUM_OBSERVATIONS:
            #         self.obs_buffer[l_idx]["edge_list"].append(edge)
            #     else:
            #         # Add all buffered reprojection edges and the current one.
            #         for buffered_edge in self.obs_buffer[l_idx]["edge_list"]:
            #             self.optimizer.add_edge(buffered_edge)
            #         # Add the current edge
            #         self.optimizer.add_edge(edge)

    def optimize(self, num_iterations=10):
        """
        Optimize the graph and return optimized poses and landmark positions.
        
        Returns:
            A tuple (pose_ids, poses, landmark_ids, landmarks, success)
        """
        if self.verbose:
            print("Optimizing with g2o...")

        self.optimizer.initialize_optimization()
        self.optimizer.optimize(num_iterations)

        # Extract the optimized estimates.
        opt_pose_ids, opt_poses, opt_l_ids, opt_l_pos = self.get_poses_and_landmarks()

        return opt_pose_ids, list(opt_poses), opt_l_ids, list(opt_l_pos), True

    def finalize(self):
        """
        Returns the final poses (optimized).
        """
        return self.get_poses_and_landmarks()[1]

    def get_poses_and_landmarks(self):
        """
        Retrieves optimized pose and landmark estimates from the optimizer.
        
        Returns:
            (pose_ids, pose_array, landmark_ids, landmark_array)
        """
        landmarks = {}
        # Iterate over all vertices.
        for vertex in self.optimizer.vertices().values():
            if isinstance(vertex, g2o.VertexSE3Expmap):
                frame_id = X_inv(vertex.id())
                self.frames[frame_id] = vertex.estimate().matrix()
            elif isinstance(vertex, g2o.VertexPointXYZ):
                landmarks[vertex.id()] = vertex.estimate()

        landmark_ids = sorted(landmarks.keys())
        landmark_pos = np.array([landmarks[i] for i in landmark_ids])

        return landmark_ids, landmark_pos

    def print_x_nodes_in_graph(self):
        """
        Print the pose (x) and landmark (l) node IDs currently in the graph.
        """
        x_node_names = []
        l_node_names = []
        for vertex in self.optimizer.vertices().values():
            if isinstance(vertex, g2o.VertexSE3Expmap):
                x_node_names.append(f"x{vertex.id()}")
            elif isinstance(vertex, g2o.VertexPointXYZ):
                l_node_names.append(f"l{vertex.id()}")
        print("\tx nodes in graph:", sorted(x_node_names))
        print("\tl nodes in graph:", sorted(l_node_names))

    def is_key_in_graph(self, key) -> bool:
        """
        Check whether a vertex with the given key exists in the graph.
        """
        return key in self.optimizer.vertices()

    def get_factors_for_variable(self, variable_key: int) -> list:
        """
        Returns a list of edges (factors) that reference the given vertex id.
        """
        factors_for_variable = []
        for edge in self.optimizer.edges():
            # Loop over the edge’s vertices.
            num_vertices = edge.num_vertices()
            for j in range(num_vertices):
                if edge.vertex(j).id() == variable_key:
                    factors_for_variable.append(edge)
                    break
        return factors_for_variable

    def get_landmarks_for_pose(self, pose_key: int) -> int:
        """
        Returns the number of unique landmark vertices observed from a given pose.
        """
        related_landmark_keys = set()
        for edge in self.optimizer.edges():
            # Check for projection edges.
            if isinstance(edge, g2o.EdgeProjectXYZ2UV):
                # Convention: vertex 1 is the pose.
                if edge.vertex(1).id() == pose_key:
                    related_landmark_keys.add(edge.vertex(0).id())
        return len(related_landmark_keys)

    def get_poses_for_landmark(self, landmark_key: int) -> set:
        """
        Returns a set of pose vertex ids that observe the specified landmark.
        """
        related_pose_keys = set()
        for edge in self.optimizer.edges():
            if isinstance(edge, g2o.EdgeProjectXYZ2UV):
                # Convention: vertex 0 is the landmark.
                if edge.vertex(0).id() == landmark_key:
                    related_pose_keys.add(edge.vertex(1).id())
        return related_pose_keys

    def sanity_check(self):
        """
        Checks that:
          - Each pose vertex has an estimate and observes at least 2 landmarks.
          - Each landmark vertex has an estimate and is observed by at least 3 poses.
        """
        for edge in self.optimizer.edges():
            if isinstance(edge, g2o.EdgeProjectXYZ2UV):
                pose_vertex = edge.vertex(1)
                if pose_vertex is None or not self.is_key_in_graph(pose_vertex.id()):
                    raise ValueError(f"Pose {pose_vertex} missing or not in graph")
                if self.get_landmarks_for_pose(pose_vertex.id()) < 2:
                    raise ValueError(f"Pose x({pose_vertex.id()}) observes less than 2 landmarks")
                landmark_vertex = edge.vertex(0)
                if landmark_vertex is None or not self.is_key_in_graph(landmark_vertex.id()):
                    raise ValueError(f"Landmark {landmark_vertex} missing or not in graph")
                if len(self.get_poses_for_landmark(landmark_vertex.id())) < 3:
                    raise ValueError(f"Landmark l({landmark_vertex.id()}) is observed by less than 3 poses")

class localBA:
    def __init__(self, K: np.ndarray, cgraph: ConvisibilityGraph, verbose=False):
        """
        Initializes BA_g2o with a g2o optimizer and camera intrinsics.
        
        Args:
            K: Camera intrinsics matrix.
            verbose: If True, print debug information.
        """
        self.verbose = verbose

        # Set up the g2o optimizer.
        self.optimizer = g2o.SparseOptimizer()
        # Create the linear solver and block solver for SE3 (poses)
        linear_solver = g2o.LinearSolverEigenSE3()
        solver = g2o.BlockSolverSE3(linear_solver)
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        self.optimizer.set_algorithm(algorithm)

        # Set up camera parameters.
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        self.cam = g2o.CameraParameters(fx, [cx, cy], 0.0)
        self.cam.set_id(0)
        self.optimizer.add_parameter(self.cam)

        # Landmark observation information matrix
        self.measurement_sigma = MEASUREMENT_SIGMA
        self.measurement_information = np.eye(2) * (1.0 / (self.measurement_sigma ** 2))

        # Buffers and tracking dictionaries.
        self.obs_buffer = {}        # Buffer for observation edges per landmark id.

        # Convisibility Graph
        self.cgraph = cgraph

        # The keyframes to optimize
        self.main_frame: Frame = None
        self.frames: dict[int, Frame] = None

    def add_pose(self, pose_id: int, pose: np.ndarray, fixed=False):
        # Convert the 4x4 pose matrix into an SE3Quat.
        R = pose[:3, :3]
        t = pose[:3, 3]
        se3 = g2o.SE3Quat(R, t)
        vertex = g2o.VertexSE3Expmap()
        vertex.set_id(X(pose_id))
        vertex.set_estimate(se3)

        # Possibly fix the pose
        if fixed:
            vertex.set_fixed(True)
        
        # Add the vertex to the graph
        self.optimizer.add_vertex(vertex)

    def add_landmark(self, landmark_id: int, landmark_pos: np.ndarray):
        # Create landmark vertex
        vertex = g2o.VertexPointXYZ()
        vertex.set_id(L(landmark_id))
        vertex.set_estimate(landmark_pos)
        vertex.set_marginalized(True)
                
        # Add landmark vertex
        self.optimizer.add_vertex(vertex)

    def add_edge(self, pose_id: int, landmark_id: int, measurement: tuple[int]):
        u, v = measurement          # pixels of the keypoint

        # Create the reprojection edge.
        edge = g2o.EdgeProjectXYZ2UV()
        # In g2o, convention is vertex 0 = landmark, vertex 1 = pose.
        edge.set_vertex(0, self.optimizer.vertex(L(landmark_id)))
        edge.set_vertex(1, self.optimizer.vertex(X(pose_id)))
        edge.set_measurement([u, v])
        edge.set_information(self.measurement_information)
        # Link the camera parameters (parameter id 0)
        edge.set_parameter_id(0, 0)

        # Add observation edge
        self.optimizer.add_edge(edge)

    def build(self, keyframe: Frame, map: Map, keyframes: dict[int, Frame]):
        """
        Add a pose (4x4 transformation matrix) as a VertexSE3Expmap.
        The first pose is fixed to anchor the graph.
        """
        self.main_frame = keyframe
        self.frames = keyframes

        # Get the connected keyframe and point ids from the convisibility graph
        connected_kf_ids, connected_point_ids = self.cgraph.get_connected_nodes_and_their_points(keyframe.id)
        connected_kfs = [keyframes[idx] for idx in connected_kf_ids]

        # Get the connected points
        connected_points: mapPoint = map.get_points(connected_point_ids)

        # Get all the other keyframes that see the points but are not connected to the current keyframe
        kfs_that_see_points_ids = map.get_keyframes_that_see(connected_point_ids)
        unconnected_kfs_that_see_points_ids = kfs_that_see_points_ids - connected_kf_ids
        unconnected_kfs_that_see_points = [keyframes[idx] for idx in unconnected_kfs_that_see_points_ids]

        if self.verbose:
            print(f"Adding pose {keyframe.id}, {len(connected_kf_ids)} connected ",
                  f"and {len(unconnected_kfs_that_see_points)} unconnected poses...")

        # Add the main pose
        self.add_pose(keyframe.pose)

        # Add the connected poses
        for kf in connected_kfs:
            self.add_pose(kf.id, kf.pose)

        # Iterate over all the connected points
        for point in connected_points:
            # Add the connected landmark vertices
            self.add_landmark(point.id, point.pos)
            # Iterate over all the landmark observations
            for obs in point.observations:
                kf_idx = obs["keyframe"].id
                # If the observation was done by a connected keyframe
                if kf_idx in connected_kfs:
                    # Add the connected pose->landmark observations
                    self.add_edge(obs["keyframe"].id, point.id, obs["keypoint"].pt)

        # Add the and fix the un-connected poses
        for kf in unconnected_kfs_that_see_points:
            self.add_pose(kf.id, kf.pose, fixed=True)

    def optimize(self, num_iterations=10):
        """
        Optimize the graph and return optimized poses and landmark positions.
        
        Returns:
            A tuple (pose_ids, poses, landmark_ids, landmarks, success)
        """
        if self.verbose:
            print("Optimizing with g2o...")

        self.optimizer.initialize_optimization()
        self.optimizer.optimize(num_iterations)

        # Extract the optimized estimates.
        opt_l_ids, opt_l_pos = self.get_poses_and_landmarks()

        return opt_l_ids, list(opt_l_pos), True

    def get_poses_and_landmarks(self):
        """
        Retrieves optimized pose and landmark estimates from the optimizer.
        
        Returns:
            (pose_ids, pose_array, landmark_ids, landmark_array)
        """
        landmarks = {}
        # Iterate over all vertices.
        for vertex in self.optimizer.vertices().values():
            if isinstance(vertex, g2o.VertexSE3Expmap):
                frame_id = X_inv(vertex.id())
                self.frames[frame_id] = vertex.estimate().matrix()
            elif isinstance(vertex, g2o.VertexPointXYZ):
                landmarks[vertex.id()] = vertex.estimate()

        landmark_ids = sorted(landmarks.keys())
        landmark_pos = np.array([landmarks[i] for i in landmark_ids])

        return landmark_ids, landmark_pos
    