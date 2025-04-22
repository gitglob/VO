import numpy as np
import g2o
from config import SETTINGS, log, K
from src.local_mapping.local_map import mapPoint
from src.others.frame import Frame

# Set parameters from the config
MEASUREMENT_SIGMA = float(SETTINGS["ba"]["measurement_noise"])
# NUM_OBSERVATIONS = int(SETTINGS["ba"]["num_observations"])


def X(idx: int):
    return 2 * idx
def X_inv(idx: int):
    return idx / 2

def L(idx: int):
    return 2 * idx + 1
def L_inv(idx: int):
    return (idx - 1) / 2


class BA:
    def __init__(self):
        """Initializes BA with a g2o optimizer and camera intrinsics."""
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

    def _add_frame(self, frame: Frame, fixed: bool=False):
        """Adds a pose (4x4 transformation matrix) as a VertexSE3Expmap."""
        p_id = frame.id
        p = frame.pose 

        # Convert the 4x4 pose matrix into an SE3Quat.
        R = p[:3, :3]
        t = p[:3, 3]
        se3 = g2o.SE3Quat(R, t)
        vertex = g2o.VertexSE3Expmap()
        vertex.set_id(X(p_id))
        vertex.set_estimate(se3)

        # We optimize both poses and landmarks, but fix the first pose
        if fixed:
            vertex.set_fixed(fixed)
            
        # Add the vertex to the graph
        self.optimizer.add_vertex(vertex)

    def _add_observation(self, mp: mapPoint, fixed: bool=False):
        """Adds a landmark as vertex and reprojection observations as edges."""
        pos = mp.pos      # 3D position of landmark
        l_idx = mp.id     # landmark id

        # Create landmark vertex
        v_landmark = g2o.VertexPointXYZ()
        v_landmark.set_id(L(l_idx))
        v_landmark.set_estimate(pos)
        v_landmark.set_marginalized(True)
        v_landmark.set_fixed(fixed)
                
        # Add landmark vertex
        self.optimizer.add_vertex(v_landmark)

        # Iterate over all map point observations
        for obs in mp.observations:
            pose_idx = obs["kf_id"] # id of keyframe that observed the landmark
            kpt = obs["keypoint"]   # keypoint of the observation
            u, v = kpt.pt           # pixels of the keypoint

            # Create the reprojection edge.
            edge = g2o.EdgeProjectXYZ2UV()
            # edge = g2o.EdgeSE3ProjectXYZ()
            # In g2o, convention is vertex 0 = landmark, vertex 1 = pose.
            edge.set_vertex(0, v_landmark)
            edge.set_vertex(1, self.optimizer.vertex(X(pose_idx)))
            edge.set_measurement([u, v])
            edge.set_information(self.measurement_information)
            # Link the camera parameters (parameter id 0)
            edge.set_parameter_id(0, 0)
            # edge.set_level(0)

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

    ############################################### DEBUG ###############################################

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
        log.info("[BA] \tx nodes in graph:", sorted(x_node_names))
        log.info("[BA] \tl nodes in graph:", sorted(l_node_names))

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

