from typing import Literal
import numpy as np
import g2o
from config import SETTINGS
from src.backend.convisibility_graph import ConvisibilityGraph
from src.others.local_map import Map, mapPoint
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
class poseBA:
    def __init__(self, K: np.ndarray, verbose=False):
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
            
            # Add the vertex to the graph
            self.optimizer.add_vertex(vertex)

    def add_observations(self, map: Map):
        """Add landmarks as vertices and reprojection observations as edges."""
        if self.verbose:
            print(f"Adding {map.num_points} landmarks...")

        # This kernel value is chosen based on the chi–squared distribution with 2 degrees of freedom 
        # (since the measurement is 2D) so that errors above this threshold are down–weighted.
        delta = np.sqrt(5.991)

        # Iterate over all map points
        for i, pt in enumerate(map.points_arr):
            pos = pt.pos      # 3D position of landmark
            l_idx = pt.id     # landmark id

            # Create landmark vertex
            v_landmark = g2o.VertexPointXYZ()
            v_landmark.set_id(L(l_idx))
            v_landmark.set_estimate(pos)
            v_landmark.set_marginalized(True)
            
            # In motion-only optimization, we fix the landmarks
            v_landmark.set_fixed(True)
                    
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
                # Add a Huber kernel to lessen the effect of outliers
                robust_kernel = g2o.RobustKernelHuber(delta)
                edge.set_robust_kernel(robust_kernel)
                # Link the camera parameters (parameter id 0)
                edge.set_parameter_id(0, 0)

                # Add observation edge
                self.optimizer.add_edge(edge)

    def optimize(self):
        """
        Optimize the graph and return optimized poses and landmark positions.
        
        Returns:
            A tuple (pose_ids, poses, landmark_ids, landmarks, success)
        """
        if self.verbose:
            print("Optimizing with g2o...")

        # Calculate initial number of edges
        n_edges = self.optimizer.edges()

        # We perform 4 optimizations
        optim_iterations = [10,10,7,5]
        chi2_threshold = 9.21
        n_outlier_edges = 0
        for i in range(4):
            self.optimizer.initialize_optimization(0)
            self.optimizer.optimize(optim_iterations[i])

            # Iterate over edges and mark outliers
            for e in self.optimizer.edges():
                chi2_val = e.chi2()
                if chi2_val > chi2_threshold:
                    e.setLevel(1)
                    n_outlier_edges += 1
                else:
                    e.setLevel(0)

            # Check if too little edges are left
            if (self.optimizer.edges().size() < 10):
                break

        # Calculate the number of inliers
        n_inlier_edges = n_edges - n_outlier_edges

        # Extract the optimized estimates.
        self.update_poses()

        return n_inlier_edges

    def finalize(self):
        """
        Returns the final poses (optimized).
        """
        return self.update_poses()

    def update_poses(self):
        """
        Retrieves optimized pose and landmark estimates from the optimizer.
        
        Returns:
            (pose_ids, pose_array, landmark_ids, landmark_array)
        """
        # Iterate over all vertices.
        for vertex in self.optimizer.vertices().values():
            # Find pose verticies
            if isinstance(vertex, g2o.VertexSE3Expmap):
                # Update poses
                frame_id = X_inv(vertex.id())
                self.frames[frame_id].pose = vertex.estimate().matrix()

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
