import numpy as np
import g2o
from config import SETTINGS, log, fx, fy, cx, cy
import src.utils as utils
import src.local_mapping as mapping


def X(idx: int):
    return int(2 * idx)
def X_inv(idx: int):
    return int(idx / 2)

def L(idx: int):
    return int(2 * idx + 1)
def L_inv(idx: int):
    return int((idx - 1) / 2)


class BA:
    def __init__(self):
        """Initializes BA with a g2o optimizer and camera intrinsics."""
        # Set up the g2o optimizer.
        self.optimizer = g2o.SparseOptimizer()
        # Create the linear solver and block solver for SE3 (poses)
        linear_solver = g2o.LinearSolverEigenX()
        block_solver = g2o.BlockSolverX(linear_solver)
        algorithm = g2o.OptimizationAlgorithmLevenberg(block_solver)
        self.optimizer.set_algorithm(algorithm)

        # Set up camera parameters.
        self.cam = g2o.CameraParameters(fx, [cx, cy], 0.0)
        self.cam.set_id(0)
        self.optimizer.add_parameter(self.cam)

        # This kernel value is chosen based on the chi–squared distribution with 2 degrees of freedom 
        # (since the measurement is 2D) so that errors above this threshold are down–weighted.
        self._delta = np.sqrt(5.991)

    def _add_frame(self, frame: utils.Frame, fixed: bool=False):
        """
        Adds a pose (4x4 transformation matrix) as a VertexSE3Expmap.
        Note: g2o expects world2frame transformations.
        """
        kf_id = frame.id
        T_w2f = utils.invert_transform(frame.pose) 

        # Convert the 4x4 pose matrix into an SE3Quat.
        R = T_w2f[:3, :3]
        t = T_w2f[:3, 3]
        se3 = g2o.SE3Quat(R, t)
        vertex = g2o.VertexSE3Expmap()
        vertex.set_id(X(kf_id))
        vertex.set_estimate(se3)

        # We optimize both poses and landmarks, but fix the first pose
        if fixed:
            vertex.set_fixed(fixed)
            
        # Add the vertex to the graph
        self.optimizer.add_vertex(vertex)

    def _add_landmark(self, l_idx: int, pos: np.ndarray, fixed=False):
        """
        Adds a landmark as vertex.
        
        Args:
            l_idx: landmark id
            pos: 3D position of landmark in world coordinates
        """
        # Create landmark vertex
        v_landmark = g2o.VertexPointXYZ()
        v_landmark.set_id(L(l_idx))
        v_landmark.set_estimate(pos)
        v_landmark.set_marginalized(True)
        v_landmark.set_fixed(fixed)
                
        # Add landmark vertex
        self.optimizer.add_vertex(v_landmark)

    def _add_observation(self, pid: int, kf: utils.Frame, pt: tuple, octave: int, 
                         kernel: bool=True, level: int = None):
        """
        Adds reprojection observation as edge.
        
        Args:
            pid: The observed map point
            kf: The keyframe that observed the map point
            pt: The pixel that corresponded to the map point
            octave: The ORB octave (scale) that the matched ORB keypoint belonged to
            kernel:
            level:        
        """
        u, v = pt  # pixels of the keypoint

        # Create the reprojection edge.
        edge = g2o.EdgeProjectXYZ2UV()
        # In g2o, convention is vertex 0 = landmark, vertex 1 = pose.
        edge.set_vertex(0, self.optimizer.vertex(L(pid)))
        edge.set_vertex(1, self.optimizer.vertex(X(kf.id)))
        edge.set_measurement([u, v])
        
        # Add the information matrix based on the octave's uncertainty
        measurement_uncertainty = kf.scale_uncertainties[octave]
        measurement_information = np.eye(2) * (1.0 / measurement_uncertainty)
        edge.set_information(measurement_information)

        # Add a Huber kernel to lessen the effect of outliers
        if kernel:
            robust_kernel = g2o.RobustKernelHuber(self._delta)
            edge.set_robust_kernel(robust_kernel)
        
        # Link the camera parameters (parameter id 0)
        set_param_success = edge.set_parameter_id(0, 0)
        if set_param_success is False:
            raise(ValueError("Couldn't set parameter ID."))

        # Set the level
        if level is not None:
            edge.set_level(level)

        # Add observation edge
        self.optimizer.add_edge(edge)

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

