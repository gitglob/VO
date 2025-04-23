from typing import Literal
import numpy as np
import g2o
from config import SETTINGS, log, K
from src.backend.convisibility_graph import ConvisibilityGraph
from src.local_mapping.map import Map, mapPoint
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


class localBA:
    def __init__(self, cgraph: ConvisibilityGraph, verbose=False):
        """
        Initializes BA_g2o with a g2o optimizer and camera intrinsics.
        
        Args:
            K: Camera intrinsics matrix.
            verbose: If True, show debug information.
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
            log.info(f"[BA] Adding pose {keyframe.id}, {len(connected_kf_ids)} connected ",
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
            log.info("[BA] Optimizing with g2o...")

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
    