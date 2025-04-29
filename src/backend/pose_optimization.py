import numpy as np
import g2o
import src.utils as utils
import src.globals as ctx
from config import K, log, SETTINGS

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


class poseBA:
    def __init__(self, verbose=False):
        """Initializes BA_g2o with a g2o optimizer and camera intrinsics."""
        log.info("[BA] Performing Pose Optimization...")
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
        self._add_frames()
        self._add_observations()

    def _add_frames(self):
        """
        Add a pose (4x4 transformation matrix) as a VertexSE3Expmap.
        The first pose is fixed to anchor the graph.
        """
        if self.verbose:
            log.info(f"\t Adding {ctx.map.num_keyframes} poses...")
        
        frames = list(ctx.map.keyframes.values())
        for frame in frames:
            self._add_frame(frame)

    def _add_frame(self, frame: utils.Frame, fixed=False):
        """
        Add a pose (4x4 transformation matrix) as a VertexSE3Expmap.
        The first pose is fixed to anchor the graph.
        """
        p_id = frame.id
        p = frame.pose if frame.pose is not None else frame.noopt_pose

        # Convert the 4x4 pose matrix into an SE3Quat.
        R = p[:3, :3]
        t = p[:3, 3]
        se3 = g2o.SE3Quat(R, t)
        vertex = g2o.VertexSE3Expmap()
        vertex.set_id(X(p_id))
        vertex.set_estimate(se3)
        vertex.set_fixed(fixed)
            
        # Add the vertex to the graph
        self.optimizer.add_vertex(vertex)

    def _add_observations(self):
        """Add landmarks as vertices and reprojection observations as edges."""
        if self.verbose:
            log.info(f"\t Adding {ctx.map.num_points} landmarks...")

        # This kernel value is chosen based on the chi–squared distribution with 2 degrees of freedom 
        # (since the measurement is 2D) so that errors above this threshold are down–weighted.
        delta = np.sqrt(5.991)

        # Iterate over all map points
        for pt in ctx.map.points_arr:
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
                pose_idx = obs["keyframe"].id # id of keyframe that observed the landmark
                kpt = obs.kpt         # keypoint of the observation
                u, v = kpt.pt                 # pixels of the keypoint

                # Create the reprojection edge.
                edge = g2o.EdgeProjectXYZ2UV()
                # edge = g2o.EdgeSE3ProjectXYZ()
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
                edge.set_level(0)

                # Add observation edge
                self.optimizer.add_edge(edge)

    def optimize(self):
        """
        Optimize the graph and return optimized poses and landmark positions.
        
        Returns:
            A tuple (pose_ids, poses, landmark_ids, landmarks, success)
        """
        if self.verbose:
            log.info("\t Optimizing with g2o...")

        # Calculate initial number of edges
        n_edges = len(self.optimizer.edges())

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
                    e.set_level(1)
                    n_outlier_edges += 1
                else:
                    e.set_level(0)

            # Check if too little edges are left
            if len(self.optimizer.edges()) < 10:
                break

        # Calculate the number of inliers
        n_inlier_edges = n_edges - n_outlier_edges

        # Optimize poses
        self.update_poses()

        return n_inlier_edges

    def finalize(self):
        """
        Returns the final poses (optimized).
        """
        self.update_poses()

    def update_poses(self):
        """
        Retrieves optimized pose and landmark estimates from the optimizer.
        
        Returns:
            (pose_ids, pose_array, landmark_ids, landmark_array)
        """
        log.info(f"\t Updating poses...")
        # Iterate over all vertices.
        for vertex in self.optimizer.vertices().values():
            # Find pose verticies
            if isinstance(vertex, g2o.VertexSE3Expmap):
                # Update poses
                pose = vertex.estimate().matrix()
                frame_id = X_inv(vertex.id())
                ctx.map.keyframes[frame_id].set_pose(pose)
