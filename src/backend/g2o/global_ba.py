import g2o
from src.backend.g2o.ba import BA, X_inv, L_inv
from src.others.linalg import invert_transform
from src.local_mapping.local_map import Map
from config import log


class globalBA(BA):
    def __init__(self, map: Map, verbose=False):
        """Performs global Bungle Adjustment fixing only the very first pose"""
        super().__init__(map)
        log.info("[BA] Performing full BA...")
        self.verbose = verbose

        # The keyframes to optimize
        self._add_frames()
        self._add_observations()

    def _add_frames(self):
        """
        Add a pose (4x4 transformation matrix) as a VertexSE3Expmap.
        The first pose is fixed to anchor the graph.
        """
        if self.verbose:
            log.info(f"\t Adding {self.map.num_keyframes()} poses...")
        
        frames = list(self.map.keyframes.values())
        self._add_frame(frames[0], fixed=True)
        for frame in frames[1:]:
            self._add_frame(frame, fixed=False)

    def _add_observations(self):
        """Add landmarks as vertices and reprojection observations as edges."""
        if self.verbose:
            log.info(f"\t Adding {self.map.num_points()} landmarks...")

        # Iterate over all map points
        for pt in self.map.points.values():
            self._add_observation(pt, fixed=False)

    def optimize(self, num_iterations=20):
        """
        Optimize the graph and return optimized poses and landmark positions.
        
        Returns:
            A tuple (pose_ids, poses, landmark_ids, landmarks, success)
        """
        if self.verbose:
            log.info("\t Optimizing with g2o...")

        self.optimizer.initialize_optimization()
        self.optimizer.optimize(num_iterations)

        # Optimize poses and landmarks
        self.map.get_mean_projection_error()
        self.update_poses_and_landmarks()
        self.map.get_mean_projection_error()

        return True

    def finalize(self):
        """Returns the final poses (optimized)."""
        self.update_poses_and_landmarks()

    def update_poses_and_landmarks(self):
        """Retrieves optimized pose and landmark estimates from the optimizer."""
        log.info("[BA] Updating poses and landmark positions...")
        # Iterate over all vertices.
        for vertex in self.optimizer.vertices().values():
            if isinstance(vertex, g2o.VertexSE3Expmap):
                pose = invert_transform(vertex.estimate().matrix())
                frame_id = X_inv(vertex.id())
                self.map.keyframes[frame_id].optimize_pose(pose)
            elif isinstance(vertex, g2o.VertexPointXYZ):
                pid = L_inv(vertex.id())
                self.map.points[pid].pos = vertex.estimate()
