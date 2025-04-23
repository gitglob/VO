import g2o
import numpy as np
from src.backend.g2o.ba import BA, X_inv, L_inv
from src.others.linalg import invert_transform
from src.local_mapping.map import Map
from config import log


class globalBA(BA):
    def __init__(self, map: Map, verbose=False):
        """Performs global Bungle Adjustment fixing only the very first pose"""
        super().__init__()
        log.info("[BA] Performing full BA...")
        self.verbose = verbose
        self.map = map

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
        for pid, mp in self.map.points.items():
            self._add_landmark(mp.id, mp.pos, fixed=False)
            # Iterate over all the point observations
            for obs in mp.observations:
                kf_id = obs["kf_id"]  # id of keyframe that observed the landmark
                kf = self.map.keyframes[kf_id]
                kpt = obs["keypoint"] # keypoint of the observation
                self._add_observation(pid, kf, kpt.pt, kpt.octave)

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

        self.map.get_mean_projection_error()
        self.update_poses_and_landmarks()
        self.map.get_mean_projection_error()

    def finalize(self):
        """Returns the final poses (optimized)."""
        self.update_poses_and_landmarks()

    def update_poses_and_landmarks(self):
        """Retrieves optimized pose and landmark estimates from the optimizer."""
        log.info("[BA] Updating poses and landmark positions...")
        # Iterate over all vertices.
        for vertex in self.optimizer.vertices().values():
            if isinstance(vertex, g2o.VertexSE3Expmap):
                new_pose = invert_transform(vertex.estimate().matrix()).copy()
                frame_id = X_inv(vertex.id())
                self.map.keyframes[frame_id].optimize_pose(new_pose)
            elif isinstance(vertex, g2o.VertexPointXYZ):
                pid = L_inv(vertex.id())
                new_pos = vertex.estimate().copy()
                self.map.points[pid].pos = new_pos
