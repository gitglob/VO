from copy import deepcopy
from config import SETTINGS
import g2o
import src.local_mapping as mapping
import src.utils as utils
import src.backend as backend
import src.globals as ctx
from .ba import BA

from config import K, log

# Set parameters from the config
DEBUG = SETTINGS["generic"]["debug"]


class poseBA(BA):
    def __init__(self):
        """Initializes Single Pose Optimization with a g2o optimizer and camera intrinsics."""
        super().__init__()
        if DEBUG:
            log.info(f"[poseBA] Performing Pose Optimization...")

        # The keyframes to optimize
        self._add_frames()
        self._add_observations()

    def _add_frames(self):
        """
        Add a pose (4x4 transformation matrix) as a VertexSE3Expmap.
        The first pose is fixed to anchor the graph.
        """
        if DEBUG:
            log.info(f"\t Adding {ctx.map.num_keyframes} poses...")
        
        frames = list(ctx.map.keyframes.values())
        self._add_frame(frames[0], fixed=True)
        for frame in frames[1:]:
            self._add_frame(frame, fixed=False)

    def _add_observations(self):
        """Add landmarks as vertices and reprojection observations as edges."""
        if DEBUG:
            log.info(f"\t Adding {ctx.map.num_points} landmarks...")

        # Iterate over all map points
        for pid, mp in ctx.map.points.items():
            self._add_landmark(mp.id, mp.pos, fixed=True)
            # Iterate over all the point observations
            for obs in mp.observations:
                kf_id = obs.kf_id  # id of keyframe that observed the landmark
                kf = ctx.map.keyframes[kf_id]
                kpt = obs.kpt # keypoint of the observation
                self._add_observation(pid, kf, kpt.pt, kpt.octave)

    def optimize(self):
        """
        Optimize the graph and return optimized poses and landmark positions.
        
        Returns:
            A tuple (pose_ids, poses, landmark_ids, landmarks, success)
        """
        if DEBUG:
            log.info("\t Optimizing...")

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
        e1 = ctx.map.get_mean_projection_error()
        self.update_poses()
        e2 = ctx.map.get_mean_projection_error()
        if DEBUG:
            log.info(f"\t RMS Re-Projection Error: {e1:.2f} -> {e2:.2f}")

        return n_inlier_edges

    def finalize(self):
        """Returns the final poses (optimized)."""
        self.update_poses()

    def update_poses(self):
        """Retrieves optimized pose and landmark estimates from the optimizer."""
        log.info(f"\t Updating poses...")
        # Iterate over all vertices.
        for vertex in self.optimizer.vertices().values():
            if isinstance(vertex, g2o.VertexSE3Expmap):
                new_pose = utils.invert_transform(vertex.estimate().matrix()).copy()
                frame_id = backend.X_inv(vertex.id())
                ctx.map.optimize_pose(frame_id, new_pose)
