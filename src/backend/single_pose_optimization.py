from copy import deepcopy
from config import SETTINGS

import src.local_mapping as mapping
import src.utils as utils
import src.backend as backend
import src.globals as ctx

from .ba import BA

from config import K, log

# Set parameters from the config
DEBUG = SETTINGS["generic"]["debug"]


class singlePoseBA(BA):
    def __init__(self, frame: utils.Frame):
        """Initializes Single Pose Optimization with a g2o optimizer and camera intrinsics."""
        super().__init__()
        if DEBUG:
            log.info(f"[singlePoseBA] Performing Pose #{frame.id} Optimization...")

        # The keyframes to optimize
        self.frame = frame
        self._add_frame(self.frame, fixed=False)
        self._add_observations()

    def _add_observations(self):
        """Add landmarks as vertices and reprojection observations as edges."""
        if DEBUG:
            log.info(f"\t Adding {ctx.map.num_points} landmarks...")

        # Extract the frame <-> map feature matches
        feat_mp_matches = self.frame.get_map_matches()

        # Iterate over all matches
        for feat, point in feat_mp_matches:
            # Add the landmark observation
            self._add_landmark(point.id, deepcopy(point.pos), fixed=True)

            # Extract the frame feature
            kpt = feat.kpt
            pt = deepcopy(kpt.pt)
            octave = kpt.octave
            # Add the observation
            self._add_observation(point.id, self.frame, pt, octave, level=0)

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
        """Optimizes pose estimates from the optimizer."""
        if DEBUG:
            log.info(f"\t Updating Pose #{self.frame.id}...")

        frame_id = self.frame.id
        vertex = self.optimizer.vertex(backend.X(frame_id))
        new_pose = utils.invert_transform(vertex.estimate().matrix()).copy()
        ctx.map.optimize_pose(frame_id, new_pose)
