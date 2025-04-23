from copy import deepcopy
import g2o
import numpy as np
from config import SETTINGS
from src.local_mapping.map import Map
from src.others.frame import Frame
from src.backend.g2o.ba import BA, X, L_inv
from src.others.linalg import invert_transform
from config import K, log

# Set parameters from the config
MEASUREMENT_SIGMA = float(SETTINGS["ba"]["measurement_noise"])
NUM_OBSERVATIONS = int(SETTINGS["ba"]["num_observations"])


class singlePoseBA(BA):
    def __init__(self, map: Map, frame: Frame, verbose=False):
        """Initializes Single Pose Optimization with a g2o optimizer and camera intrinsics."""
        super().__init__()
        log.info(f"[BA] Performing Pose #{frame.id} Optimization...")
        self.verbose = verbose

        # The keyframes to optimize
        self.map: Map = map
        self.frame = frame
        self._add_frame(self.frame, fixed=False)
        self._add_observations()

    def _add_observations(self):
        """Add landmarks as vertices and reprojection observations as edges."""
        if self.verbose:
            log.info(f"\t Adding {self.map.num_points()} landmarks...")

        # Extract the frame <-> map feature matches
        feat_mp_matches = self.frame.get_map_matches()

        # Iterate over all matches
        for feat_id, pid in feat_mp_matches:
            # Extract the map point position
            mp = self.map.points[pid]
            # Add the landmark observation
            self._add_landmark(mp.id, deepcopy(mp.pos), fixed=True)

            # Extract the frame feature
            kpt = self.frame.features[feat_id].kpt
            pt = deepcopy(kpt.pt)
            octave = kpt.octave
            # Add the observation
            self._add_observation(pid, self.frame, pt, octave, level=0)

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
        self.map.get_mean_projection_error()
        self.update_poses()
        self.map.get_mean_projection_error()

        return n_inlier_edges

    def finalize(self):
        """Returns the final poses (optimized)."""
        self.update_poses()

    def update_poses(self):
        """Optimizes pose estimates from the optimizer."""
        log.info(f"[BA] Optimizing Pose #{self.frame.id}...")

        frame_id = self.frame.id
        vertex = self.optimizer.vertex(X(frame_id))
        new_pose = invert_transform(vertex.estimate().matrix()).copy()
        self.map.keyframes[frame_id].optimize_pose(new_pose)
