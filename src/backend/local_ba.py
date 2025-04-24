import g2o
import numpy as np
from src.backend.ba import BA, X_inv, L_inv
from src.backend.convisibility_graph import ConvisibilityGraph
from src.local_mapping.map import Map, mapPoint
from src.others.frame import Frame
from src.others.linalg import invert_transform
from config import SETTINGS, log, K


class localBA(BA):
    def __init__(self, frame: Frame, map: Map, cgraph: ConvisibilityGraph, verbose=False):
        """
        Performs Local Bundle Adjustment.
        Local BA means that ...         
        """
        super().__init__()
        self.verbose = verbose

        self.keyframe = frame
        self.map = map
        self.cgraph = cgraph

        self._build()

    def _build(self):
        """
        Add a pose (4x4 transformation matrix) as a VertexSE3Expmap.
        The first pose is fixed to anchor the graph.
        """
        # Get the connected keyframe and point ids from the convisibility graph
        connected_kf_ids, connected_point_ids = self.cgraph.get_connected_frames_and_their_points(self.keyframe.id)
        connected_kfs = [self.map.keyframes[idx] for idx in connected_kf_ids]

        # Add the current frame to the connected ones
        all_kfs = [kf for kf in connected_kfs]
        all_kfs.append(self.keyframe)
        first_kf_id = np.min([kf.id for kf in all_kfs])

        # Get the connected points
        connected_points: list[mapPoint] = self.map.get_points(connected_point_ids)

        # Get all the other keyframes that see the points but are not connected to the current keyframe
        kfs_that_see_points_ids = self.map.get_keyframes_that_see(connected_point_ids)
        unconnected_kfs_that_see_points_ids = kfs_that_see_points_ids - connected_kf_ids
        unconnected_kfs_that_see_points = [self.map.keyframes[idx] for idx in unconnected_kfs_that_see_points_ids]

        if self.verbose:
            msg = f"[BA] Adding frame {self.keyframe.id}, {len(connected_kf_ids)} "
            msg += f"connected frames with {len(connected_points)} points, "
            msg += f"and {len(unconnected_kfs_that_see_points)} unconnected poses..."
            log.info(msg)

        # Add the frames
        for kf in all_kfs:
            is_first_kf = kf.id == first_kf_id
            self._add_frame(kf, fixed=is_first_kf)

        # Iterate over all the connected points
        for point in connected_points:
            pid = point.id
            # Add the connected landmark vertices
            self._add_landmark(point.id, point.pos)
            # Iterate over all the landmark observations
            for obs in point.observations:
                kf_id = obs["kf_id"]
                kf = self.map.keyframes[kf_id]
                kpt = obs["keypoint"]
                pt = kpt.pt
                octave = kpt.octave
                # If the observation was done by a connected keyframe...
                if kf.id in connected_kf_ids:
                    # ... add the connected pose->landmark observations
                    self._add_observation(pid, kf, pt, octave)

        # Add the and fix the un-connected poses
        for kf in unconnected_kfs_that_see_points:
            self._add_frame(kf, fixed=True)

    def optimize(self):
        """Optimize the poses and landmark positions."""
        self.map.get_mean_projection_error()

        # Optimize again with the outliers
        num_edges = len(self.optimizer.edges())
        log.info(f"\t Optimizing {num_edges} edges...")

        self.optimizer.initialize_optimization()
        self.optimizer.optimize(5)

        # 95% CI
        chi2_threshold = 9.21

        # Iterate over all the edges
        removed_edges = set()
        for e in self.optimizer.edges():
            # Extract edge info
            kf_id, mp_id, depth = self.edge_info(e)
            # Check the chi2 value
            if e.chi2() > chi2_threshold or depth <= 0:
                removed_edges.add((mp_id, kf_id))
                # Remove edge from the graph
                self.optimizer.remove_edge(e)

        # Remove feature<->map point match and map point observation
        for (mp_id, kf_id) in removed_edges:
            self.map.keyframes[kf_id].remove_mp_match(mp_id)
            self.map.points[mp_id].remove_observation(kf_id)
            self.cgraph.remove_point(kf_id, mp_id)

        log.info(f"\t Removed {num_edges - len(self.optimizer.edges())} edges...")
        num_edges = len(self.optimizer.edges())

        self.update_poses_and_landmarks()

        e = self.map.get_mean_projection_error()
        log.info(f"\t RMS Re-Projection Error: {e:.2f}")

        # Optimize again without the outliers
        log.info(f"\t Optimizing {num_edges} edges...")
        self.optimizer.initialize_optimization()
        self.optimizer.optimize(10) # TODO: this crashes

        # Iterate over all the edges
        removed_edges = set()
        for e in self.optimizer.edges():
            # Extract edge info
            kf_id, mp_id, depth = self.edge_info(e)
            if e.chi2() > chi2_threshold or depth <= 0:
                removed_edges.add((mp_id, kf_id))

        # Remove feature<->map point match and map point observation
        for (mp_id, kf_id) in removed_edges:    
            self.map.keyframes[kf_id].remove_mp_match(mp_id)
            self.map.points[mp_id].remove_observation(kf_id)
            self.cgraph.remove_point(kf_id, mp_id)

        log.info(f"\t Removed {num_edges - len(self.optimizer.edges())} edges...")
        self.cgraph._update_edges_on_point_culling()

        self.update_poses_and_landmarks()

        e = self.map.get_mean_projection_error()
        log.info(f"\t RMS Re-Projection Error: {e:.2f}")

    def edge_info(self, edge: g2o.EdgeProjectXYZ2UV) -> tuple[int, int, float]:
        """Using an g2o edge extracts the landmark's Z position (depth) in the camera's frame"""
        vertices = edge.vertices()

        # Extract landmark id and world position
        landmark_vertex = vertices[0]
        assert isinstance(landmark_vertex, g2o.VertexPointXYZ)
        pid = L_inv(landmark_vertex.id())
        landmark_pos = landmark_vertex.estimate()

        # Extract camera pose and frame id
        cam_vertex = vertices[1]
        assert isinstance(cam_vertex, g2o.VertexSE3Expmap)
        frame_id = X_inv(cam_vertex.id())
        T_w2c = cam_vertex.estimate().matrix()
        R = T_w2c[:3, :3]
        t = T_w2c[:3, 3]

        # Convert the landmark world coordinate to camera coordinates
        landmark_pos_c = R @ landmark_pos + t
        return frame_id, pid, landmark_pos_c[2]

    def update_poses_and_landmarks(self):
        """Retrieves optimized pose and landmark estimates from the optimizer."""
        log.info("\t Updating poses and landmark positions...")
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
    