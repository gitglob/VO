import g2o
import numpy as np
import src.local_mapping as mapping
import src.utils as utils
import src.backend as backend
import src.globals as ctx
from .ba import BA
from config import SETTINGS, log, K


DEBUG = SETTINGS["generic"]["debug"]


class localBA(BA):
    def __init__(self, frame: utils.Frame):
        """
        Performs Local Bundle Adjustment.
        Local BA means that ...         
        """
        super().__init__()
        self.keyframe = frame

        self._build()

    def _build(self):
        """
        Add a pose (4x4 transformation matrix) as a VertexSE3Expmap.
        The first pose is fixed to anchor the graph.
        """
        # Get the connected keyframe and point ids from the convisibility graph
        connected_kf_ids, connected_point_ids = ctx.cgraph.get_connected_frames_and_their_points(self.keyframe.id)

        # Add the current frame to the connected ones
        connected_kf_ids1 = connected_kf_ids.copy()
        connected_kf_ids1.add(self.keyframe.id)
        connected_kfs = [ctx.map.keyframes[idx] for idx in connected_kf_ids1]
        first_kf_id = np.min([kf.id for kf in connected_kfs])

        # Get all the other keyframes that see the points but are not connected to the current keyframe
        kfs_that_see_points_ids = ctx.cgraph.get_frames_that_observe_points(connected_point_ids)
        unconnected_kfs_ids = kfs_that_see_points_ids - connected_kf_ids1
        unconnected_kfs = [ctx.map.keyframes[idx] for idx in unconnected_kfs_ids]

        # Get all the used keyframe ids
        all_kf_ids = connected_kf_ids1.copy()
        all_kf_ids.update(unconnected_kfs_ids)

        if DEBUG:
            msg = f"[localBA] Adding frame {self.keyframe.id}, {len(connected_kf_ids)} "
            msg += f"connected frames with {len(connected_point_ids)} points, "
            msg += f"and {len(unconnected_kfs)} unconnected frames..."
            log.info(msg)

        # Add the main frame and the connected ones
        for kf in connected_kfs:
            is_first_kf = kf.id == first_kf_id
            self._add_frame(kf, fixed=is_first_kf)

        # Add the and fix the un-connected frames
        for kf in unconnected_kfs:
            self._add_frame(kf, fixed=True)

        # Iterate over all the points that the connected frames see
        for pid in connected_point_ids:
            point: mapping.mapPoint = ctx.map.points[pid]
            # Add the landmark vertex
            self._add_landmark(point.id, point.pos)
            # Iterate over all the landmark observations
            for obs in point.observations:
                kf = ctx.map.keyframes[obs.kf_id]
                # Skip observations from
                assert kf.id in all_kf_ids
                kpt = obs.kpt
                # Add the pose->landmark observations
                assert point.id in ctx.cgraph.nodes[kf.id]
                self._add_observation(point.id, kf, kpt.pt, kpt.octave)

    def optimize(self):
        """Optimize the poses and landmark positions."""
        e1 = ctx.map.get_mean_projection_error()

        # Optimize with the outliers
        num_edges = len(self.optimizer.edges())
        if DEBUG: log.info(f"\t Optimizing {num_edges} edges...")
        self.optimizer.initialize_optimization()
        self.optimizer.optimize(5)
        removed_edges = self.prune_bad_edges()

        # Remove feature<->map point match and map point observation
        ctx.map.remove_matches(removed_edges)
        ctx.cgraph.remove_matches(removed_edges)

        # Update the poses and landmarks
        self.update_poses_and_landmarks()

        if DEBUG: log.info(f"\t Removed {num_edges - len(self.optimizer.edges())} edges...")
        num_edges = len(self.optimizer.edges())

        e2 = ctx.map.get_mean_projection_error()

        # Optimize again without the outliers
        if DEBUG: log.info(f"\t Optimizing {num_edges} edges...")
        self.optimizer.initialize_optimization()
        self.optimizer.optimize(10)
        removed_edges = self.prune_bad_edges()

        # Remove feature<->map point match and map point observation
        ctx.map.remove_matches(removed_edges)
        ctx.cgraph.remove_matches(removed_edges)

        # Update the poses and landmarks
        self.update_poses_and_landmarks()

        if DEBUG: log.info(f"\t Removed {num_edges - len(self.optimizer.edges())} edges...")

        e3 = ctx.map.get_mean_projection_error()

        if DEBUG:
            log.info(f"\t RMS Re-Projection Error: {e1:.2f} -> {e2:.2f} -> {e3:.2f}")

    def prune_bad_edges(self, chi2_threshold=9.21) -> set[tuple[int, int]]:
        """
        Remove any edges whose chi2 is above threshold (95% CI) or whose depth≤0.
        Returns a set of (point_id, keyframe_id) tuples that were removed.
        """
        # 1) Snapshot all edges up-front
        all_edges = list(self.optimizer.edges())
        removed = set()

        # 2) Identify the ones to drop
        for e in all_edges:
            kf_id, pid, depth = self.edge_info(e)
            if e.chi2() > chi2_threshold or depth <= 0:
                removed.add((pid, kf_id))
        
        # 3) Now do the removals in a separate loop
        for e in all_edges:
            kf_id, pid, depth = self.edge_info(e)
            if (pid, kf_id) in removed:
                self.optimizer.remove_edge(e)

        return removed

    def edge_info(self, edge: g2o.EdgeProjectXYZ2UV) -> tuple[int, int, float]:
        """Using an g2o edge extracts the landmark's Z position (depth) in the camera's frame"""
        vertices = edge.vertices()

        # Extract landmark id and world position
        landmark_vertex = vertices[0]
        assert isinstance(landmark_vertex, g2o.VertexPointXYZ)
        pid = backend.L_inv(landmark_vertex.id())
        landmark_pos = landmark_vertex.estimate()

        # Extract camera pose and frame id
        cam_vertex = vertices[1]
        assert isinstance(cam_vertex, g2o.VertexSE3Expmap)
        frame_id = backend.X_inv(cam_vertex.id())
        T_w2c = cam_vertex.estimate().matrix()
        R = T_w2c[:3, :3]
        t = T_w2c[:3, 3]

        # Convert the landmark world coordinate to camera coordinates
        landmark_pos_c = R @ landmark_pos + t
        return frame_id, pid, landmark_pos_c[2]

    def update_poses_and_landmarks(self):
        """Retrieves optimized pose and landmark estimates from the optimizer."""
        if DEBUG:
            log.info("\t Updating poses and landmark positions...")
        # Iterate over all vertices.
        for vertex in self.optimizer.vertices().values():
            if isinstance(vertex, g2o.VertexSE3Expmap):
                new_pose = utils.invert_transform(vertex.estimate().matrix()).copy()
                frame_id = backend.X_inv(vertex.id())
                ctx.map.optimize_pose(frame_id, new_pose)
            elif isinstance(vertex, g2o.VertexPointXYZ):
                pid = backend.L_inv(vertex.id())
                new_pos = vertex.estimate().copy()
                # Check if the point is still in the map
                # (it may have been removed after removing edges because it was
                # left with no observations)
                if pid in ctx.map.points:
                    ctx.map.optimize_point(pid, new_pos)
    