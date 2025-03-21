from typing import List 
import numpy as np
from src.utils import invert_transform
from config import image_width, image_height, debug, SETTINGS


class Point():
    def __init__(self, pos: np.ndarray, id: int, desc: np.ndarray):
        self.pos: np.ndarray = pos 
        self.id: int = id
        self.desc: np.ndarray = desc
        self.match_counter: int = 0
        self.view_counter: int = 0

class Map():
    def __init__(self, frame_id: int):
        self.origin_frame = frame_id  # ID of the frame when the map was first created
        self.points: np.ndarray = np.empty((0,), dtype=object)

        # Mask that indicates which of the current points are in the camera view
        self._in_view_mask = None

        # The pixel coordinates of the points in the current camera view
        self._u = None
        self._v = None

    @property
    def num_points(self):
        return len(self.points)

    @property
    def point_positions(self):
        """Returns the points xyz positions"""
        return np.vstack([p.pos for p in self.points])
    @property
    def point_ids(self):
        """Returns the point ids"""
        return np.vstack([p.id for p in self.points])

    @property
    def point_descriptors(self):
        """Returns the associated descriptors from the keyframe that created them"""
        return np.vstack([p.desc for p in self.points])
    
    @property
    def num_points_in_view(self):
        """Returns the number of points that are currently in view"""
        if self._in_view_mask is None:
            return 0
        return self._in_view_mask.sum()
    
    @property
    def points_in_view(self):
        return self.points[self._in_view_mask]
    
    @property
    def point_positions_in_view(self):
        return self.point_positions[self._in_view_mask]

    @property
    def descriptors_in_view(self):
        return self.point_descriptors[self._in_view_mask]

    @property
    def pixels_in_view(self):
        u_in_view = self._u[self._in_view_mask]
        v_in_view = self._v[self._in_view_mask]
        pxs_c_n_view = np.column_stack([u_in_view, v_in_view])

        return pxs_c_n_view

    def add_points(self, points: np.ndarray, point_ids: np.ndarray, descriptors: np.ndarray):
        # Get initial number of points
        prev_num_points = self.num_points
        # Create an array of empty cells
        empty_cells = np.full(len(points), None, dtype=object)
        # Concatenate the new empty cells to the existing points array
        self.points = np.concatenate((self.points, empty_cells))

        # Iterate over the new points
        for i in range(len(points)):
            p = Point(points[i], point_ids[i], descriptors[i])
            self.points[prev_num_points + i] = p
        print(f"[Map] Adding {len(points)} points. Total: {len(self.points)} points.")

    def view(self, T_wc: np.ndarray, K: np.ndarray, pred=False):
        """
        Returns the points and descriptors that are in the current view.

        Args:
            T_wc: The transform from the world to the current camera coordinate frame
            K: The camera intrinsics matrix
            pref: Whether the view was for a real or predicted pose
        """
        print("[Map] Getting points in the current view...")

        # No map points at all
        if self.num_points == 0:
            return

        # 0) Collect ALL points and descriptors from the map
        point_positions = self.point_positions     # (M, 3)
        point_descriptors = self.point_descriptors # (M, 32)
        assert(len(point_positions) == len(point_descriptors))
        
        # 1) Convert map points to homogeneous: (X, Y, Z, 1).
        ones = np.ones((len(point_positions), 1))
        point_positions_hom = np.hstack([point_positions, ones])  # (M, 4)

        # 2) Transform points to camera coords:
        points_c_hom = (T_wc @ point_positions_hom.T).T # (M, 4)
        points_c = points_c_hom[:, :3]           # (M, 3)

        # 3) Keep only points in front of the camera (z > 0).
        z_positive_mask = points_c[:, 2] > 0

        # 4) Project into pixel coordinates using K.
        x_cam = points_c[:, 0]
        y_cam = points_c[:, 1]
        z_cam = points_c[:, 2]

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        self._u = fx * x_cam / z_cam + cx
        self._v = fy * y_cam / z_cam + cy

        # 5) Check if the projected points lie within image boundaries.
        boundary_mask = (
            (self._u >= 0) & (self._u < image_width) &
            (self._v >= 0) & (self._v < image_height)
        )

        # 6) Combine the z_positive and in_view mask
        self._in_view_mask = z_positive_mask & boundary_mask

        # 7) Increase the view counter for every visible point
        if not pred:
            for p in self.points[self._in_view_mask]:
                p.view_counter += 1

        if debug:
            print(f"\tFound {self._in_view_mask.sum()} map points in the predicted camera pose view.")
    
    def cleanup(self, T_wc: np.ndarray, K: np.ndarray):
        """
        Args:
            T_wc: Transformation from world to camera coordinate frame
            K: Intrinsics matrix

        Remove map points that are 
            (1) not in current view
            (2) with a view angle larger than the threshold
            (3) rarely matched as inlier point
        """
        print("[Map] Cleaning up map points")

        T_cw = invert_transform(T_wc)
        prev_num_points = self.num_points
        self.view(T_wc, K)

        # Remove out of view points
        self.points = self.points[self._in_view_mask]
        print(f"[Map] View check removed {np.count_nonzero(~self._in_view_mask)} points!")

        # Remove points that are rarely matched
        match_view_ratio_mask = np.ones(self.num_points, dtype=bool)
        for i, p in enumerate(self.points):
            match_view_ratio = p.match_counter / p.view_counter

            if p.view_counter > 5 and match_view_ratio < SETTINGS["map"]["match_view_ratio"]:
                match_view_ratio_mask[i] = False

        self.points = self.points[match_view_ratio_mask]
        print(f"[Map] Match-View ratio check removed {np.count_nonzero(~match_view_ratio_mask)} points!")

        # Remove points that have a very large view angle
        view_angle_mask = np.ones(self.num_points, dtype=bool)

        ## Extract the normalized camera center vector
        cam_center_vec = T_cw[:3, 3]
        cam_center_vec_norm = cam_center_vec / np.linalg.norm(cam_center_vec)
        for i, p in enumerate(self.points):
            # Extract the normalized point vectors
            pos_vec = p.pos
            pos_vec_norm = pos_vec / np.linalg.norm(pos_vec)

            # Extract the angle between the 2 vectors
            dot_product = np.clip(np.dot(cam_center_vec_norm, pos_vec_norm), -1.0, 1.0)
            view_angle = np.degrees(np.arccos(dot_product))

            # Check if the view angle is larger than the threshold
            if view_angle > SETTINGS["map"]["view_angle_threshold"]:
                view_angle_mask[i] = False

        self.points = self.points[view_angle_mask]
        print(f"[Map] View-Angle check removed {np.count_nonzero(~view_angle_mask)} points!")

        print(f"[Map] Removed {prev_num_points - self.num_points}/{prev_num_points} points from the map!")

        # Reset the in-view mask
        self._in_view_mask = None
