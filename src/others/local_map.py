from typing import List
import numpy as np
import cv2
from config import debug, SETTINGS


scale_factor = SETTINGS["orb"]["scale_factor"]
n_levels = SETTINGS["orb"]["level_pyramid"]
min_observations = SETTINGS["orb"]["level_pyramid"]
W = SETTINGS["image"]["width"]
H = SETTINGS["image"]["height"]

class mapPoint():
    def __init__(self, kf_id: int, 
                 cam_center: np.ndarray, 
                 pos: np.ndarray, 
                 keypoint: int, 
                 desc: np.ndarray):
        self.kf_id: int = kf_id           # The observation id of the keyframe inside the local map
        self.cam_center = cam_center
        self.pos: np.ndarray = pos        # 3D position
        self.kpt: cv2.KeyPoint = keypoint # ORB keypoint
        self.desc: np.ndarray = desc      # ORB descriptor

        self.init()

    def init(self):
        self.id: int = self.kpt.class_id    # The id of the keypoint
        self.match_counter: int = 0         # Number of times the point was tracked with PnP
        self.obs_counter: int = 0           # Number of times the point was observed in a Frame

    def view_ray(self, cam_center_vec):
        v = self.pos - cam_center_vec
        v = v / np.linalg.norm(v)
        return v
    
    def mean_view_ray(self): # TODO: integrate multiple points
        return self.view_ray(self.cam_center)

    def getScaleInvarianceLimits(self):
        dist = np.linalg.norm(self.pos - self.cam_center)
        level = self.kpt.octave
        minLevelScaleFactor = scale_factor**level
        maxLlevelScaleFactor = scale_factor**(n_levels - 1 - level)

        dmin = (1 / scale_factor) * dist / minLevelScaleFactor
        dmax = scale_factor * dist * maxLlevelScaleFactor

        return (dmin, dmax)

class Map():
    def __init__(self, frame_id: int):
        self.origin_frame = frame_id  # ID of the frame when the map was first created
        self.points: np.ndarray = np.empty((0,), dtype=object)

        # Mask that indicates which of the current points are in the camera view
        self._in_view_mask = None

        # The pixel coordinates of the points in the current camera view
        self._u = None
        self._v = None

        # Read settings
        self._match_view_ratio = SETTINGS["map"]["match_view_ratio"]
        self._max_size = SETTINGS["map"]["max_size"]

        # Frame counter
        self._kf_counter = 0

    @property
    def num_points(self):
        return len(self.points)

    @property
    def point_positions(self):
        """Returns the points xyz positions"""
        return np.vstack([p.pos for p in self.points])

    @property
    def point_ids(self):
        """Returns the points xyz positions"""
        return np.vstack([p.id for p in self.points])
    
    @property
    def num_points_in_view(self):
        """Returns the number of points that are currently in view"""
        if self._in_view_mask is None:
            return 0
        return self._in_view_mask.sum()
    
    @property
    def points_in_view(self):
        return self.points[self._in_view_mask]
    
    def add_points(self, points: np.ndarray, 
                   keypoints: List[cv2.KeyPoint], 
                   descriptors: np.ndarray,
                   T_cw: np.ndarray):
        # Extract the camera center vector
        cam_center = T_cw[:3, 3]
        # Get initial number of points
        prev_num_points = self.num_points
        # Create an array of empty cells
        empty_cells = np.full(len(points), None, dtype=object)
        # Concatenate the new empty cells to the existing points array
        self.points = np.concatenate((self.points, empty_cells))

        # Iterate over the new points
        for i in range(len(points)):
            p = mapPoint(self._kf_counter, cam_center, points[i], keypoints[i], descriptors[i])
            self.points[prev_num_points + i] = p
        self._kf_counter += 1
        print(f"Adding {len(points)} points to the Map. Total: {len(self.points)} points.")

    def view(self, T_wc: np.ndarray, K: np.ndarray, pred=False):
        """
        Returns the points and descriptors that are in the current view.

        Args:
            T_wc: The transform from the world to the current camera coordinate frame
            K: The camera intrinsics matrix
            pref: Whether the view was for a real or predicted pose
        """
        print("Getting points in the current view...")

        # No map points at all
        if self.num_points == 0:
            return

        # 0) Collect ALL points and descriptors from the map
        point_positions = self.point_positions     # (M, 3)
        assert(len(point_positions) == self.num_points)
        
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
            (self._u >= 0) & (self._u < W) &
            (self._v >= 0) & (self._v < H)
        )

        # 6) Combine the z_positive and in_view mask
        self._in_view_mask = z_positive_mask & boundary_mask

        # 7) Increase the view counter for every visible point
        if pred:
            for p in self.points[self._in_view_mask]:
                p.obs_counter += 1

        if debug:
            print(f"\tFound {self._in_view_mask.sum()} map points in the predicted camera pose view.")
    
    def cull(self):
        """
        Args:
            T_wc: Transformation from world to camera coordinate frame
            K: Intrinsics matrix

        Remove map points that are 
            (1) not in current view
            (2) with a view angle larger than the threshold
            (3) rarely matched as inlier point
        """
        print("Cleaning up map points...")

        prev_num_points = self.num_points

        # 1) Remove points that are rarely matched
        match_view_ratio_mask = np.ones(self.num_points, dtype=bool)
        for i, p in enumerate(self.points):
            if p.obs_counter > 4:
                match_view_ratio = p.match_counter / p.obs_counter
                if match_view_ratio < self._match_view_ratio:
                    match_view_ratio_mask[i] = False

        self.points = self.points[match_view_ratio_mask]
        print(f"\t Match-View ratio check removed {np.count_nonzero(~match_view_ratio_mask)} points!")

        # 2) Remove points that are rarely seen
        num_views_mask = np.ones(self.num_points, dtype=bool)
        for i, p in enumerate(self.points):
            num_kf_passed = self._kf_counter
            point_creation_kf = p.kf_id
            num_kf_passed_since_point_creation = num_kf_passed - point_creation_kf

            if num_kf_passed_since_point_creation > 3 and p.obs_counter > min_observations:
                num_views_mask[i] = False

        self.points = self.points[num_views_mask]
        print(f"\t Observability check removed {np.count_nonzero(~num_views_mask)} points!")

        print(f"\t Removed {prev_num_points - self.num_points}/{prev_num_points} points from the map!")

        # Reset the in-view mask
        self._in_view_mask = None
