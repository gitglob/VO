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
    def __init__(self, 
                 kf_number: int,
                 kf_id: int, 
                 cam_pose: np.ndarray, 
                 pos: np.ndarray, 
                 keypoint: cv2.KeyPoint, 
                 desc: np.ndarray):
        self.observations = [
            { 
                "keyframe": kf_id,    # The observation id of the keyframe inside the local map
                "cam_pose": cam_pose, # Camera pose in that keyframe
                "keypoint": keypoint, # ORB keypoint
                "descriptor": desc    # ORB descriptor
            }
        ]

        self.kf_number: int = kf_number  # The keyframe number (not ID!) when it was created
        self.pos: np.ndarray = pos       # 3D position
        self.id: int = keypoint.class_id # The unique id of the keypoint

        self.match_counter: int = 0         # Number of times the point was tracked with PnP
        self.obs_counter: int = 0           # Number of times the point was observed in a Frame

    def observe(self,
               kf_id: int, 
               cam_pose: np.ndarray, 
               keypoint: cv2.KeyPoint, 
               desc: np.ndarray):
        
        new_observation = {
            "keyframe": kf_id,    
            "cam_pose": cam_pose, 
            "keypoint": keypoint, 
            "descriptor": desc
        }
        self.observations.append(new_observation)

    def view_ray(self, cam_pos):
        v = self.pos - cam_pos
        v = v / np.linalg.norm(v)
        return v
    
    def mean_view_ray(self):
        view_rays = []
        for obs in self.observations:
            v = self.view_ray(obs["cam_pose"][:3, 3])
            view_rays.append(v)

        return np.mean(view_rays, axis=0)

    def getScaleInvarianceLimits(self):
        cam_pos = self.observations[-1]["cam_pose"][:3, 3]
        level = self.observations[-1]["keypoint"].octave

        dist = np.linalg.norm(self.pos - cam_pos)
        minLevelScaleFactor = scale_factor**level
        maxLlevelScaleFactor = scale_factor**(n_levels - 1 - level)

        dmin = (1 / scale_factor) * dist / minLevelScaleFactor
        dmax = scale_factor * dist * maxLlevelScaleFactor

        return (dmin, dmax)

class Map():
    def __init__(self, frame_id: int):
        self.origin_frame = frame_id  # ID of the frame when the map was first created
        self.points: dict = {}   # Dictionary with id<->mapPoint pairs

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
    def points_arr(self):
        """Returns the mapPoints as an arrayu"""
        p_arr = []
        for k,v in self.points.items():
            p_arr.append(v)
        p_arr = np.array(p_arr, dtype=object)
        return p_arr

    @property
    def point_positions(self):
        """Returns the points xyz positions"""
        positions = []
        for k,v in self.points.items():
            positions.append(v.pos)
        positions = np.vstack(positions)
        return positions

    @property
    def point_ids(self):
        """Returns the points IDs"""
        ids = []
        for k,v in self.points.items():
            ids.append(k)
        ids = np.vstack(ids)
        return ids
    
    @property
    def num_points_in_view(self):
        """Returns the number of points that are currently in view"""
        if self._in_view_mask is None:
            return 0
        return self._in_view_mask.sum()
    
    @property
    def points_in_view(self):
        return self.points_arr[self._in_view_mask]
    
    def add_points(self, 
                   kf_id: int,
                   points_pos: np.ndarray, 
                   keypoints: List[cv2.KeyPoint], 
                   descriptors: np.ndarray,
                   T_cw: np.ndarray):
        # Iterate over the new points
        for i in range(len(points_pos)):
            kpt_id = keypoints[i].class_id
            p = mapPoint(self._kf_counter, kf_id, T_cw, points_pos[i], keypoints[i], descriptors[i])
            self.points[kpt_id] = p
        self._kf_counter += 1

        print(f"Adding {len(points_pos)} points to the Map. Total: {len(self.points)} points.")

    def update_points(self, 
                      kf_id: int,
                      keypoints: List[cv2.KeyPoint], 
                      descriptors: np.ndarray,
                      T_cw: np.ndarray):
        for i in range(len(keypoints)):
            kpt_id = keypoints[i].class_id
            p = self.points[kpt_id]
            p.observe(kf_id, T_cw, keypoints[i], descriptors[i])

        print(f"Updating {len(keypoints)} map points.")
        

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
            for p in self.points_arr[self._in_view_mask]:
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
        removed_point_ids = []
        for p_id, p in self.points.items():
            if p.obs_counter > 4:
                match_view_ratio = p.match_counter / p.obs_counter
                if match_view_ratio < self._match_view_ratio:
                    removed_point_ids.append(p_id)

        for p_id in removed_point_ids:
            del self.points[p_id]

        print(f"\t Match-View ratio check removed {len(removed_point_ids)} points!")

        # 2) Remove points that are rarely seen
        removed_point_ids1 = []
        for p_id, p in self.points.items():
            num_kf_passed = self._kf_counter
            point_creation_kf = p.kf_number
            num_kf_passed_since_point_creation = num_kf_passed - point_creation_kf

            if num_kf_passed_since_point_creation > 3 and p.obs_counter > min_observations:
                removed_point_ids1.append(p_id)

        for p_id in removed_point_ids1:
            del self.points[p_id]

        print(f"\t Observability check removed {len(removed_point_ids1)} points!")

        print(f"\t Removed {len(removed_point_ids) + len(removed_point_ids1)}/{prev_num_points} points from the map!")

        # Reset the in-view mask
        self._in_view_mask = None
