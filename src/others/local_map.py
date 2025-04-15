from typing import List
import numpy as np
import matplotlib.pyplot as plt
import cv2
from src.others.frame import Frame
from src.others.utils import invert_transform
from config import SETTINGS, K


scale_factor = SETTINGS["orb"]["scale_factor"]
n_levels = SETTINGS["orb"]["level_pyramid"]

MIN_OBSERVATIONS = SETTINGS["map"]["min_observations"]
MAX_KEYFRAMES_SINCE_LAST_OBS = SETTINGS["map"]["max_keyframes_since_last_observation"]
MATCH_VIEW_RATIO = SETTINGS["map"]["match_view_ratio"]

W = SETTINGS["camera"]["width"]
H = SETTINGS["camera"]["height"]

debug = SETTINGS["generic"]["debug"]


class mapPoint():
    def __init__(self, 
                 kf_number: int,
                 kf: Frame, 
                 pos: np.ndarray, 
                 keypoint: cv2.KeyPoint, 
                 desc: np.ndarray):
        self.observations = [
            { 
                "kf_number": kf_number, # The keyframe number (not ID!) when it was obsrved
                "keyframe": kf,         # The keyframe that observed it
                "keypoint": keypoint,   # ORB keypoint
                "descriptor": desc      # ORB descriptor
            }
        ]

        self.pos: np.ndarray = pos          # 3D position
        self.id: int = keypoint.class_id    # The unique id of the keypoint

        self.match_counter: int = 0         # Number of times the point was tracked with PnP
        self.obs_counter: int = 0           # Number of times the point was observed in a Frame

    def observe(self,
                kf_number: int, 
                kf: Frame, 
                keypoint: cv2.KeyPoint, 
                desc: np.ndarray):
        
        new_observation = {
            "kf_number": kf_number,
            "keyframe": kf,    
            "keypoint": keypoint, 
            "descriptor": desc
        }
        self.observations.append(new_observation)

    @property
    def best_descriptor(self):
        """
        The descriptor whose hamming distance is minimum with respect to all other 
        associated descriptors in the keyframes in which the point is observed.
        Basically the mean descriptor.
        """
        # Iterate over all observations
        min_dist = np.inf
        best_obs_idx = 0
        for i, obs in enumerate(self.observations):
            # Get the descriptor
            desc = obs["descriptor"]
            dist_sum = 0
            # Iterate over all other observations
            for j, other_obs in enumerate(self.observations):
                if i == j:
                    continue
                # Calculate the distance with their descriptor
                else:
                    other_desc = other_obs["descriptor"]
                    dist = cv2.norm(desc, other_desc, cv2.NORM_HAMMING)
                    dist_sum += dist

            # Find the minimum distance
            if dist_sum < min_dist:
                min_dist = dist_sum
                best_obs_idx = i

        # Keep the descriptor of the observation with the minimum distance to the others
        best_desc = self.observations[best_obs_idx]["descriptor"]
        
        return best_desc

    def view_ray(self, cam_pos):
        v = self.pos - cam_pos
        v = v / np.linalg.norm(v)
        return v
    
    def mean_view_ray(self):
        view_rays = []
        for obs in self.observations:
            v = self.view_ray(obs["keyframe"].pose[:3, 3])
            view_rays.append(v)

        return np.mean(view_rays, axis=0)

    def getScaleInvarianceLimits(self):
        cam_pos = self.observations[-1]["keyframe"].pose[:3, 3]
        level = self.observations[-1]["keypoint"].octave

        dist = np.linalg.norm(self.pos - cam_pos)
        minLevelScaleFactor = scale_factor**level
        maxLlevelScaleFactor = scale_factor**(n_levels - 1 - level)

        dmin = (1 / scale_factor) * dist / minLevelScaleFactor
        dmax = scale_factor * dist * maxLlevelScaleFactor

        return (dmin, dmax)

    def project2frame(self, frame: Frame) -> tuple[int]:
        """Projects a point into a frame"""
        # Get the world2frame coord
        T_wf = invert_transform(frame.pose)
        
        # Convert the world coordinates to frame coordinates
        pos_c = T_wf @ self.pos

        # Convert the xyz coordinates to pixels
        fx = K[0,0]
        fy = K[1,1]
        cx = K[2,0]
        cy = K[2,1]
        x, y, z = pos_c
        u = int(fx * x / z + cx)
        v = int(fy * y / z + cy)

        # Ensure it is inside the image bounds
        if u < 0 or u > W or v < 0 or v > H:
            return None
        else:
            return (u, v)

class Map():
    def __init__(self, ref_frame_id: int = None):
        self.points: dict = {}   # Dictionary with id<->mapPoint pairs

        # Mask that indicates which of the current points are in the camera view
        self._in_view_mask = None

        # Mask that indicates which of the current points were tracked with PnP
        self._tracking_mask = None

        # The pixel coordinates of the points in the current camera view
        self._u = None
        self._v = None

        # Frame counter
        self._kf_counter = 0

        # Reference frame
        if ref_frame_id is not None:
            self.ref_frame_id = ref_frame_id

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
        positions = np.vstack(positions, dtype=np.float64)
        return positions

    @property
    def point_ids(self):
        """Returns the points IDs"""
        ids = []
        for k in self.points.keys():
            ids.append(k)
        ids = np.array(ids, dtype=int)
        return ids
       
    def get_frustum_point_ids(self, keyframe: Frame):
        """Returns the points that originated from a given frame"""
        kf_pt_ids = set()
        for kpt in keyframe.keypoints:
            if kpt.class_id in self.point_ids():
                kf_pt_ids.add(kpt.class_id)

        return kf_pt_ids
   
    def discard_point(self, point_id: int):
        del self.points[point_id]

    def add_points(self, 
                   kf: Frame,
                   points_pos: np.ndarray, 
                   keypoints: List[cv2.KeyPoint], 
                   descriptors: np.ndarray):
        # Iterate over the new points
        for i in range(len(points_pos)):
            kpt_id = keypoints[i].class_id
            p = mapPoint(self._kf_counter, kf, points_pos[i], keypoints[i], descriptors[i])
            self.points[kpt_id] = p
        self._kf_counter += 1

        if debug:
            print(f"Adding {len(points_pos)} points to the Map. Total: {len(self.points)} points.")

    def update_points(self, 
                      kf: Frame,
                      keypoints: List[cv2.KeyPoint], 
                      descriptors: np.ndarray):
        for i in range(len(keypoints)):
            kpt_id = keypoints[i].class_id
            p = self.points[kpt_id]
            p.observe(self._kf_counter, kf, keypoints[i], descriptors[i])

        if debug:
            print(f"Updating {len(keypoints)} map points.")
        
    def update_landmarks(self, point_ids: set, point_positions: List):
        """Updates the 3d positions of given map points"""
        if debug:
            print("Updating landmark positions...")

        prev_point_positions = self.point_positions.copy()
        point_ids = np.array(point_ids, dtype=int)
        point_positions = np.array(point_positions, dtype=np.float64)

        # Create a boolean mask: True for IDs that exist in the map
        mask = np.isin(point_ids, self.point_ids)
        
        # Filter valid point IDs and their corresponding positions
        to_update_point_ids = point_ids[mask]
        to_update_positions = point_positions[mask]
        
        # Update the positions of the landmarks
        for pid, pos in zip(to_update_point_ids, to_update_positions):
            # print(f"Updating point {pid}:{self.points[pid].id} from {self.points[pid].pos} to {pos}")
            self.points[pid].pos = pos

        # self.show(prev_point_positions, self.point_positions)

    def get_points(self, point_ids: set[int]) -> list:
        """Returns the points that correspond to the given point ids"""
        points = [self.points[idx] for idx in point_ids]
        return points

    def set_tracking_mask(self, mask):
        """Saves a mask that shows which points were tracked by PnP"""
        self._tracking_mask = mask

    def update_counters(self):
        # Increase the view and match counters 
        for p in self.points_arr[self._in_view_mask]:
            p.obs_counter += 1
        for p in self.points_arr[self._tracking_mask]:
            p.match_counter += 1

    def get_keyframes_that_see(self, point_ids: set[int]) -> set[int]:
        """Returns all the keyframes that see a set of points"""
        keyframe_observers_ids = set()

        # Iterate over the points
        for pid in point_ids:
            point = self.points[pid]
            # Iterate over all the point observations
            for obs in point.observations:
                # Extract the keyframe of that observation
                kf = obs["keyframe"]
                keyframe_observers_ids.add(kf.id)

        return keyframe_observers_ids

    def cull(self):
        """
        Args:
            T_wc: Transformation from world to camera coordinate frame

        Remove map points that are 
            (1) rarely matches in PnP
            (2) not observed over M frames after their creation
            (3) not observed over the last N frames
        """
        if debug:
            print("Cleaning up map points...")

        prev_num_points = self.num_points

        # 1) Remove points that are rarely matched
        removed_point_ids = []
        for p_id, p in self.points.items():
            if p.obs_counter > 4:
                r = p.match_counter / p.obs_counter
                if r < MATCH_VIEW_RATIO:
                    removed_point_ids.append(p_id)

        for p_id in removed_point_ids:
            del self.points[p_id]

        if debug:
            print(f"\t Match-View ratio check removed {len(removed_point_ids)} points!")

        # 2) Remove points that are rarely seen
        removed_point_ids1 = []
        for p_id, p in self.points.items():
            num_kf_passed_since_point_creation = self._kf_counter - p.observations[0]["kf_number"]

            if num_kf_passed_since_point_creation > 3 and p.obs_counter < MIN_OBSERVATIONS:
                removed_point_ids1.append(p_id)

        for p_id in removed_point_ids1:
            del self.points[p_id]

        if debug:
            print(f"\t Observability check removed {len(removed_point_ids1)} points!")

        # 3) Remove points that have not been observed in the last N frames
        removed_point_ids2 = []
        for p_id, p in self.points.items():
            num_kf_passed_since_last_observation = self._kf_counter - p.observations[-1]["kf_number"]

            if num_kf_passed_since_last_observation > MAX_KEYFRAMES_SINCE_LAST_OBS:
                removed_point_ids2.append(p_id)

        for p_id in removed_point_ids2:
            del self.points[p_id]

        if debug:
            print(f"\t Oldness check removed {len(removed_point_ids2)} points!")

        all_removed_point_ids = removed_point_ids + removed_point_ids1 + removed_point_ids2
        if debug:
            total_removed = len(all_removed_point_ids)
            print(f"\t Removed {total_removed}/{prev_num_points} points from the map!")

        # Reset the in-view mask
        self._in_view_mask = None

    def show(self, prev_point_positions, point_positions):
        """
        Visualize the map points in 3D.
        """
        # Create a new figure and a 3D subplot
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the previous point positions in red
        ax.scatter(prev_point_positions[:, 0],
                prev_point_positions[:, 1],
                prev_point_positions[:, 2],
                facecolors='none', edgecolors='r', marker='o', label='Landmarks')
        
        # Plot the current point positions in blue
        ax.scatter(point_positions[:, 0],
                point_positions[:, 1],
                point_positions[:, 2],
                c='b', marker='o', alpha=0.2, label='Optimized Landmarks')
        
        # Label the axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Add a legend to differentiate the point clouds
        ax.legend()
        errors = point_positions - prev_point_positions
        errors_norm = np.linalg.norm(errors, axis=1)
        ax.set_title("Map Points <-> Error" + 
                     f"\nTotal: {np.sum(errors):.2f}" +
                     f", Mean: {np.mean(errors_norm):.2f}" +
                     f", Median: {np.median(errors_norm):.2f}" +
                     f"\nMin: {np.min(errors_norm):.2f}" +
                     f", Max: {np.max(errors_norm):.2f}")
        
        # Display the plot
        plt.show()
