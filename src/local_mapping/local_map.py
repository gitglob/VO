from typing import List
import numpy as np
import matplotlib.pyplot as plt
import cv2
from src.others.frame import Frame
from src.others.linalg import invert_transform, transform_points
from src.others.epipolar_geometry import dist_epipolar_line, compute_F12, compute_T12, triangulate, triangulation_angles, reprojection_error
from src.others.scale import get_scale_invariance_limits
from config import SETTINGS, K, log


scale_factor = SETTINGS["orb"]["scale_factor"]
n_levels = SETTINGS["orb"]["level_pyramid"]

MIN_OBSERVATIONS = SETTINGS["map"]["min_observations"]
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

        self.found_counter: int = 1         # Number of times the point was tracked
        self.visible_counter: int = 1       # Number of times the point was predicted to be visible by a Frame

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

    def view_ray(self, cam_pos: np.ndarray):
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
        T_w2f = invert_transform(frame.pose)
        
        # Convert the world coordinates to frame coordinates
        pos_c = T_w2f @ self.pos

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
        
    def project2frame(self, T_f2w: np.ndarray) -> tuple[int]:
        """Projects a point into a frame"""
        # Get the world2frame coord
        T_w2f = invert_transform(T_f2w)
        
        # Convert the world coordinates to frame coordinates
        pos_c = T_w2f[:3, :3] @ self.pos + T_w2f[:3, 3]

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
        self.keyframes: dict = {} # Dictionary with id <-> keyframe pairs
        self.points: dict = {}    # Dictionary with id <-> mapPoint pairs

        # Masks that show which of the current points were visible/tracked
        self._in_view_mask = None
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
       
    def get_points_from_frame(self, kf_id: int):
        """Returns the points that are seen by a specific frame"""
        kf_pt_ids = set()
        for point in self.points.values():
            for obs in point.observations:
                if obs["keyframe"].id == kf_id:
                    kf_pt_ids.add(point.id)

        return kf_pt_ids
   
    def discard_point(self, point_id: int):
        del self.points[point_id]

    def add_keyframe(self, kf: Frame):
        self.keyframes[kf.id] = kf

    def add_point(self, 
                   kf: Frame,
                   pos: np.ndarray, 
                   keypoint: List[cv2.KeyPoint], 
                   descriptor: np.ndarray):
        # Iterate over the new points
        kpt_id = keypoint.class_id
        p = mapPoint(self._kf_counter, kf, pos, keypoint, descriptor)
        self.points[kpt_id] = p

        # if debug:
        #     log.info(f"[Map] Added point #{kpt_id} to the Map. Total: {len(self.points)} points.")

    def add_points(self, 
                   kf: Frame,
                   points_pos: np.ndarray, 
                   keypoints: List[cv2.KeyPoint], 
                   descriptors: np.ndarray):
        # Iterate over the new points
        num_new_points = len(points_pos)
        for i in range(num_new_points):
            self.add_point(kf, points_pos[i], keypoints[i], descriptors[i])
        self._kf_counter += 1

        if debug:
            log.info(f"[Map] Adding {num_new_points} points to the Map. Total: {len(self.points)} points.")

    def update_points(self, 
                      kf: Frame,
                      keypoints: List[cv2.KeyPoint], 
                      descriptors: np.ndarray):
        for i in range(len(keypoints)):
            kpt_id = keypoints[i].class_id
            p = self.points[kpt_id]
            p.observe(self._kf_counter, kf, keypoints[i], descriptors[i])

        if debug:
            log.info(f"[Map] Updating {len(keypoints)} map points.")
        
    def update_landmarks(self, point_ids: set, point_positions: List):
        """Updates the 3d positions of given map points"""
        if debug:
            log.info("[Map] Updating landmark positions...")

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
            # log.info(f"[Map] Updating point {pid}:{self.points[pid].id} from {self.points[pid].pos} to {pos}")
            self.points[pid].pos = pos

        # self.show(prev_point_positions, self.point_positions)

    def get_points(self, point_ids: set[int]) -> list:
        """Returns the points that correspond to the given point ids"""
        points = [self.points[idx] for idx in point_ids]
        return points

    def view(self, t_map_pairs: dict):
        """Increases the counter that shows how many times a point was predicted to be visible"""
        for pid, _ in t_map_pairs.values():
            self.points[pid].visible_counter += 1

    def found(self, t_map_pairs: dict):
        """Increases the counter that shows how many times a point was tracked"""
        for pid, _ in t_map_pairs.values():
            self.points[pid].found_counter += 1

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
    
    def num_keyframes_that_see(self, pid: int) -> set[int]:
        """Returns the number of keyframes that see a single point"""
        count = 0
        point = self.points[pid]
        for obs in point.observations:
            # Extract the keyframe of that observation
            kf = obs["keyframe"]
            count += 1

        return count

    def create_points(self, cgraph, t_frame, keyframes, bow_db):
        """
        Creates and adds new points to the map, by triangulating matches in so far
        un-matched points in connected keyframes.
        """
        log.info(f"[Map] Creating new map points using frame {t_frame.id}")
        # For each unmatched ORB in Ki we search a match with an un-matched point in other keyframe
        pairs = [] # (frame1_id, feature1_id, frame2_id, feature2_id)

        # Get the neighbor frames in the convisibility graph
        neighbor_kf_ids = cgraph.get_connected_frames(t_frame.id)

        # Iterate over the bow feature vector of the current frame
        for t_word_id, t_kpt_id in t_frame.feature_vector.items():
            t_feature = t_frame.features[t_kpt_id]
            # Check if the feature is unmatched
            if t_feature.matched:
                continue

            # Iterate over neighbor keyframes
            for n_kf_id in neighbor_kf_ids:
                # Check if this keyframe sees the same word
                if n_kf_id not in bow_db[t_word_id]:
                    continue
                n_frame = keyframes[n_kf_id]
                # Extract the neighbor feature for the same word
                n_kpt_id = n_frame.feature_vector[t_word_id]
                n_feature = n_frame.features[n_kpt_id]
                # Check if the feature is unmatched
                if n_feature.matched:
                    continue

                # Compute the descriptor distance for this word and check if it is low enough
                d = cv2.norm(t_feature.desc, n_feature.desc, cv2.NORM_HAMMING)
                if d < SETTINGS["point_association"]["hamming_distance"]:
                    # Check if the candidate pair satisfies the epipolar constraint
                    F = compute_F12(t_frame, n_frame)
                    d_epi_sqr = dist_epipolar_line(t_feature.kpt.pt, n_feature.kpt.pt, F)
                    if d_epi_sqr < n_frame.get_sigma2(n_feature.kpt.octave):
                        pairs.append((t_frame.id, t_kpt_id, n_frame.id, n_kpt_id))

        # For every formed pair, triangulate new points and add them to the map
        num_created_points = 0
        for pair in pairs:
            _, t_kpt_id, n_frame_id, n_kpt_id = pair
            n_frame = keyframes[n_frame_id]
            T_tn = compute_T12(t_frame, n_frame)
            new_t_point = triangulate(t_frame, keyframes[n_frame_id], T_tn)

            # To accept the new points, ensure
            # 1) positive depth in both cameras, 
            new_n_point = transform_points(new_t_point, T_tn)
            if new_t_point[2] < 0 or new_n_point[2] < 0:
                break

            # 2) sufficient parallax, 
            angle = triangulation_angles(new_t_point, new_n_point, T_tn)
            if angle < SETTINGS["map"]["min_triang_angle"]:
                break
             
            # 3) reprojection error
            t_feat = t_frame.features[t_kpt_id]
            n_feat = n_frame.features[n_kpt_id]
            t_pxs = t_feat.kpt.pt
            n_pxs = n_feat.kpt.pt
            error = reprojection_error(t_pxs, n_pxs, T_tn)
            if error > SETTINGS["map"]["max_reproj_threshold"]:
                break
            
            # 4) and scale consistency are checked.
            t_cam_center = t_frame.pose[:3, 3]
            dist = np.linalg.norm(new_t_point - t_cam_center)
            dmin, dmax = get_scale_invariance_limits(dist, t_feat.kpt.octave)
            if dist < dmin or dist > dmax:
                break

            # Point was accepted
            new_w_point = transform_points(new_t_point, t_frame.pose)
            self.add_point(new_w_point)
            num_created_points += 1

        log.info(f"\t Created {num_created_points} points!") 

        # TODO:
        # Initially a map point is observed from two keyframes but
        # it could be matched in others, so it is projected in the rest
        # of connected keyframes, and correspondences are searched

    def cull(self, keyframes, frame, cgraph):
        self._cull_points(cgraph)
        self._cull_frames(keyframes, frame, cgraph)

    def _cull_frames(self, keyframes, frame, cgraph):
        """
        Discard all the connected keyframes whose 90% of the
        map points have been seen in at least other three keyframes in
        the same or finer scale.
        """
        if debug:
            log.info("[Map] Cleaning up frames...")
        # Get the neighbor keyframes in the convisibility graph
        neighbor_kf_ids = cgraph.get_connected_frames(frame.id)

        # Iterate over all connected keyframes
        removed_kf_ids = set()
        for kf_id in neighbor_kf_ids:
            kf = keyframes[kf_id]
            # Extract their map points
            kf_point_ids = cgraph.get_frustum_point_ids(kf_id)
            num_points = len(kf_point_ids)
            # Count the number of co-observing points
            count_coobserving_points = 0
            # Iterate over all the points
            for pid in kf_point_ids:
                # Extract their scale
                scale = kf.features[pid].kpt.octave
                # Get how many keyframes observe the same point in the same or finer scale
                observing_frame_ids = cgraph.get_frames_that_observe(pid, keyframes, scale)
                num_observing_frame_ids = len(observing_frame_ids)
                # Check if at least 3 keyframes observe this point
                if num_observing_frame_ids >= 3:
                    count_coobserving_points += 1

            # Calculate the percentage of co-observing points
            if count_coobserving_points / num_points > 0.9:
                # Remove keyframe
                removed_kf_ids.add(kf_id)

        for kf_id in removed_kf_ids:
            if debug:
                log.info(f"\t Removed Keyframe {kf_id}!")
            cgraph.remove(kf_id)

    def _cull_points(self, cgraph):
        """
        A point must fulfill these two conditions during the first three keyframes after creation:

        1) The tracking must find the point in more than the 25%
           of the frames in which it is predicted to be visible.
        2) If more than one keyframe has passed from map point 
           creation, it must be observed from at least three keyframes.

        Once a map point have passed this test, it can only be
        removed if at any time it is observed from less than three
        keyframes.

        Basically, remove map points that are 
            (1) rarely matched
            (2) not observed from at least 3 keyframes
        """
        if debug:
            log.info("[Map] Cleaning up map points...")

        prev_num_points = self.num_points
        removed_point_ids = set()

        # Iterate over all points
        for pid, p in self.points.items():
            
            # The tracking must find the point in more than 25% of the frames in which 
            # it is predicted to be visible.
            r = p.found_counter / p.visible_counter
            if r > MATCH_VIEW_RATIO:
                removed_point_ids.add(pid)
                continue

            # If more than one keyframe has passed from map point creation, 
            # it must be observed from at least three keyframes.
            num_kf_passed_since_creation = self._kf_counter - p.observations[0]["kf_number"]
            num_observations = len(cgraph.get_frames_that_observe(pid))
            if num_kf_passed_since_creation > 1 and num_observations < 3:
                removed_point_ids.add(pid)
                continue
            
            # Once a map point has passed this test, it can only be 
            # removed if at any time it is observed from less than three keyframes.
            # elif num_kf_passed_since_creation > 2:
            #     num_observations = cgraph.get_frames_that_observe(pid)
            #     if num_observations < 3:
            #         removed_point_ids.add(pid)
            #         continue

        for pid in removed_point_ids:
            del self.points[pid]

        if debug:
            log.info(f"\t Removed {len(removed_point_ids)}/{prev_num_points} points from the map!")

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
