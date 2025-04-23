from typing import List
import numpy as np
import cv2
from src.others.frame import Frame
from src.others.linalg import invert_transform, transform_points
from src.others.epipolar_geometry import dist_epipolar_line, compute_F12, compute_T12, triangulate, triangulation_angles, reprojection_error
from src.others.scale import get_scale_invariance_limits
from config import SETTINGS, K, log, fx, fy, cx, cy


scale_factor = SETTINGS["orb"]["scale_factor"]
n_levels = SETTINGS["orb"]["level_pyramid"]

MIN_OBSERVATIONS = SETTINGS["map"]["min_observations"]
MATCH_VIEW_RATIO = SETTINGS["map"]["match_view_ratio"]

W = SETTINGS["camera"]["width"]
H = SETTINGS["camera"]["height"]

debug = SETTINGS["generic"]["debug"]


class Observation():
    """Represents a single observation of a map point"""
    def __init__(self):
        pass


class mapPoint():
    # This is a class-level (static) variable that all mapPoint instances share.
    _mp_id_counter = -1
    def __init__(self, pos: np.ndarray):
        mapPoint._mp_id_counter += 1

        self.observations = []
        """
        observations = [
            { 
                "kf_number": kf_number, # The keyframe number (not ID!) when it was observed
                "kf_id": kf_id,         # The id of the keyframe that observed it
                "keypoint": keypoint,   # ORB keypoint
                "descriptor": desc      # ORB descriptor
            }
        ]
        """

        self.pos: np.ndarray = pos             # 3D position
        self.id: int = mapPoint._mp_id_counter # The unique id of this map point

        self.found_counter: int = 1         # Number of times the point was tracked
        self.visible_counter: int = 1       # Number of times the point was predicted to be visible by a Frame

    def observe(self,
                kf_number: int, 
                kf_id: int, 
                keypoint: cv2.KeyPoint, 
                desc: np.ndarray):
        
        new_observation = {
            "kf_number": kf_number,
            "kf_id": kf_id,    
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

    def get_observation(self, kf_id: int):
        """Returns the observation from a specific keyframe"""
        for obs in self.observations:
            if obs["kf_id"] == kf_id:
                return obs
        return None

    def view_ray(self, cam_pos: np.ndarray):
        v = self.pos - cam_pos
        v = v / np.linalg.norm(v)
        return v
    
    def mean_view_ray(self, map_keyframes: dict[int, Frame]):
        view_rays = []
        for obs in self.observations:
            kf_id = obs["kf_id"]
            if kf_id not in map_keyframes.keys():
                continue
            frame = map_keyframes[obs["kf_id"]]
            v = self.view_ray(frame.pose[:3, 3])
            view_rays.append(v)

        return np.mean(view_rays, axis=0)

    def getScaleInvarianceLimits(self, map_keyframes: dict[int, Frame]):
        for last_obs in reversed(self.observations):
            kf_id = last_obs["kf_id"]
            if kf_id not in map_keyframes.keys():
                continue
            last_obs_frame = map_keyframes[kf_id]
        cam_pos = last_obs_frame.pose[:3, 3]
        level = last_obs["keypoint"].octave

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
        R = T_w2f[:3, :3]
        t = T_w2f[:3, 3]
        
        # Convert the world coordinates to frame coordinates
        pos_c = R @ self.pos + t

        # Convert the xyz coordinates to pixels
        x, y, z = pos_c

        if z <= 0:
            return None

        u = fx * x / z + cx
        v = fy * y / z + cy

        # Ensure it is inside the image bounds
        if u < 0 or u >= W or v < 0 or v >= H:
            return None
        else:
            return (u, v)
        

class localMap():
    def __init__(self, ref_frame_id: int, 
                 K1_frames: set[int], K1_points: set[int], 
                 K2_frames: set[int], K2_points: set[int]):
        self.ref: int = ref_frame_id
        self.K1 = {
            "frames": K1_frames,
            "points": K1_points
        }
        self.K2 = {
            "frames": K2_frames,
            "points": K2_points
        }

        self.point_ids = K1_points.union(K2_points)
        self.frame_ids = K1_frames.union(K2_frames).union({ref_frame_id})
        log.info(f"[Local Map] Created map with {len(self.frame_ids)} frames and {len(self.point_ids)} points")

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
   
    def point_positions(self):
        """Returns the points xyz positions"""
        positions = np.empty((len(self.points.keys()), 3), dtype=np.float64)
        for i, v in enumerate(self.points.values()):
            positions[i] = v.pos
        return positions

    def point_ids(self):
        """Returns the points IDs"""
        ids = np.array(self.points.keys(), dtype=int)
        return ids


    def get_points_from_frame(self, kf_id: int):
        """Returns the points that are seen by a specific frame"""
        kf_pt_ids = set()
        for pid, point in self.points.items():
            for obs in point.observations:
                if obs["kf_id"] == kf_id:
                    kf_pt_ids.add(pid)

        return kf_pt_ids
    
    def get_frustum_points(self, kf_id: int):
        """Returns the points that are seen by a specific frame"""
        kf_pt_ids = set()
        for pid, point in self.points.items():
            for obs in point.observations:
                if obs["kf_id"] == kf_id:
                    kf_pt_ids.add(pid)

        return kf_pt_ids
   
    def get_points(self, point_ids: set[int]) -> list:
        """Returns the points that correspond to the given point ids"""
        points = [self.points[idx] for idx in point_ids]
        return points

    def get_keyframe(self, frame_id: int) -> Frame:
        return self.keyframes[frame_id]

    def get_keyframes_that_see(self, point_ids: set[int]) -> set[int]:
        """Returns all the keyframes that see a set of points"""
        keyframe_observers_ids = set()

        # Iterate over the points
        for pid in point_ids:
            point = self.points[pid]
            # Iterate over all the point observations
            for obs in point.observations:
                # Extract the keyframe of that observation
                kf_id = obs["kf_id"]
                keyframe_observers_ids.add(kf_id)

        return keyframe_observers_ids
    
    def get_keyframes_that_see_at_scale(self, pid: int, scale: int) -> set[int]:
        """Returns all the keyframes that see a point at a specific or finer scale"""
        keyframe_observers_ids = set()
        # Extract point
        point = self.points[pid]
        # Iterate over all the point observations
        for obs in point.observations:
            # Extract the octave of the observation
            octave = obs["keypoint"].octave
            # Compare the observing octave with the desired one
            if octave <= scale:
                keyframe_observers_ids.add(obs["kf_id"])

        return keyframe_observers_ids
    
    def get_mean_projection_error(self) -> float:
        errors = []
        # Iterate over all map points
        mp: mapPoint
        for mp in self.points.values():
            # Extract their positions
            pos = mp.pos
            # Iterate over all their observations
            for obs in mp.observations:
                # Extract the pixel of the observation
                px = np.array(obs["keypoint"].pt)
                # Extract the frame of the observation
                frame: Frame = self.keyframes[obs["kf_id"]]
                # Project the point in the frame
                proj_px = frame.project(pos)
                # Skip points that lie behind the camera
                if proj_px is None:
                    continue
                # Calculate the error
                e = np.linalg.norm(px - np.array(proj_px))
                assert not np.isnan(e)
                
                errors.append(e)

        mean_error = np.sqrt( np.mean( np.square(errors) ) )
        log.info(f"[Map] RMS Re-Projection Error: {mean_error:.2f}")
        return mean_error


    def num_points(self) -> int:
        return len(self.points.keys())

    def num_keyframes(self) -> int:
        return len(self.keyframes.keys())
    
    def num_keyframes_that_see(self, pid: int) -> set[int]:
        """Returns the number of keyframes that see a single point"""
        return len(self.points[pid].observations)


    def add_keyframe(self, kf: Frame):
        self.keyframes[kf.id] = kf
        self._kf_counter += 1

    def add_init_points(self, points_pos: np.ndarray, 
                        q_kf: Frame, q_kpts: List[cv2.KeyPoint], q_descriptors: np.ndarray, 
                        t_kf: Frame, t_kpts: List[cv2.KeyPoint], t_descriptors: np.ndarray):
        # Iterate over the new points
        num_new_points = len(points_pos)
        for i in range(num_new_points):
            # Create a point
            pos = points_pos[i]
            assert not np.any(np.isnan(pos))
            point = mapPoint(pos)
            self.points[point.id] = point

            # Add 2 point observations (for the q_frame and t_frame that matched)
            point.observe(self._kf_counter-1, q_kf.id, q_kpts[i], q_descriptors[i])
            point.observe(self._kf_counter,   t_kf.id, t_kpts[i], t_descriptors[i])    
            
            # Set the feature <-> mapPoint matches
            q_kf.features[q_kpts[i].class_id].match_map_point(point.id, 0)
            t_kf.features[t_kpts[i].class_id].match_map_point(point.id, 0)

        if debug:
            log.info(f"[Map] Adding {num_new_points} points to the Map. Total: {len(self.points)} points.")

    def create_track_points(self, cgraph, t_frame, keyframes, bow_db):
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
            self._add_point(t_frame, new_w_point, n_feat.kpt, n_feat.desc)
            num_created_points += 1

        log.info(f"\t Created {num_created_points} points!") 

        # TODO:
        # Initially a map point is observed from two keyframes but
        # it could be matched in others, so it is projected in the rest
        # of connected keyframes, and correspondences are searched


    def remove_keyframe(self, frame_id: int):
        del self.keyframes[frame_id]

    def remove_point(self, pid: int):
        del self.points[pid]


    def cull(self, frame, cgraph):
        self._cull_points(cgraph)
        self._cull_keyframes(frame, cgraph)

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

        prev_num_points = self.num_points()
        removed_point_ids = set()

        # Iterate over all points
        for pid, p in self.points.items():
            
            # The tracking must find the point in more than 25% of the frames in which 
            # it is predicted to be visible.
            r = p.found_counter / p.visible_counter
            if r < MATCH_VIEW_RATIO:
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
            self.remove_point(pid)
        cgraph.remove_points(removed_point_ids)

        if debug:
            log.info(f"\t Removed {len(removed_point_ids)}/{prev_num_points} points from the map!")

    def _cull_keyframes(self, frame, cgraph):
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
            # Extract their map points
            kf_mp_ids = cgraph.get_frustum_point_ids(kf_id)
            num_points = len(kf_mp_ids)
            
            # Count the number of co-observing points
            count_coobserving_points = 0
            # Iterate over all their map points
            for pid in kf_mp_ids:
                # Extract the scale of their observation
                point = self.points[pid]
                obs = point.get_observation(kf_id)
                scale = obs["keypoint"].octave
                # Get how many keyframes observe the same point in the same or finer scale
                observing_frame_ids = self.get_keyframes_that_see_at_scale(pid, scale)
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
            cgraph.remove_keyframe(kf_id)

    def view(self, frame: Frame):
        """Increases the counter that shows how many times a point was predicted to be visible"""
        for feat in frame.features.values():
            if feat.matched:
                pid = feat.mp["id"]
                self.points[pid].visible_counter += 1

    def found(self, frame: Frame):
        """Increases the counter that shows how many times a point was tracked"""
        for feat in frame.features.values():
            if feat.matched:
                pid = feat.mp["id"]
                self.points[pid].found_counter += 1
