from typing import List
import numpy as np
import cv2
import src.utils as utils
import src.tracking as track
import src.visualization as vis
import src.globals as ctx
from config import SETTINGS, K, log, fx, fy, cx, cy, results_dir


scale_factor = SETTINGS["orb"]["scale_factor"]
n_levels = SETTINGS["orb"]["level_pyramid"]

MIN_OBSERVATIONS = SETTINGS["map"]["min_observations"]
MATCH_VIEW_RATIO = SETTINGS["map"]["match_view_ratio"]

DIST_THRESH = SETTINGS["point_association"]["hamming_threshold"]

W = SETTINGS["camera"]["width"]
H = SETTINGS["camera"]["height"]

DEBUG = SETTINGS["generic"]["debug"]


class mpObservation():
    """Represents a single observation of a map point"""
    def __init__(self, kf_number: int, kf_id: int, kpt: cv2.KeyPoint, desc: np.ndarray):
        self.kf_number: int=kf_number # The keyframe map number (not ID!) when it was observed
        self.kf_id: int=kf_id         # The id of the keyframe that observed it
        self.kpt: cv2.KeyPoint=kpt    # ORB keypoint
        self.desc: np.ndarray=desc    # ORB descriptor


class mapPoint():
    """Represents a single point inside a map"""
    # This is a class-level (static) variable that all mapPoint instances share.
    _mp_id_counter = -1
    def __init__(self, pos: np.ndarray):
        mapPoint._mp_id_counter += 1

        self.kf_number: int = None             # The keyframe number when it was first observed
        self.pos: np.ndarray = pos             # 3D position
        self.pos_before_ba: np.ndarray = None  # 3D position before BA
        self.ba = False                        # Whether the point was uptimized with BA
        self.id: int = mapPoint._mp_id_counter # The unique id of this map point

        self.observations = []
        self.tracked_counter: int = 1        # Number of times the point was tracked
        self.visible_counter: int = 1       # Number of times the point was predicted to be visible by a Frame

    def observe(self,
                kf_number: int, 
                kf_id: int, 
                kpt: cv2.KeyPoint, 
                desc: np.ndarray):
        
        new_observation = mpObservation(kf_number, kf_id, kpt, desc)
        if len(self.observations) == 0:
            self.kf_number = kf_number
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
            desc = obs.desc
            dist_sum = 0
            # Iterate over all other observations
            for j, other_obs in enumerate(self.observations):
                if i == j:
                    continue
                # Calculate the distance with their descriptor
                else:
                    other_desc = other_obs.desc
                    dist = cv2.norm(desc, other_desc, cv2.NORM_HAMMING)
                    dist_sum += dist

            # Find the minimum distance
            if dist_sum < min_dist:
                min_dist = dist_sum
                best_obs_idx = i

        # Keep the descriptor of the observation with the minimum distance to the others
        best_desc = self.observations[best_obs_idx].desc
        
        return best_desc

    @property
    def num_observations(self):
        return len(self.observations)

    def optimize_pos(self, pos: np.ndarray):
        self.pos = pos
        self.pos_before_ba = pos.copy()
        self.ba = True

    def get_observation(self, kf_id: int):
        """Returns the observation from a specific keyframe"""
        for obs in self.observations:
            if obs.kf_id == kf_id:
                return obs
        return None
    
    def get_scale_observations(self, scale: int):
        """Returns the keyframes that observe this map point at a scale or finer"""
        kf_ids = set()
        for obs in self.observations:
            octave = obs.kpt.octave
            # Compare the observing octave with the desired one
            if octave <= scale:
                kf_ids.add(obs.kf_id)
        return kf_ids

    def remove_observation(self, kf_id: int):
        """Removes an observation from a specific keyframe"""
        self.observations = [
           obs for obs in self.observations
            if obs.kf_id != kf_id
        ]

    def view_ray(self, pos: np.ndarray):
        v = self.pos - pos
        v = v / np.linalg.norm(v)
        return v
    
    def view_ray_(self, frame_id: int):
        frame = ctx.map.keyframes[frame_id]
        v = self.pos - frame.t
        v = v / np.linalg.norm(v)
        return v
    
    def mean_view_ray(self):
        if self.num_observations == 0:
            return None
        
        keyframes = ctx.map.keyframes
        view_rays = []
        for obs in self.observations:
            kf_id = obs.kf_id
            if kf_id not in keyframes.keys():
                continue
            frame = keyframes[obs.kf_id]
            v = self.view_ray(frame.t)
            view_rays.append(v)

        return np.mean(view_rays, axis=0)

    def getScaleInvarianceLimits(self):
        """
        Compute the minimum/maximum distances at which this point 
        should be visible on different ORB pyramid levels
        """
        keyframes = ctx.map.keyframes
        for last_obs in reversed(self.observations):
            kf_id = last_obs.kf_id
            if kf_id not in keyframes.keys():
                continue
            last_obs_frame = keyframes[kf_id]
        cam_pos = last_obs_frame.pose[:3, 3]
        level = last_obs.kpt.octave

        dist = np.linalg.norm(self.pos - cam_pos)
        minLevelScaleFactor = scale_factor**level
        maxLlevelScaleFactor = scale_factor**(n_levels - 1 - level)

        dmin = (1 / scale_factor) * dist / minLevelScaleFactor
        dmax = scale_factor * dist * maxLlevelScaleFactor

        return (dmin, dmax)

    def project2frame(self, frame: utils.Frame) -> tuple[int]:
        """Projects a point into a frame"""
        # Get the world2frame coord
        T_w2f = utils.invert_transform(frame.pose)
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
    """Represents a part of the full map"""
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

        if DEBUG:
            log.info(f"[Local Map] Created map with {len(self.frame_ids)} frames and {len(self.point_ids)} points")


class Map():
    def __init__(self, ref_frame_id: int = None):
        self.trajectory = {} # Dictionary with id <-> position pairs
        self.ba_trajectory = {} # Dictionary with id <-> optimized_position pairs
        self.gt_trajectory = {} # Dictionary with id <-> ground turth pose

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

        # ID of last relocalization keyframe
        self.last_loop = 0

        # Reference frame
        if ref_frame_id is not None:
            self.ref_frame_id = ref_frame_id

    def poses(self, ba=False) -> np.ndarray:
        if ba:
            traj_3d = np.array(list(self.ba_trajectory.values()))
        else:
            traj_3d = np.array(list(self.trajectory.values()))
        return traj_3d

    def ground_truth(self) -> np.ndarray:
        gt = np.array(list(self.gt_trajectory.values()))
        return gt

    def point_positions(self, ba: bool):
        """Returns the points xyz positions"""
        positions = np.empty((len(self.points.keys()), 3), dtype=np.float64)
        # Positions after Bundle Adjustment
        if ba:
            for i, v in enumerate(self.points.values()):
                positions[i] = v.pos
        # Positions before Bundle Adjustment
        else:
            for i, v in enumerate(self.points.values()):
                positions[i] = v.pos_before_ba if v.ba else v.pos
        return positions

    def point_ids(self):
        """Returns the points IDs"""
        ids = np.array(self.points.keys(), dtype=int)
        return ids

      
    def get_points(self, point_ids: set[int]) -> list:
        """Returns the points that correspond to the given point ids"""
        points = [self.points[idx] for idx in point_ids]
        return points

    def get_keyframe(self, frame_id: int) -> utils.Frame:
        return self.keyframes[frame_id]

    def get_num_observations(self, kf_id: int):
        num_obs = 0
        for p in self.points.values():
            for obs in p.observations:
                if obs.kf_id == kf_id:
                    num_obs += 1

        return num_obs

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
                px = np.array(obs.kpt.pt)
                # Extract the frame of the observation
                frame: utils.Frame = self.keyframes[obs.kf_id]
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
        return mean_error


    @property
    def num_points(self) -> int:
        return len(self.points.keys())

    @ property
    def num_keyframes(self) -> int:
        return len(self.keyframes.keys())

    @property
    def keyframe_ids(self) -> set[int]:
        return set(self.keyframes.keys())


    def add_keyframe(self, kf: utils.Frame):
        if kf.id in self.keyframes.keys():
            return
        
        if DEBUG:
            log.info(f"[Map] Adding frame #{kf.id} to the map.")
        
        self.keyframes[kf.id] = kf

        self.trajectory[kf.id] = kf.pose.copy()
        self.ba_trajectory[kf.id] = kf.pose.copy()
        self.gt_trajectory[kf.id] = kf.gt.copy()

        self._kf_counter += 1

    def add_init_points(self, points_pos: np.ndarray, distances: np.ndarray,
                        q_kf: utils.Frame, q_kpts: List[cv2.KeyPoint], q_descriptors: np.ndarray, 
                        t_kf: utils.Frame, t_kpts: List[cv2.KeyPoint], t_descriptors: np.ndarray):
        # Iterate over the new points
        num_new_points = len(points_pos)
        for i in range(num_new_points):
            # Create a point
            pos = points_pos[i]
            dist = distances[i]
            assert not np.any(np.isnan(pos))
            point = mapPoint(pos)
            self.points[point.id] = point

            # Add 2 point observations (for the q_frame and t_frame that matched)
            point.observe(self._kf_counter-1, q_kf.id, q_kpts[i], q_descriptors[i])
            point.observe(self._kf_counter,   t_kf.id, t_kpts[i], t_descriptors[i])    
            
            # Set the feature <-> mapPoint matches
            q_kf.features[q_kpts[i].class_id].match_map_point(point, dist)
            t_kf.features[t_kpts[i].class_id].match_map_point(point, dist)

        if DEBUG:
            log.info(f"[Map] Adding {num_new_points} points to the Map. Total: {len(self.points)} points.")

    def _add_new_point(self, pos: np.ndarray, dist: float,
                       q_frame: utils.Frame, t_frame: utils.Frame, 
                       q_feat: utils.orbFeature, t_feat: utils.orbFeature):
        """Adds a new point that was the result of triangulation between 2 frames"""
        point = mapPoint(pos)
        self.points[point.id] = point

        # Add 2 point observations (for the t_frame and q_frame that matched)
        point.observe(self._kf_counter-1, q_frame.id, q_feat.kpt, q_feat.desc)    
        point.observe(self._kf_counter,   t_frame.id, t_feat.kpt, t_feat.desc)
        
        # Set the feature <-> mapPoint matches
        q_feat.match_map_point(point, dist)
        t_feat.match_map_point(point, dist)

        # Add the point to the convisibility graph too
        ctx.cgraph.add_observation(q_frame.id, point.id)
        ctx.cgraph.add_observation(t_frame.id, point.id)

    def create_points(self, t_frame: utils.Frame):
        """
        Creates and adds new points to the map, by triangulating matches in so far
        un-matched points in connected keyframes.
        """
        # For each unmatched ORB in Ki we search a match with an un-matched point in other keyframe

        # Prepare matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Get the neighbor frames in the convisibility graph
        neighbor_kf_ids = ctx.cgraph.get_connected_frames(t_frame.id)
        ratio_factor = 1.5 * t_frame.scale_factors[1]
        if DEBUG:
            log.info(f"[Map] Creating new map points using frame {t_frame.id} and {len(neighbor_kf_ids)} neighbors!")

        # Iterate over all neighbor frames
        in_map_counter = 0
        cheirality_counter = 0
        reprojection_counter = 0
        parallax_counter = 0
        dist_counter = 0
        scale_counter = 0
        num_created_points = 0
        for q_frame_id in neighbor_kf_ids:
            matches = []
            q_frame: utils.Frame = ctx.map.keyframes[q_frame_id]

            # Check that the baseline is not too short
            # Small translation errors for short baseline keyframes make scale to diverge
            baseline = np.linalg.norm(t_frame.pose[:3, 3] - q_frame.pose[:3, 3])
            median_depth = q_frame.median_depth(self)
            if baseline / median_depth < 0.01:
                continue

            # Match descriptors with ratio test
            matches_knn = bf.knnMatch(q_frame.descriptors, t_frame.descriptors, k=2)
            filtered_matches = utils.filterMatches(matches_knn, 0.7)
            if DEBUG:
                log.info(f"\t Connected frame #{q_frame_id}: Found {len(filtered_matches)} potential new points!") 

            # For every formed pair, utils.triangulate new points and add them to the map
            T_q2t = utils.invert_transform(t_frame.pose) @ q_frame.pose
            for m in filtered_matches:
                q_kpt = q_frame.keypoints[m.queryIdx]
                q_feat = q_frame.features[q_kpt.class_id]
                # Skip features that are already in the map
                if q_feat.in_map:
                    in_map_counter += 1
                    continue
                t_kpt = t_frame.keypoints[m.trainIdx]
                t_feat = t_frame.features[t_kpt.class_id]

                # Compute the transformation between the 2 frames and the new point
                new_q_point = utils.triangulate(q_kpt.pt, t_kpt.pt, T_q2t).flatten()
                new_t_point = T_q2t[:3, :3] @ new_q_point + T_q2t[:3, 3]

                # To accept the new points, ensure
                # 1) positive depth in both cameras, 
                if new_t_point[2] < 0 or new_q_point[2] < 0:
                    cheirality_counter += 1
                    continue

                # 2) sufficient parallax, 
                angle = utils.triangulation_angles(new_q_point, new_t_point, T_q2t)
                if angle < 2.0:
                    parallax_counter += 1
                    continue
                
                # 3) reprojection error
                error = utils.triang_points_reprojection_error(new_q_point, t_kpt.pt, T_q2t)
                if error > 2.0:
                    reprojection_counter += 1
                    continue
                
                # 4) and scale consistency are checked.
                ## Compute the distance between the point and the camera frame
                t_dist = np.linalg.norm(new_t_point - t_frame.pose[:3, 3])
                q_dist = np.linalg.norm(new_q_point - q_frame.pose[:3, 3])
                if t_dist == 0 or q_dist == 0:
                    dist_counter += 1
                    continue
                ratio_dist = t_dist / q_dist
                ## Compute the ORB scale factor of every feature 
                ## The scale factor is basically how big each feature is expected to be,
                ## based on the pyramid level that it is detected on
                t_scale_factor = t_frame.scale_factors[t_kpt.octave]
                q_scale_factor = q_frame.scale_factors[q_kpt.octave] 
                ratio_octave = t_scale_factor / q_scale_factor
                ## Make sure that the change in point size *somewhat* agrees
                ## with the predicted size change based on pyramid levels
                if not (ratio_octave / ratio_factor < ratio_dist and 
                    ratio_dist < ratio_octave * ratio_factor):
                    scale_counter += 1
                    continue

                # Point was accepted
                new_w_point = t_frame.R @ new_t_point + t_frame.pose[:3, 3]
                self._add_new_point(new_w_point, m.distance, t_frame, q_frame, t_feat, q_feat)

                matches.append((q_feat.idx, t_feat.idx, m.distance))
                num_created_points += 1

            if DEBUG:
                cv2_matches = [cv2.DMatch(q,t,d) for (q,t,d) in matches]
                save_path = results_dir / "tracking/new_points" / f"{t_frame.id}_{q_frame.id}.png"
                vis.plot_matches(cv2_matches, q_frame, t_frame, save_path)

        # Update graph edges
        ctx.cgraph.update_edges()

        if DEBUG:
            log.info(f"\t Created {num_created_points} points! Filtered the following...")
            log.info(f"\t\t In map: {in_map_counter}")
            log.info(f"\t\t Cheirality: {cheirality_counter}")
            log.info(f"\t\t Reprojection: {reprojection_counter}")
            log.info(f"\t\t Parallax: {parallax_counter}")
            log.info(f"\t\t Distance: {dist_counter}")
            log.info(f"\t\t Scale: {scale_counter}")
            log.info(f"\t Total points: {self.num_points}!")


    def optimize_pose(self, kf_id: int, pose: np.ndarray):
        self.ba_trajectory[kf_id] = pose.copy()
        self.keyframes[kf_id].optimize_pose(pose)

    def optimize_point(self, pid: int, new_pos: np.ndarray):
        self.points[pid].optimize_pos(new_pos)


    def remove_matches(self, matches: set[int, int]):
        """
        Removes a match completely:
            1) Remove the match from the feature
            2) Remove the observation from the map point
            3) Remove the point form the convisibility graph
        """
        for pid, kf_id in matches:
            self.keyframes[kf_id].remove_mp_match(pid)
            self.points[pid].remove_observation(kf_id)
        ctx.cgraph.remove_matches(matches)

    def remove_keyframe(self, kf_id: int):
        """
        Completely removes a keyframe:
            1) Removes the keyframe from the map
            2) Removes all the keyframe observations from the map points
            3) Removes the keyframe from the convisibility graph
        """
        del self.keyframes[kf_id]
        for p in self.points.values():
            p.remove_observation(kf_id)
        if DEBUG:
            log.info(f"\t Removed Keyframe {kf_id}. {self.num_keyframes} left!")
        ctx.cgraph.remove_keyframe(kf_id)

    def remove_points(self, pids: set[int]):
        for pid in pids:
            del self.points[pid]


    def cull_points(self):
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
        if DEBUG:
            log.info("[Map] Culling map points...")

        removed_point_ids = set()

        # Iterate over all points
        for pid, p in self.points.items():
            
            # The tracking must find the point in more than 25% of the frames in which 
            # it is predicted to be visible.
            r = p.tracked_counter / p.visible_counter
            if r < MATCH_VIEW_RATIO:
                removed_point_ids.add(pid)
                continue

            # The point must always be observed during the first 2 keyframes
            num_kf_passed_since_creation = self._kf_counter - p.kf_number
            if num_kf_passed_since_creation <= 2 and p.num_observations < 2:
                removed_point_ids.add(pid)
                continue
            
            # After that, it must always be observed by at least 3 keyframes
            elif num_kf_passed_since_creation >= 3 and p.num_observations < 3:
                removed_point_ids.add(pid)
                continue

        self.remove_points(removed_point_ids)
        ctx.cgraph.remove_points(removed_point_ids)

        if DEBUG:
            log.info(f"\t Removed {len(removed_point_ids)} points. {self.num_points} left!")

    def cull_keyframes(self, frame: utils.Frame):
        """
        Discard all the connected keyframes whose 90% of the
        map points have been seen in at least other three keyframes in
        the same or finer scale.
        """
        if DEBUG:
            log.info("[Map] Culling frames...")

        # Get the neighbor keyframes in the convisibility graph
        neighbor_kf_ids = ctx.cgraph.get_connected_frames(frame.id, num_edges=30)

        removed_kf_ids = set()
        # Iterate over all connected keyframes
        for kf_id in neighbor_kf_ids:
            # Extract their map points
            kf_points = ctx.cgraph.get_frustum_points(kf_id)
            num_points = len(kf_points)
            
            # Count the number of co-observing points
            count_coobserving_points = 0
            
            # Iterate over all their map points
            for point in kf_points:
                # Skip points with too few observations
                if point.num_observations < 4:
                    continue
                # Extract the scale of their observation
                obs = point.get_observation(kf_id)
                
                # Iterate over all other point observations
                for other_obs in point.observations:
                    if other_obs.kf_id == kf_id:
                        continue
                    # Check if the other observation is in the same or finer scale
                    if other_obs.kpt.octave <= obs.kpt.octave:
                        count_coobserving_points += 1

            # Calculate the percentage of co-observing points
            if count_coobserving_points / num_points > 0.9:
                # Remove keyframe
                removed_kf_ids.add(kf_id)

        for kf_id in removed_kf_ids:
            self.remove_keyframe(kf_id)


    def view(self, frame: utils.Frame):
        """Increases the counter that shows how many times a point was predicted to be visible"""
        for p in self.points.values():
            if frame.is_in_frustum(p):
                p.visible_counter += 1

    def tracked(self, frame: utils.Frame):
        """Increases the counter that shows how many times a point was tracked"""
        for feat in frame.features.values():
            if feat.matched:
                point = feat.mp
                point.tracked_counter += 1


    def add_loop_closure(self, frame_id: int):
        self.last_loop = frame_id

    @property
    def num_keyframes_since_last_loop(self) -> int:
        return self._kf_counter - self.last_loop
