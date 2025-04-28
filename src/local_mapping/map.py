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

debug = SETTINGS["generic"]["debug"]


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

        self.pos: np.ndarray = pos             # 3D position
        self.pos_before_ba: np.ndarray = None  # 3D position before BA
        self.ba = False                        # Whether the point was uptimized with BA
        self.id: int = mapPoint._mp_id_counter # The unique id of this map point

        self.observations = []
        self.tracked_counter: int = 1         # Number of times the point was tracked
        self.visible_counter: int = 1       # Number of times the point was predicted to be visible by a Frame

    def observe(self,
                kf_number: int, 
                kf_id: int, 
                kpt: cv2.KeyPoint, 
                desc: np.ndarray):
        
        new_observation = mpObservation(kf_number, kf_id, kpt, desc)
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
        """Removes the observation from a specific keyframe"""
        for obs in self.observations:
            if obs.kf_id == kf_id:
                self.observations.remove(obs)
                break

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
        self.last_reloc = 0

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


    def num_points(self) -> int:
        return len(self.points.keys())

    def num_keyframes(self) -> int:
        return len(self.keyframes.keys())


    def add_keyframe(self, kf: utils.Frame):
        if kf.id in self.keyframes.keys():
            return
        
        log.info(f"[Map] Adding frame #{kf.id} to the map.")
        self.keyframes[kf.id] = kf

        self.trajectory[kf.id] = kf.pose.copy()
        self.ba_trajectory[kf.id] = kf.pose.copy()
        self.gt_trajectory[kf.id] = kf.gt.copy()

        self._kf_counter += 1

    def add_init_points(self, points_pos: np.ndarray, 
                        q_kf: utils.Frame, q_kpts: List[cv2.KeyPoint], q_descriptors: np.ndarray, 
                        t_kf: utils.Frame, t_kpts: List[cv2.KeyPoint], t_descriptors: np.ndarray):
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
            q_kf.features[q_kpts[i].class_id].match_map_point(point, 0)
            t_kf.features[t_kpts[i].class_id].match_map_point(point, 0)

        if debug:
            log.info(f"[Map] Adding {num_new_points} points to the Map. Total: {len(self.points)} points.")

    def _add_new_point(self, pos: np.ndarray, 
                       t_frame: utils.Frame, n_frame: utils.Frame, 
                       t_feat: utils.orbFeature, n_feat: utils.orbFeature):
        """Adds a new point that was the result of triangulation between 2 frames"""
        point = mapPoint(pos)
        self.points[point.id] = point

        # Add 2 point observations (for the t_frame and n_frame that matched)
        point.observe(self._kf_counter-1, t_frame.id, t_feat.kpt, t_feat.desc)
        point.observe(self._kf_counter,   n_frame.id, n_feat.kpt, n_feat.desc)    
        
        # Set the feature <-> mapPoint matches
        t_feat.match_map_point(point, 0)
        n_feat.match_map_point(point, 0)

    def add_observation(self, frame: utils.Frame, feat: utils.orbFeature, point: mapPoint):
        point.observe(self._kf_counter, frame.id, feat.kpt, feat.desc)

    def create_track_points(self, t_frame: utils.Frame):
        """
        Creates and adds new points to the map, by triangulating matches in so far
        un-matched points in connected keyframes.
        """
        log.info(f"[Map] Creating new map points using frame {t_frame.id}")
        # For each unmatched ORB in Ki we search a match with an un-matched point in other keyframe

        # Get the neighbor frames in the convisibility graph
        neighbor_kf_ids = ctx.cgraph.get_connected_frames(t_frame.id)

        matches = []
        num_created_points = 0
        ratio_factor = 1.5 * t_frame.scale_factors[1]

        # Iterate over all neighbor frames
        for n_frame_id in neighbor_kf_ids:
            n_frame: utils.Frame = ctx.map.keyframes[n_frame_id]

            # Check that the baseline is not too short
            # Small translation errors for short baseline keyframes make scale to diverge
            baseline = np.linalg.norm(t_frame.pose[:3, 3] - n_frame.pose[:3, 3])
            median_depth = n_frame.median_depth(self)
            if baseline / median_depth < 0.01:
                continue

            # Find t<->n pairs for every word
            # (t_feature_id: neighbor_feature_id, dist)
            pairs = track.search_for_triangulation(n_frame, t_frame)
            log.info(f"\t Connected frame #{n_frame_id}: Found {len(pairs.keys())} potential points from Visual Words!") 

            # For every formed pair, utils.triangulate new points and add them to the map
            for t_feat_id, (n_feat_id, dist) in pairs.items():
                t_feat: utils.orbFeature = t_frame.features[t_feat_id]
                n_feat: utils.orbFeature = n_frame.features[n_feat_id]

                # Compute the transformation between the 2 frames and the new point
                T_tn = utils.compute_T12(t_frame, n_frame)
                new_t_point = utils.triangulate(t_feat.kpt.pt, n_feat.kpt.pt, T_tn).flatten()
                new_n_point = T_tn[:3, :3] @ new_t_point + T_tn[:3, 3]

                # To accept the new points, ensure
                # 1) positive depth in both cameras, 
                if new_t_point[2] < 0 or new_n_point[2] < 0:
                    continue

                # 2) sufficient parallax, 
                angle = utils.triangulation_angles(new_t_point, new_n_point, T_tn)
                if angle < SETTINGS["map"]["min_triang_angle"]:
                    continue
                
                # 3) reprojection error
                error = utils.triang_points_reprojection_error(new_t_point, n_feat.kpt.pt, T_tn)
                if error > SETTINGS["map"]["max_reproj_threshold"]:
                    continue
                
                # 4) and scale consistency are checked.
                ## Compute the distance between the point and the camera frame
                t_dist = np.linalg.norm(new_t_point - t_frame.pose[:3, 3])
                n_dist = np.linalg.norm(new_n_point - n_frame.pose[:3, 3])
                if t_dist == 0 or n_dist == 0:
                    continue
                ratio_dist = t_dist / n_dist
                ## Compute the ORB scale factor of every feature 
                ## The scale factor is basically how big each feature is expected to be,
                ## based on the pyramid level that it is detected on
                t_scale_factor = t_frame.scale_factors[t_feat.kpt.octave]
                n_scale_factor = n_frame.scale_factors[n_feat.kpt.octave] 
                ratio_octave = t_scale_factor / n_scale_factor
                ## Make sure that the change in point size *somewhat* agrees
                ## with the predicted size change based on pyramid levels
                if (ratio_octave / ratio_factor < ratio_dist and 
                    ratio_dist < ratio_octave * ratio_factor):
                    continue

                # Point was accepted
                new_w_point = t_frame.R @ new_t_point + t_frame.pose[:3, 3]
                self._add_new_point(new_w_point, t_frame, n_frame, t_feat, n_feat)
                matches.append((t_feat.idx, n_feat.idx, dist))

                num_created_points += 1

            if debug:
                cv2_matches = [cv2.DMatch(t, n, d) for (t,n,d) in matches]
                save_path = results_dir / "map" / "new_points" / f"{t_frame.id}_{n_frame.id}.png"
                vis.plot_matches(cv2_matches, t_frame, n_frame, save_path)

        log.info(f"\t Created {num_created_points} points!") 

        # TODO:
        # Initially a map point is observed from two keyframes but
        # it could be matched in others, so it is projected in the rest
        # of connected keyframes, and correspondences are searched


    def optimize_pose(self, kf_id: int, pose: np.ndarray):
        self.ba_trajectory[kf_id] = pose.copy()
        self.keyframes[kf_id].optimize_pose(pose)

    def optimize_point(self, pid: int, new_pos: np.ndarray):
        self.points[pid].optimize_pos(new_pos)


    def remove_observation(self, kf_id: int):
        """Removes all the point ovservations from the given keyframe"""
        for p in self.points.values():
            p.remove_observation(kf_id)

    def remove_keyframe(self, kf_id: int):
        del self.keyframes[kf_id]
        self.remove_observation(kf_id)
        if debug:
            log.info(f"\t Removed Keyframe {kf_id}. {self.num_keyframes()} left!")

    def remove_point(self, pid: int):
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
        if debug:
            log.info("[Map] Cleaning up map points...")

        prev_num_points = self.num_points()
        removed_point_ids = set()

        # Iterate over all points
        for pid, p in self.points.items():
            
            # The tracking must find the point in more than 25% of the frames in which 
            # it is predicted to be visible.
            r = p.tracked_counter / p.visible_counter
            if r < MATCH_VIEW_RATIO:
                removed_point_ids.add(pid)
                continue

            # If more than one keyframe has passed from map point creation, 
            # it must be observed from at least three keyframes.
            num_kf_passed_since_creation = self._kf_counter - p.observations[0].kf_number
            num_observations = len(ctx.cgraph.get_frames_that_observe_point(pid))
            if num_kf_passed_since_creation > 1 and num_observations < 3:
                removed_point_ids.add(pid)
                continue
            
            # Once a map point has passed this test, it can only be 
            # removed if at any time it is observed from less than three keyframes.
            # elif num_kf_passed_since_creation > 2:
            #     num_observations = cgraph.get_frames_that_observe_point(pid)
            #     if num_observations < 3:
            #         removed_point_ids.add(pid)
            #         continue

        for pid in removed_point_ids:
            self.remove_point(pid)
        ctx.cgraph.remove_points(removed_point_ids)

        if debug:
            log.info(f"\t Removed {len(removed_point_ids)} points. {self.num_points()} left!")

    def cull_keyframes(self, frame: utils.Frame):
        """
        Discard all the connected keyframes whose 90% of the
        map points have been seen in at least other three keyframes in
        the same or finer scale.
        """
        if debug:
            log.info("[Map] Cleaning up frames...")
        # Get the neighbor keyframes in the convisibility graph
        neighbor_kf_ids = ctx.cgraph.get_connected_frames(frame.id)

        # Iterate over all connected keyframes
        removed_kf_ids = set()
        for kf_id in neighbor_kf_ids:
            # Extract their map points
            kf_mp_ids = ctx.cgraph.get_frustum_point_ids(kf_id)
            num_points = len(kf_mp_ids)
            
            # Count the number of co-observing points
            count_coobserving_points = 0
            # Iterate over all their map points
            for pid in kf_mp_ids:
                # Extract the scale of their observation
                point: mapPoint = self.points[pid]
                obs = point.get_observation(kf_id)
                scale = obs.kpt.octave
                # Get how many keyframes observe the same point in the same or finer scale
                observing_frame_ids = ctx.cgraph.get_frames_that_observe_point_at_scale(pid, scale)
                num_observing_frame_ids = len(observing_frame_ids)
                # Check if at least 3 keyframes observe this point
                if num_observing_frame_ids >= 3:
                    count_coobserving_points += 1

            # Calculate the percentage of co-observing points
            if count_coobserving_points / num_points > 0.9:
                # Remove keyframe
                removed_kf_ids.add(kf_id)

        for kf_id in removed_kf_ids:
            self.remove_keyframe(kf_id)
            ctx.cgraph.remove_keyframe(kf_id)


    def view(self, frame: utils.Frame):
        """Increases the counter that shows how many times a point was predicted to be visible"""
        for p in self.points.values():
            if frame.is_in_frustum(p):
                p.visible_counter += 1

    def tracked(self, frame: utils.Frame):
        """Increases the counter that shows how many times a point was tracked"""
        frame_pid_matches = set()
        for feat in frame.features.values():
            if feat.matched:
                pid = feat.mp.id
                frame_pid_matches.add(pid)
        
        for pid in frame_pid_matches:
            self.points[pid].tracked_counter += 1

    def relocalize(self, frame_id: int):
        self.last_reloc = frame_id