import os
from multiprocessing import Pool, cpu_count
import numpy as np
import cv2
import src.utils as utils
import src.visualization as vis
import src.globals as ctx
from .point_association import search_for_triangulation, window_search
from .parallel_functions import process_neighbor
from config import SETTINGS, K, log, fx, fy, cx, cy, results_dir


scale_factor = SETTINGS["orb"]["scale_factor"]
n_levels = SETTINGS["orb"]["level_pyramid"]

W = SETTINGS["camera"]["width"]
H = SETTINGS["camera"]["height"]

LOWE_RATIO = SETTINGS["map"]["lowe_ratio"]
MIN_PARALLAX = SETTINGS["map"]["min_parallax"]
MAX_REPROJECTION = SETTINGS["map"]["max_reprojection"]

MATCH_VIEW_RATIO = SETTINGS["map"]["point_culling_ratio"]
KF_CULLING_RATIO = SETTINGS["map"]["kf_culling_ratio"]

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

    def best_feature(self):
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
        kpt = self.observations[best_obs_idx].kpt
        
        return best_desc, kpt

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

    def remove_observation(self, kf_id: int) -> int:
        """Removes an observation from a specific keyframe"""
        self.observations = [
           obs for obs in self.observations
            if obs.kf_id != kf_id
        ]
        obs_kf_ids = [obs.kf_id for obs in self.observations]
        assert kf_id not in obs_kf_ids
        return len(obs_kf_ids)            

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

        log.info(f"[Local Map] Created map with {len(self.frame_ids)} frames and {len(self.point_ids)} points")


class Map():
    def __init__(self, ref_frame_id: int = None):
        self.trajectory = {} # Dictionary with id <-> position pairs
        self.ba_trajectory = {} # Dictionary with id <-> optimized_position pairs
        self.gt_trajectory = {} # Dictionary with id <-> ground turth pose

        self.keyframes: dict = {} # Dictionary with id <-> keyframe pairs
        self.points: dict = {}    # Dictionary with id <-> mapPoint pairs

        self.last_loop = None   # The last loop closure frame id
        self.loop_closures = [] # List of loop closures (loop_closure_frame, frame)

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

    def poses(self, ba=False) -> np.ndarray:
        if ba:
            traj_3d = np.array(list(self.ba_trajectory.values()))
        else:
            traj_3d = np.array(list(self.trajectory.values()))
        return traj_3d

    def ground_truth(self) -> np.ndarray:
        gt = np.array(list(self.gt_trajectory.values()))
        return gt

    def keyframe_positions(self) -> np.ndarray:
        """Returns the keyframes xyz positions"""
        positions = np.empty((len(self.keyframes.keys()), 3), dtype=np.float64)
        for i, v in enumerate(self.keyframes.values()):
            positions[i] = v.pose[:3, 3]
        return positions
    
    def loop_closure_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the loop closure xyz positions"""
        lc_poses = np.empty((len(self.loop_closures), 3), dtype=np.float64)
        lc_kf_poses = np.empty((len(self.loop_closures), 3), dtype=np.float64)
        for i, (lc_frame, frame) in enumerate(self.loop_closures):
            lc_poses[i] = lc_frame.pose[:3, 3]
            lc_kf_poses[i] = frame.pose[:3, 3]
        return lc_poses, lc_kf_poses

    def point_positions(self, ba: bool=True) -> np.ndarray:
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
                if proj_px is None: continue
                # Calculate the error
                e = e = abs(px[0] - proj_px[0]) + abs(px[1] - proj_px[1])
                assert not np.isnan(e)
                
                errors.append(e)

        mean_error = np.mean(errors)
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
        
        log.info(f"[Map] Adding frame #{kf.id} to the map.")
        
        self.keyframes[kf.id] = kf

        self.trajectory[kf.id] = kf.pose.copy()
        self.ba_trajectory[kf.id] = kf.pose.copy()
        self.gt_trajectory[kf.id] = kf.gt.copy()

        self._kf_counter += 1

    def add_new_points(self, w_points: np.ndarray, matches: np.ndarray[cv2.DMatch], 
                        q_frame: utils.Frame, t_frame: utils.Frame):
        """Adds new points to the map"""
        # Iterate over all new points
        created_points = np.empty((len(w_points),), dtype=mapPoint)
        for i, pos in enumerate(w_points):
            # Get the match
            m = matches[i]
            dist = m.distance
            q_kpt = q_frame.keypoints[m.queryIdx]
            t_kpt = t_frame.keypoints[m.trainIdx]
            q_feat = q_frame.features[q_kpt.class_id]
            t_feat = t_frame.features[t_kpt.class_id]

            # Create a point
            assert not np.any(np.isnan(pos))
            
            # Add the point to the map
            p = self._add_new_point(pos, dist, q_frame, t_frame, q_feat, t_feat)
            created_points[i] = p

        log.info(f"[Map] Adding {len(w_points)} points to the Map. Total: {self.num_points} points.")
        return created_points

    def _add_new_point(self, pos: np.ndarray, dist: float,
                       q_frame: utils.Frame, t_frame: utils.Frame, 
                       q_feat: utils.orbFeature, t_feat: utils.orbFeature):
        """Adds a new point that was the result of triangulation between 2 frames"""
        point = mapPoint(pos)
        self.points[point.id] = point

        # Add 2 point observations (for the t_frame and q_frame that matched)
        point.observe(self._kf_counter, q_frame.id, q_feat.kpt, q_feat.desc)    
        point.observe(self._kf_counter, t_frame.id, t_feat.kpt, t_feat.desc)
        
        # Set the feature <-> mapPoint matches
        q_feat.match_map_point(point, dist)
        t_feat.match_map_point(point, dist)

        # Add the point to the convisibility graph too
        ctx.cgraph.add_observation(q_frame.id, point.id)
        ctx.cgraph.add_observation(t_frame.id, point.id)

        return point


    def create_points(self, t_frame: utils.Frame):
        """
        Creates and adds new points to the map, by triangulating matches in so far
        un-matched points in connected keyframes.
        """
        # For each unmatched ORB in Ki we search a match with an un-matched point in other keyframe

        # Prepare matcher
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Get the neighbor frames in the convisibility graph
        neighbor_kf_ids = ctx.cgraph.get_connected_frames(t_frame.id, num_edges=30)
        ratio_factor = 1.5 * t_frame.scale_factors[1]
        log.info(f"[Map] Creating new map points using frame {t_frame.id} and {len(neighbor_kf_ids)} neighbors!")

        # Iterate over all neighbor frames
        epipolar_counter = 0
        cheirality_counter = 0
        reprojection_counter = 0
        parallax_counter = 0
        dist_counter = 0
        scale_counter = 0
        depth_counter = 0
        num_created_points = 0
        all_matches = {}
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
            # matches_knn = bf.knnMatch(q_frame.descriptors, t_frame.descriptors, k=2)
            # filtered_matches = utils.ratio_filter(matches_knn, LOWE_RATIO)
            # log.info(f"\t Lowe's Test filtered {len(matches_knn) - len(filtered_matches)}/{len(matches_knn)} matches!")
            # matches = utils.unique_filter(matches)
            # log.info(f"\t Uniqueness filtered {len(filtered_matches) - len(matches)}/{len(filtered_matches)} matches!")

            # Match descriptors with ratio test
            matches = search_for_triangulation(q_frame, t_frame)
            if len(matches) < 5:
                continue # We need at least 5 matches to enforce the epipolar constraint
            log.info(f"\t Connected frame #{q_frame_id}: Found {len(matches)} potential new points!") 

            # Extract kpts
            q_kpts = np.array([q_frame.keypoints[m.queryIdx] for m in matches])
            t_kpts = np.array([t_frame.keypoints[m.trainIdx] for m in matches])

            # Enforce epipolar constraint
            q_kpt_pixels = np.float64([kpt.pt for kpt in q_kpts])
            t_kpt_pixels = np.float64([kpt.pt for kpt in t_kpts])
            ret = utils.enforce_epipolar_constraint(q_kpt_pixels, t_kpt_pixels)
            if ret is None: continue
            epi_mask, _, use_homography = ret
            log.info(f"\t Epipolar Constraint: Filtered {sum(~epi_mask)}/{len(q_kpts)} matches! (Using: {'Homography' if use_homography else 'Essential'}.)")
            epipolar_counter += np.sum(~epi_mask)
            if epi_mask.sum() == 0: continue
            matches = np.array(matches)[epi_mask]
            q_kpts = q_kpts[epi_mask]
            t_kpts = t_kpts[epi_mask]

            # Compute the transformation between the 2 frames
            T_q2t = utils.invert_transform(t_frame.pose) @ q_frame.pose

            # Triangulate the points
            q_kpt_pixels = np.float64([kpt.pt for kpt in q_kpts])
            t_kpt_pixels = np.float64([kpt.pt for kpt in t_kpts])
            q_points = utils.triangulate(q_kpt_pixels, t_kpt_pixels, T_q2t)
            t_points = utils.transform_points(q_points, T_q2t)  # (N, 3)
            if q_points is None or len(q_points) == 0: continue

            # Cheirality filter
            cheirality_mask = utils.filter_cheirality(q_points, t_points)
            cheirality_counter += np.sum(~cheirality_mask)
            if cheirality_mask is None or cheirality_mask.sum() == 0: continue
            log.info(f"\t Cheirality check filtered {sum(~cheirality_mask)}/{len(q_points)} points!")
            matches = matches[cheirality_mask]
            q_points = q_points[cheirality_mask]
            t_points = t_points[cheirality_mask]
            q_kpts = q_kpts[cheirality_mask]
            t_kpts = t_kpts[cheirality_mask]

            # Low parallax filter
            parallax_mask = utils.filter_parallax(q_points, t_points, T_q2t, MIN_PARALLAX)
            parallax_counter += np.sum(~parallax_mask)
            if parallax_mask is None or parallax_mask.sum() == 0: continue
            log.info(f"\t Parallax check filtered {sum(~parallax_mask)}/{len(q_points)} points!")
            matches = matches[parallax_mask]
            q_points = q_points[parallax_mask]
            t_points = t_points[parallax_mask]
            q_kpts = q_kpts[parallax_mask]
            t_kpts = t_kpts[parallax_mask]

            # Reprojection error filter
            t_kpt_pixels = np.float64([kpt.pt for kpt in t_kpts])
            reproj_mask, _ = utils.filter_by_reprojection(q_points, t_kpt_pixels, T_q2t, MAX_REPROJECTION)
            reprojection_counter += np.sum(~reproj_mask)
            if reproj_mask is None or reproj_mask.sum() == 0: continue
            log.info(f"\t Reprojection filtered: {sum(~reproj_mask)}/{len(q_points)} matches!")
            matches = matches[reproj_mask]
            q_points = q_points[reproj_mask]
            t_points = t_points[reproj_mask]
            q_kpts = q_kpts[reproj_mask]
            t_kpts = t_kpts[reproj_mask]

            # Scale consistency filter
            ## Compute the distances between the points and the camera frames
            q_dists = np.linalg.norm(q_points - q_frame.t, axis=1)
            t_dists = np.linalg.norm(t_points - t_frame.t, axis=1)
            dist_mask = np.logical_and(t_dists > 0, q_dists > 0)
            dist_counter += np.sum(~dist_mask)
            log.info(f"\t Dist filtered {np.sum(~dist_mask)}/{len(q_points)} points!")
            if dist_mask.sum() == 0: continue
            ratio_dists = t_dists / q_dists
            matches = matches[dist_mask]
            q_points = q_points[dist_mask]
            t_points = t_points[dist_mask]
            q_kpts = q_kpts[dist_mask]
            t_kpts = t_kpts[dist_mask]
            ## Compute the orb scale factors of every feature
            ## The scale factor is basically how big each feature is expected to be,
            ## based on the pyramid level that it is detected on
            q_scale_factors = np.array([q_frame.scale_factors[kpt.octave] for kpt in q_kpts])
            t_scale_factors = np.array([t_frame.scale_factors[kpt.octave] for kpt in t_kpts])
            ratio_octaves = t_scale_factors / q_scale_factors
            ## Make sure that the change in point size *somewhat* agrees
            ## with the predicted size change based on pyramid levels
            scale_mask = np.logical_and(ratio_octaves / ratio_factor < ratio_dists,
                                        ratio_dists < ratio_octaves * ratio_factor)
            scale_counter += np.sum(~scale_mask)
            log.info(f"\t Scale filtered {np.sum(~scale_mask)}/{len(q_points)} points!")
            if scale_mask.sum() == 0: continue
            matches = matches[scale_mask]
            q_points = q_points[scale_mask]
            t_points = t_points[scale_mask]

            # Depth filter
            q_median_depth = np.median(q_points[:, 2])
            t_median_depth = np.median(t_points[:, 2])
            depth_mask = np.logical_and(q_points[:, 2] < 5*q_median_depth, 
                                        t_points[:, 2] < 5*t_median_depth)
            depth_counter += np.sum(~depth_mask)
            log.info(f"\t Depth filtered {np.sum(~depth_mask)}/{len(q_points)} points!")
            if depth_mask.sum() == 0: continue
            q_points = q_points[depth_mask]
            matches = matches[depth_mask]

            # Transform the triangulated points to world coordinates
            w_points = utils.transform_points(q_points, q_frame.pose)
            num_created_points += len(w_points)

            # Gather all the matches
            all_matches[q_frame.id] = (matches, w_points)

        # Iterate over the matches with all the connected keyframes
        num_projection_matches = 0
        for q_frame_id, (matches, w_points) in all_matches.items():
            # Add the points to the map
            q_frame = ctx.map.keyframes[q_frame_id]
            points: np.ndarray[mapPoint] = self.add_new_points(w_points, matches, q_frame, t_frame)

            # The created points also need to be projected to the rest of the keyframes
            # and correspondences to be added if they are matched
            rest_kf_ids = neighbor_kf_ids - {q_frame_id}
            num_projection_matches += window_search(points, rest_kf_ids)
            
            if DEBUG:
                save_path = results_dir / "tracking/new_points" / f"{t_frame.id}_{q_frame.id}.png"
                vis.plot_matches(matches, q_frame, t_frame, save_path)

        log.info(f"\t Created {num_created_points} points! Filtered the following...")
        log.info(f"\t\t Epipolar: {epipolar_counter}")
        log.info(f"\t\t Cheirality: {cheirality_counter}")
        log.info(f"\t\t Reprojection: {reprojection_counter}")
        log.info(f"\t\t Parallax: {parallax_counter}")
        log.info(f"\t\t Distance: {dist_counter}")
        log.info(f"\t\t Scale: {scale_counter}")
        log.info(f"\t\t Depth: {depth_counter}")
        log.info(f"\t Total points: {self.num_points}!")
        
        log.info(f"\t The created points generated an additional {num_projection_matches} edges!")

    def create_points_parallel(self, t_frame: utils.Frame):
        """
        Parallelized version of create_points: each neighbor is processed
        in its own thread, results and filter‐counters are aggregated at the end.
        """
        neighbor_ids = ctx.cgraph.get_connected_frames(t_frame.id, num_edges=30)
        ratio_factor = 1.5 * t_frame.scale_factors[1]
        tasks = [(q_id, t_frame.id, ratio_factor) for q_id in neighbor_ids]

        all_new = []
        total_ctrs = {k:0 for k in [
            'epipolar','cheirality','reprojection',
            'parallax','distance','scale','depth'
        ]}

        # Create pool once per call
        with Pool(processes=os.cpu_count()) as pool:
            # map returns an iterator of (results, counters)
            for local_results, ctrs in pool.imap_unordered(process_neighbor, tasks, chunksize=1):
                all_new.extend(local_results)
                for key,val in ctrs.items():
                    total_ctrs[key] += val

        # Merge back on the main process
        new_points = {}
        for wpt_list, dist, q_idx, t_idx, qid in all_new:
            wpt    = np.array(wpt_list)
            q_frame= ctx.map.keyframes[qid]
            q_feat = q_frame.features[q_frame.keypoints[q_idx].class_id]
            t_feat = t_frame.features[t_frame.keypoints[t_idx].class_id]
            p = self._add_new_point(wpt, dist, t_frame, q_frame, t_feat, q_feat)
            if qid not in new_points.keys():
                new_points[qid] = [p]
            else:
                new_points[qid].append(p)

        # The created points also need to be projected to the rest of the keyframes
        # and correspondences to be added if they are matched
        num_projection_matches = 0
        for qid, points in new_points.items():
            rest_kf_ids = neighbor_ids - {qid}
            num_projection_matches += window_search(points, rest_kf_ids)
        
        log.info(f"\tCreated {len(all_new)} points! Filtered:")
        for k,v in total_ctrs.items():
            log.info(f"\t\t{k.capitalize()}: {v}")
        log.info(f"\t The created points generated an additional {num_projection_matches} edges!")

    def create_local_map(self, frame: utils.Frame):
        """
        Projects the map into a given frame and returns a local map.
        This local map contains: 
        - The frames that share map points with the current frame -> K1
        - The neighboring frames of K1 in the convisibility graph -> K2
        - A reference frame Kref in K1 which shares the most points with the current frame
        """
        log.info("[Graph] Creating local map...")
            
        frame_map_point_ids = frame.get_map_point_ids()

        # Find the frames that share map points and their points
        K1_frame_ids = set()
        K1_frame_counts = {}
        # Iterate over all the matched map points
        for pid in frame_map_point_ids:
            point = ctx.map.points[pid]
            # Iterate over all the point observations
            for obs in point.observations:
                # Keep the frame ids that are different than the current frame and exist in the graph
                frame_id = obs.kf_id
                if frame_id != frame.id and frame_id in ctx.cgraph.nodes.keys():
                    K1_frame_ids.add(frame_id)
                    # Increase the counter of the shared map points
                    if frame_id not in K1_frame_counts.keys():
                        K1_frame_counts[frame_id] = 1
                    else:
                        K1_frame_counts[frame_id] += 1

        # Find the points of the K1 frames
        K1_point_ids = set()
        for frame_id in K1_frame_ids:
            K1_point_ids.update(ctx.cgraph.nodes[frame_id])

        # Find neighboring frames to K1 and their points
        K2_frame_ids, K2_point_ids = ctx.cgraph.get_neighbor_frames_and_their_points(K1_frame_ids)

        # Find the frame(s) that shares the most map points
        max_shared_count = max(K1_frame_counts.values())
        ref_frame_ids = [k for k, v in K1_frame_counts.items() if v == max_shared_count]
        ref_frame_id = ref_frame_ids[0]

        # Create the local map
        local_map = localMap(ref_frame_id, K1_frame_ids, K1_point_ids, K2_frame_ids, K2_point_ids)

        return local_map


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
        """
        removed_pids = set()
        for pid, kf_id in matches:
            self.keyframes[kf_id].remove_matches_with(pid)
            num_obs = self.points[pid].remove_observation(kf_id)
            if num_obs == 0:
                removed_pids.add(pid)

        # self.remove_points(removed_pids)


    def remove_keyframe(self, kf_id: int):
        """
        Completely removes a keyframe:
            1) Removes the keyframe from the map
            2) Removes all the keyframe observations from the map points
            3) Removes the keyframe from the convisibility graph
        """
        del self.keyframes[kf_id]
        removed_pids = set()
        for p in self.points.values():
            num_obs = p.remove_observation(kf_id)
            if num_obs == 0:
                removed_pids.add(p.id)
        # self.remove_points(removed_pids)
        log.info(f"\t Removed Keyframe {kf_id}. {self.num_keyframes} left!")

    def remove_points(self, pids: set[int]):
        for pid in pids:
            del self.points[pid]
        for kf in self.keyframes.values():
            kf.remove_matches_with(pids)

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
        log.info("[Map] Culling map points...")

        removed_point_ids = set()

        # Iterate over all points
        r1 = 0
        r2 = 0
        r3 = 0
        for pid, p in self.points.items():
            
            # The tracking must find the point in more than 25% of the frames in which 
            # it is predicted to be visible.
            r = p.tracked_counter / p.visible_counter
            if r < MATCH_VIEW_RATIO:
                removed_point_ids.add(pid)
                r1 += 1
                continue

            # The point must always be observed during the first 2 keyframes
            num_kf_passed_since_creation = self._kf_counter - p.kf_number
            if (num_kf_passed_since_creation <= 2) and (p.num_observations-1 != num_kf_passed_since_creation):
                removed_point_ids.add(pid)
                r2 += 1
                continue
            
            # After that, it must always be observed by at least 3 keyframes
            elif num_kf_passed_since_creation >= 3 and p.num_observations < 3:
                removed_point_ids.add(pid)
                r3 += 1
                continue

        self.remove_points(removed_point_ids)
        ctx.cgraph.remove_points(removed_point_ids)

        log.info(f"\t Removed {len(removed_point_ids)} points. {self.num_points} left!")
        log.info(f"\t\t Ratio: {r1}")
        log.info(f"\t\t Creation: {r2}")
        log.info(f"\t\t Observations: {r3}")

    def cull_keyframes(self, frame: utils.Frame):
        """
        Discard all the connected keyframes whose 90% of the
        map points have been seen in at least other three keyframes in
        the same or finer scale.
        """
        log.info("[Map] Culling frames...")

        # Get the neighbor keyframes in the convisibility graph
        neighbor_kf_ids = ctx.cgraph.get_connected_frames(frame.id, num_edges=30)

        removed_kf_ids = set()
        # Iterate over all connected keyframes
        for kf_id in neighbor_kf_ids:
            # Extract their map points
            kf_points = ctx.cgraph.get_frame_points(kf_id)

            # Skip points with too few observations
            valid_pts = [p for p in kf_points if p.num_observations >= 4]
            num_valid = len(valid_pts)
            if num_valid == 0: continue
            
            # Count the number of points that are observed in at least 3 other keyframes
            count_coobserving_points = 0
            
            # Iterate over all their map points
            for point in kf_points:
                # Extract the neighbor frame observation
                obs = point.get_observation(kf_id)
                
                # Iterate over all other point observations
                count_point_observations = 0
                for other_obs in point.observations:
                    # Ignore observation from the same keyframe
                    if other_obs.kf_id == kf_id:
                        continue

                    # Check if the other observation is in the same or finer scale
                    if other_obs.kpt.octave <= obs.kpt.octave:
                        count_point_observations += 1

                    # Check if the point is observed at least 3 times
                    if count_point_observations >= 3:
                        count_coobserving_points += 1
                        break

            # Calculate the percentage of co-observing points
            ratio = count_coobserving_points / num_valid
            if ratio > KF_CULLING_RATIO:
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
        for feat in frame.features.values():
            if feat.in_map:
                point = feat.mp
                point.tracked_counter += 1


    def add_loop_closure(self, lc_frame: int, frame: int):
        self.loop_closures.append((lc_frame, frame))
        self.last_loop = (lc_frame, frame)

    @property
    def num_keyframes_since_last_loop(self) -> int:
        if len(self.loop_closures) == 0:
            return self._kf_counter
        else:
            last_lc_frame_id = self.last_loop[1].id
            return self._kf_counter - last_lc_frame_id
