from typing import Tuple
import numpy as np
import cv2
# import src.local_mapping as mapping
import src.utils as utils
import src.visualization as vis
import src.globals as ctx

from config import results_dir, SETTINGS, log, fx, fy, cx, cy


ORB_SETTINGS = SETTINGS["orb"]
W = SETTINGS["camera"]["width"]
H = SETTINGS["camera"]["height"]
DEBUG = SETTINGS["generic"]["debug"]


class orbFeature():
    def __init__(self, kpt: cv2.KeyPoint, desc: np.ndarray, idx: int):
        self.kpt = kpt
        self.desc = desc
        self.id = kpt.class_id
        self.idx = idx

        self.mp = None
        self.mp_dist = None

    @property
    def in_map(self):
        return False if self.mp is None else True

    @property
    def matched(self):
        return False if self.mp is None else True

    def match_map_point(self, point, dist: np.float64):
        self.mp = point
        self.mp_dist = dist

    def reset_mp_match(self):
        self.mp = None
        self.mp_dist = None


class Frame():
    # This is a class-level (static) variable that all Frame instances share.
    _keypoint_id_counter = -1

    def __init__(self, id: int, img: np.ndarray):
        self.id: int = id                    # The frame id
        self.time: float = None              # The timestamp during that frame
        self.img: np.ndarray = img.copy()    # The BW image
        self.bow_hist = None                 # The histogram of bag of visual words of that image

        self.keypoints: Tuple                # The extracted ORB keypoints
        self.descriptors: np.ndarray         # The extracted ORB descriptors
        self.scale_factors: np.ndarray       # The per-octave scale factors
        self.scale_uncertainties: np.ndarray # The per-octave measurement uncertainties
        self.gt: np.ndarray = None           # The camera -> world Ground Truth pose
        self.pose: np.ndarray = None         # The camera -> world pose transformation matrix

        self.bow_hist: np.ndarray = None     # Histogram of visual words (1, vocab_size)
        self.features: dict = {}             # Mapping: keypoint_id -> ORB feature (keypoint, descriptor)
        self.feature_vector: dict = {}       # Mapping: visual word -> keypoint id

        self._extract_features()    # Extract ORB features from the image

    @property
    def num_tracked_points(self):
        count = 0
        for feat_id, f in self.features.items():
            if f.matched:
                count += 1
        return count
    
    @property
    def tracked_points(self):
        tracked_point_ids = set()
        for feat_id, f in self.features.items():
            if f.matched:
                tracked_point_ids.add(feat_id)
        return tracked_point_ids

    @property
    def R(self):
        return self.pose[:3, :3]
    
    @property
    def t(self):
        return self.pose[:3, 3]

    def set_time(self, t: float):
        self.time = t

    def set_gt(self, gt_pose: np.ndarray):
        self.gt = gt_pose

    def set_pose(self, pose: np.ndarray):
        self.pose = pose


    def remove_matches_with(self, pids: set[int] | int):
        """Removes matches with the given map points"""
        if isinstance(pids, int):
            for feat in self.features.values():
                if feat.matched:
                    if feat.mp.id == pids:
                        feat.reset_mp_match()
        else:
            for feat in self.features.values():
                if feat.matched:
                    if feat.mp.id in pids:
                        feat.reset_mp_match()


    def get_features_at_level(self, level: int) -> tuple[list, list]:
        """Returns the feature ids of a specific ORB scale level"""
        level_kpt_ids = []
        level_kpt_idxs = []
        for i, kpt in enumerate(self.keypoints):
            if kpt.octave == level:
                level_kpt_idxs.append(i)
                level_kpt_ids.append(kpt.class_id)

        return level_kpt_idxs, level_kpt_ids

    def get_features_in_area(self, u: float, v: float, r: float, min_level: int = -1, max_level: int = -1) -> list[orbFeature]:
        """
        Return the indices of keypoints whose undistorted positions lie within
        a square of half‐size r centered at (u, v).
        Optionally restrict to a pyramid level range [min_level, max_level].

        :param u:     x‐coordinate of the search center (pixels)
        :param v:     y‐coordinate of the search center (pixels)
        :param r:     half‐width of the search square (pixels)
        :param min_level: minimum pyramid octave (inclusive), or –1 to ignore
        :param max_level: maximum pyramid octave (inclusive), or –1 to ignore
        :return:       list of indices into self.mv_keys_un of all matching keypoints
        """
        # Decide whether to filter by pyramid level
        check_levels = not (min_level == -1 and max_level == -1)
        same_level = (min_level == max_level) and (min_level != -1)

        # Collect candidate current frame keypoints whose pixel coordinates fall within 
        # a window around the predicted pixel
        candidates = set()
        for kpt_id, feat in self.features.items():
            kpt = feat.kpt
            cand_u, cand_v = kpt.pt
            # Check octave levels if asked
            if check_levels:
                level = kpt.octave
                if same_level:
                    if level != min_level:
                        continue
                else:
                    if level < min_level or level > max_level:
                        continue
            # Check pixel region
            if (abs(cand_u - u) > r or abs(cand_v - v) > r):
                continue
            
            candidates.add(feat)

        return candidates

    def project(self, point: np.ndarray) -> tuple[float, float]:
        # transform into camera frame
        T_world2cam = utils.invert_transform(self.pose)
        R = T_world2cam[:3, :3]
        t = T_world2cam[:3, 3]
        point_cam = R @ point + t
        x, y, z = point_cam

        if z <= 0:
            return None

        # apply intrinsics
        u = fx * x / z + cx
        v = fy * y / z + cy
        
        if u < 0 or u >= W or v < 0 or v >= H:
            return None 
        
        return (u, v)

    def is_in_frustum(self, point):
        """Checks if a map point is inside this frame's frustum (cone of view)"""
        # 1) Projection check (positive depth and in image boundaries)
        px = self.project(point.pos)
        if px is None:
            return False
        u, v = px

        # If the point is already observed, we can do more checks
        if point.num_observations > 0:
            # 2) Viewing‐angle check
            # Compute the angle between the current viewing ray v
            # and the map point mean viewing direction n. Discard if v · n < cos(60◦).
            v1 = point.view_ray(self.t)
            v2 = point.mean_view_ray()
            if v1.dot(v2) < np.cos(np.deg2rad(60)):
                return False

            # 3) Distance‐based scale invariance check
            d = np.linalg.norm(point.pos - self.R)
            dmin, dmax = point.getScaleInvarianceLimits()
            if d < dmin or d > dmax:
                return False

            # 4) Compute the scale in the frame by the ration d/d_min
            scale = d / dmin
        else:
            scale = 1.0

        return u, v, scale

    def median_depth(self, map):
        """Returns the median depth of all the map points, projected in the camera frame"""
        depths = []
        for mp in map.points.values():
            point = mp.pos
            # transform into camera frame
            T_world2cam = utils.invert_transform(self.pose)
            R = T_world2cam[:3, :3]
            t = T_world2cam[:3, 3]
            point_cam = R @ point + t
            depths.append(point_cam[2])

        return np.median(depths)

    def is_keyframe(self):
        """
        New Keyframe conditions:
            1) More than X frames must have passed from the last global relocalization.
            2) Local mapping is idle, or more than X frames have passed from last keyframe insertion.
            3) Current frame tracks at least 50 points.
            4) Current frame tracks less than 90% points than Kref .
        """
        keyframes = ctx.map.keyframes
        local_map = ctx.local_map
        last_frame = list(keyframes.values())[-2]

        # dt = np.linalg.norm(self.t - last_frame.t)
        # c1 = dt > SETTINGS["keyframe"]["t"] # Translation change

        # dr = abs(utils.get_yaw(self.R) - utils.get_yaw(last_frame.R))
        # c2 = dr > SETTINGS["keyframe"]["r"] # Rotation change

        c3 = self.num_tracked_points > SETTINGS["keyframe"]["num_tracked_points"]

        ref_frame = keyframes[local_map.ref]
        A = ref_frame.tracked_points
        B = self.tracked_points
        common_features_ratio = len(A.intersection(B)) / len(A)
        c4 = common_features_ratio < SETTINGS["keyframe"]["common_feat_ratio"]
        
        is_keyframe = c3 and c4 #and (c1 or c2) 
        if DEBUG:
            if is_keyframe:
                log.info("\t\t Keyframe!")
            else:
                log.warning("\t\t Not a keyframe!")
                # if not c1:
                #     log.warning(f"\t\t Translation: {dt:.2f} < 2 !")
                # if not c2:
                #     log.warning(f"\t\t Rotation: {dr:.2f} < 5 !")
                if not c3:
                    log.warning(f"\t\t # of tracked points: {self.num_tracked_points} <= 50 !")
                if not c4:
                    log.warning(f"\t\t Common features ratio: {common_features_ratio} > 0.9 !")

        return is_keyframe


    def get_map_points_and_features(self):
        """Returns all the map points that are matched to a feature"""
        map_points = []
        mp_features = []
        for feat in self.features.values():
            if feat.matched:
                pid = feat.mp.id
                mp_features.append(feat)
                map_points.append(ctx.map.points[pid])
        return map_points, mp_features

    def get_map_point_ids(self):
        """Returns all the map points that are matched to a feature"""
        map_point_ids = set()
        for feat in self.features.values():
            if feat.in_map:
                map_point_ids.add(feat.mp.id)
        return map_point_ids

    def get_map_matches(self):
        """Returns all feature <-> map matches"""
        map_matches = set()
        for feat in self.features.values():
            if feat.in_map:
                map_matches.add((feat, feat.mp))
        return map_matches
    
    def get_map_matches_with(self, kf_id: int):
        """Returns all feature <-> map matches with a specific keyframe"""
        map_matches = set()
        for feat in self.features.values():
            if feat.in_map:
                point = feat.mp
                obs = point.get_observation(kf_id)
                if obs is not None:
                    map_matches.add((feat, point))
        return map_matches


    def optimize_pose(self, pose: np.ndarray):
        self.noopt_pose = self.pose.copy()
        self.pose = pose

    def _calc_scale_factors(self, levels: int):
        """Calculates the scale factors for each level in the ORB scale pyramid"""
        self.scale_factors = np.ones(levels)
        self.scale_uncertainties = np.ones(levels)
        for i in range (1, len(self.scale_factors)):
            self.scale_factors[i] = self.scale_factors[i-1] * ORB_SETTINGS["scale_factor"]
            self.scale_uncertainties[i] = self.scale_factors[i] * self.scale_factors[i]

    def _extract_features(self):
        """
        Extract image features using ORB and BoW histogram.
        
        keypoints: The detected keypoints. A 1-by-N structure array with the following fields:
            - pt: pixel coordinates of the keypoint [x,y]
            - size: diameter of the meaningful keypoint neighborhood
            - angle: computed orientation of the keypoint (-1 if not applicable); it's in [0,360) degrees and measured relative to image coordinate system (y-axis is directed downward), i.e in clockwise.
            - response: the response by which the most strong keypoints have been selected. Can be used for further sorting or subsampling.
            - octave: octave (pyramid layer) from which the keypoint has been extracted.
            - class_id: object class (if the keypoints need to be clustered by an object they belong to).
        descriptors: Computed descriptors. Descriptors are vectors that describe the image patch around each keypoint.
            Output concatenated vectors of descriptors. Each descriptor is a 32-element vector, as returned by cv.ORB.descriptorSize, 
            so the total size of descriptors will be numel(keypoints) * obj.descriptorSize(), i.e a matrix of size N-by-32 of class uint8, one row per keypoint.
        """

        # if self.id == 0:
        #     n_levels = 1
        #     log.warning(f"\t Extracting ORB features only at level {n_levels} for frame {self.id}!")
        # else:
        n_levels = ORB_SETTINGS["level_pyramid"]
        self._calc_scale_factors(n_levels)

        self._detector = cv2.ORB_create(
            nfeatures=ORB_SETTINGS["num_keypoints"],
            scaleFactor=ORB_SETTINGS["scale_factor"],
            nlevels=n_levels,
            edgeThreshold=ORB_SETTINGS["edge_threshold"],
            firstLevel=ORB_SETTINGS["first_level"],
            WTA_K=ORB_SETTINGS["WTA_K"],
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=ORB_SETTINGS["patch_size"],
            fastThreshold=ORB_SETTINGS["fast_threshold"]
        )
        
        # Detect keypoints and compute descriptors
        kpts, desc = self._detector.detectAndCompute(self.img, None)
        
        # Assign a unique class_id to each keypoint
        self.features = {}
        for i, k in enumerate(kpts):
            # Increment the class-level counter
            Frame._keypoint_id_counter += 1
            # Assign the keypoint's class_id
            k.class_id = Frame._keypoint_id_counter
            self.features[k.class_id] = orbFeature(k, desc[i], i)
        
        self.keypoints = kpts
        self.descriptors = desc
        
    def compute_bow(self):
        # Create the descriptor matcher
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Create the BoW extractor
        bow_extractor = cv2.BOWImgDescriptorExtractor(self._detector, matcher)
        bow_extractor.setVocabulary(ctx.vocab)

        # Compute the BoW histogram using the extractor
        # The histogram is typically a NumPy array of shape (1, vocab_size)
        self.bow_hist = bow_extractor.compute(self.img, self.keypoints)
        if self.bow_hist is None:
            log.warning(f"Frame {self.id}: No BoW histogram computed!")
            return
        
        ## This next step would normally not be needed if we used a SOTA vBoW
        ## extractor like DBoW2, but since we don't it's a way around

        # Initialize the feature vector with nothing
        distance_vector = {}
        for wid in range(len(ctx.vocab)): 
            self.feature_vector[wid] = []
            distance_vector[wid] = []

        # Build the feature vector (maps visual words -> keypoint indices)
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(self.descriptors, ctx.vocab)
        for match in matches:
            word_idx = match.trainIdx               # Index of the visual word in the vocabulary.
            q_idx = match.queryIdx                  # Index of the keypoint in the image.
            kp_id = self.keypoints[q_idx].class_id  # ID of the keypoint in the image.

            self.feature_vector[word_idx].append(kp_id)
            distance_vector[word_idx].append(match.distance)

        # Loop over each visual word (i.e., each bin in the histogram)
        # and add this frame's ID to the database for every visual word that occurs in the image.
        for visual_word in range(self.bow_hist.shape[1]):
            # Check if the histogram count for this visual word is greater than zero
            if self.bow_hist[0, visual_word] > 0:
                # Append the current frame's ID to the list for this visual word.
                ctx.bow_db[visual_word].add(self.id)

    def get_features_for_word(self, word_id: int) -> list[orbFeature]:
        # Check if the frame sees this word
        if len(self.feature_vector[word_id]) == 0:
            return None

        # Extract the feature from the frames
        kpt_ids = self.feature_vector[word_id]
        features = [self.features[k] for k in kpt_ids]
        
        return features

    ############################################# LOGGING #############################################

    def log_keypoints(self):
        kpts_save_path = results_dir / "keypoints" / f"{self.id}_kpts.png"
        vis.plot_keypoints(self.img, self.keypoints, kpts_save_path)

    def health_check(self):
        """Checks if all the points that are matched to a feature are also observed by the frame"""
        for feat in self.features.values():
            if feat.in_map:
                point = feat.mp
                obs_kf_ids = [obs.kf_id for obs in point.observations]
                assert len(obs_kf_ids) == len(set(obs_kf_ids))
                assert self.id in obs_kf_ids
                assert point.id in ctx.cgraph.get_frame_point_ids(self.id)