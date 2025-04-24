from typing import List, Tuple, Dict
import numpy as np
import cv2
from cv2 import DMatch
from src.others.linalg import invert_transform
from src.others.visualize import plot_keypoints

from config import results_dir, SETTINGS, log, fx, fy, cx, cy


debug = SETTINGS["generic"]["debug"]
ORB_SETTINGS = SETTINGS["orb"]
W = SETTINGS["camera"]["width"]
H = SETTINGS["camera"]["height"]


class orbFeature():
    def __init__(self, kpt: cv2.KeyPoint, desc: np.ndarray):
        self.kpt = kpt
        self.desc = desc
        self.id = kpt.class_id

        self.mp = None
        self.matched = False

    def match_map_point(self, pid: int, dist: np.float64):
        self.mp = {
            "id": pid,
            "dist": dist
        }
        self.matched = True

    def reset_mp_match(self):
        self.mp = {}
        self.matched = False


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
        self.relocalization: bool = False    # Whether global relocalization using vBoW was performed

        self.bow_hist: np.ndarray = None     # Histogram of visual words (1, vocab_size)
        self.features: dict = {}             # Mapping: keypoint_id -> ORB feature (keypoint, descriptor)
        self.feature_vector: dict = {}       # Mapping: visual word -> keypoint id

        self._extract_features()    # Extract ORB features from the image
        if debug:
            self.log_keypoints()

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

    def set_time(self, t: float):
        self.time = t

    def set_gt(self, gt_pose: np.ndarray):
        self.gt = gt_pose

    def set_pose(self, pose: np.ndarray):
        self.pose = pose


    def reset_pose(self):
        self.pose = None

    def remove_mp_match(self, pid: int):
        """Removes matches with the given map point"""
        for feat in self.features.values():
            if feat.matched:
                if feat.mp["id"] == pid:
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

    def project(self, point: np.ndarray) -> tuple[float, float]:
        # transform into camera frame
        T_world2cam = invert_transform(self.pose)
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

    def is_in_frustum(self, point, keyframes):
        """Checks if a map point is inside this frame's frustum (cone of view)"""
        # 1) Projection check (positive depth and in image boundaries)
        px = self.project(point.pos)
        if px is None:
            return False
        u, v = px

        # 2) Viewing‐angle check
        # Compute the angle between the current viewing ray v
        # and the map point mean viewing direction n. Discard if v · n < cos(60◦).
        v1 = point.view_ray(self.pose[:3, 3])
        v2 = point.mean_view_ray(keyframes)
        if v1.dot(v2) < np.cos(np.deg2rad(60)):
            return False

        # 3) Distance‐based scale invariance check
        d = np.linalg.norm(point.pos - self.pose[:3, 3])
        dmin, dmax = point.getScaleInvarianceLimits(keyframes)
        if d < dmin or d > dmax:
            return False

        # 4) Compute the scale in the frame by the ration d/d_min
        scale = d / dmin

        return u, v, scale

    def get_map_point_ids(self):
        """Returns all the map points that are matched to a feature"""
        map_point_ids = set()
        for feat in self.features.values():
            if feat.matched:
                map_point_ids.add(feat.mp["id"])
        return map_point_ids

    def get_map_matches(self) -> set[tuple[int, int]]:
        """Returns all feature <-> map matches"""
        map_matches = set()
        for feat in self.features.values():
            if feat.matched:
                map_matches.add((feat.id, feat.mp["id"]))
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

        if self.id == 0:
            n_levels = 1
            log.warning(f"\t Extracting ORB features only at level {n_levels} for frame {self.id}!")
        else:
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
            self.features[k.class_id] = orbFeature(k, desc[i])
        
        self.keypoints = kpts
        self.descriptors = desc
        
    def compute_bow(self, vocab, bow_db: list[dict]):
        # Create the descriptor matcher
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Create the BoW extractor
        bow_extractor = cv2.BOWImgDescriptorExtractor(self._detector, matcher)
        bow_extractor.setVocabulary(vocab)

        # Compute the BoW histogram using the extractor
        # The histogram is typically a NumPy array of shape (1, vocab_size)
        self.bow_hist = bow_extractor.compute(self.img, self.keypoints)
        if self.bow_hist is None:
            log.warning(f"Frame {self.id}: No BoW histogram computed!")
            return
        
        ## This next step would normally not be needed if we used a SOTA vBoW
        ## extractor like DBoW2, but since we don't it's a way around

        # Build the feature vector (maps visual words -> keypoint indices)
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(self.descriptors, vocab)
        self.feature_vector = {}
        for match in matches:
            word_id = match.trainIdx         # Index of the visual word in the vocabulary.
            kp_idx = match.queryIdx          # Index of the keypoint in the image.
            kp_id = self.features[kp_idx].id # ID of the keypoint in the image.
            if word_id not in self.feature_vector:
                self.feature_vector[word_id] = []
            self.feature_vector[word_id].append(kp_id)

        # Loop over each visual word (i.e., each bin in the histogram)
        # and add this frame's ID to the database for every visual word that occurs in the image.
        for visual_word in range(self.bow_hist.shape[1]):
            # Check if the histogram count for this visual word is greater than zero
            if self.bow_hist[0, visual_word] > 0:
                # Append the current frame's ID to the list for this visual word.
                bow_db[visual_word].append(self.id)

        self.relocalization = True

    ############################################# LOGGING #############################################

    def log_keypoints(self):
        kpts_save_path = results_dir / "keypoints" / f"{self.id}_kpts.png"
        plot_keypoints(self.img, self.keypoints, kpts_save_path)

