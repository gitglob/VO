from typing import List, Tuple, Dict
import numpy as np
import cv2
from cv2 import DMatch
from src.others.visualize import plot_keypoints

from config import results_dir, SETTINGS


debug = SETTINGS["generic"]["debug"]
ORB_SETTINGS = SETTINGS["orb"]
W = SETTINGS["camera"]["width"]
H = SETTINGS["camera"]["height"]


class orbFeature():
    def __init__(self, kpt: cv2.KeyPoint, desc: np.ndarray):
        self.id = kpt.class_id
        self.kpt = kpt
        self.desc = desc
        self.matched = False

class Frame():
    # This is a class-level (static) variable that all Frame instances share.
    _keypoint_id_counter = -1

    def __init__(self, id: int, img: np.ndarray, is_initialized: bool):
        self.id: int = id                    # The frame id
        self.img: np.ndarray = img.copy()    # The BW image
        self.bow_hist = None                 # The histogram of bag of visual words of that image

        self.keypoints: Tuple                # The extracted ORB keypoints
        self.descriptors: np.ndarray         # The extracted ORB descriptors
        self.scale_factors: np.ndarray       # The per-octave scale factors
        self.pose: np.ndarray = None         # The camera -> world pose transformation matrix
        self.match: Dict = {}                # The matches between this frame's keypoints and others'
        self.relocalization: bool = False    # Whether global relocalization using vBoW was performed

        self.bow_hist: np.ndarray = None     # Histogram of visual words (1, vocab_size)
        self.feature_vector = {}             # Mapping: visual word -> keypoint id

        
        """
        The match dictionary looks like this:
        {
            frame_id: 
            {
                "matches": List[DMatch],  # The feature matches between the two frames
                "init_matches":
                "tracking_matches":
                "match_type": string,     # Whether the frame acted as query or train in the match
                "T": np.ndarray,          # The Transformation Matrix to get from the query frame (this frame) to the train frame (the one with frame_id)
            }
        }
        """


        # During initialization, we only extract features in the finest scale
        self.orb_n_levels = 1 if not is_initialized else ORB_SETTINGS["level_pyramid"]

        self._calc_scale_factors()  # Calculate the per octave scale factors
        self._extract_features()    # Extract ORB features from the image
        if debug:
            self.log_keypoints()

    @property
    def num_tracked_points(self):
        count = 0
        for f in self.features:
            if f.matched:
                count += 1
        return count
    
    @property
    def tracked_points(self):
        tracked_point_ids = set()
        for f in self.features:
            if f.matched:
                tracked_point_ids.add(f.id)
        return tracked_point_ids

    def _calc_scale_factors(self):
        self.scale_factors = np.ones(self.orb_n_levels)
        for i in range (1, len(self.scale_factors)):
            self.scale_factors[i] = self.scale_factors[i-1] * ORB_SETTINGS["scale_factor"]

    def set_keyframe(self, is_keyframe: bool):
        self.is_keyframe = is_keyframe

    def set_matches(self, with_frame_id: int, matches: List[DMatch], match_type: str):
        """Sets matches with another frame"""
        self.match[with_frame_id] = {}
        self.match[with_frame_id]["matches"] = np.array(matches, dtype=object)
        self.match[with_frame_id]["match_type"] = match_type
        self.match[with_frame_id]["T"] = None

    def get_matches(self, with_frame_id: int):
        """Returns matches with a specfic frame"""
        matches = self.match[with_frame_id]["matches"]
        return matches

    def get_init_matches(self, with_frame_id: int):
        """Returns matches with a specfic frame"""
        matches = self.match[with_frame_id]["init_matches"]
        return matches

    def get_tracking_matches(self, with_frame_id: int):
        """Returns matches with a specfic frame"""
        matches = self.match[with_frame_id]["tracking_matches"]
        return matches
    
    def set_pose(self, pose: np.ndarray):
        self.pose = pose
    
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
        
        self._detector = cv2.ORB_create(
            nfeatures=ORB_SETTINGS["num_keypoints"],
            scaleFactor=ORB_SETTINGS["scale_factor"],
            nlevels=self.orb_n_levels,
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
            print(f"Frame {self.id}: No BoW histogram computed!")
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

