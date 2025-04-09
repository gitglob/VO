from typing import List, Tuple, Dict
import numpy as np
import cv2
from cv2 import DMatch
from src.others.visualize import plot_keypoints
from src.others.local_map import Map

from config import results_dir, SETTINGS


debug = SETTINGS["generic"]["debug"]


class Frame():
    # This is a class-level (static) variable that all Frame instances share.
    _keypoint_id_counter = -1

    def __init__(self, id: int, img: np.ndarray, is_initialized: bool, bow=None):
        self.id: int = id                    # The frame id
        self.img: np.ndarray = img.copy()    # The BW image
        self.bow = bow                       # The bag of words of that image

        self.keypoints: Tuple                # The extracted ORB keypoints
        self.descriptors: np.ndarray         # The extracted ORB descriptors
        self._extract_features(is_initialized)  # Extract ORB features from the image

        self.pose: np.ndarray = None         # The world -> camera pose transformation matrix
        
        self.match: Dict = {}                # The matches between this frame's keypoints and others'
        """
        The match dictionary looks like this:
        {
            frame_id: 
            {
                "matches": List[DMatch],     # The feature matches between the two frames
                "init_matches":
                "tracking_matches":

                "match_type": string,        # Whether the frame acted as query or train in the match
                "initialization": bool,      # Whether this frame was used to initialize the pose
                "use_homography": bool,      # Whether the homography/essential matrix was used to initialize the pose
                
                "T": np.ndarray,          # The Transformation Matrix to get from the query frame (this frame) to the train frame (the one with frame_id)
                "points": np.ndarray,        # The triangulated keypoint points
            }
        }
        """

        if debug:
            self.log_keypoints()

    def keypoints_in_map(self, map: Map) -> set:
        """Returns the keypoints that are in the map as a set."""
        kpts_in_map = set()
        for kpt in self.keypoints:
            if kpt.class_id in map.point_ids():
                kpts_in_map.add(kpt.class_id)
        
        return kpts_in_map

    def set_keyframe(self, is_keyframe: bool):
        self.is_keyframe = is_keyframe

    def set_matches(self, with_frame_id: int, matches: List[DMatch], match_type: str):
        """Sets matches with another frame"""
        self.match[with_frame_id] = {}
        self.match[with_frame_id]["matches"] = np.array(matches, dtype=object)
        self.match[with_frame_id]["match_type"] = match_type

        # Default values for the rest
        self.match[with_frame_id]["initialization"] = None
        self.match[with_frame_id]["use_homography"] = None
        self.match[with_frame_id]["T"] = None
        self.match[with_frame_id]["points"] = None

    def initialize(self, with_frame_id: int, use_homography: bool, pose: np.ndarray):
        """
        Initializes the frame with another frame.
        """
        self.match[with_frame_id]["use_homography"] = use_homography
        self.match[with_frame_id]["T"] = pose
        self.match[with_frame_id]["initialization"] = True

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
    
    def _extract_features(self, is_initialized):
        """
        Extract image features using ORB.
        
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
        # Initialize the ORB detector
        orb_settings = SETTINGS["orb"]

        # During initialization, we only extract features in the finest scale
        levels = 1 if not is_initialized else orb_settings["level_pyramid"]
        
        orb = cv2.ORB_create(
            nfeatures=orb_settings["num_keypoints"],
            scaleFactor=orb_settings["scale_factor"],
            nlevels=levels,
            edgeThreshold=orb_settings["edge_threshold"],
            firstLevel=orb_settings["first_level"],
            WTA_K=orb_settings["WTA_K"],
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=orb_settings["patch_size"],
            fastThreshold=orb_settings["fast_threshold"]
        )
        
        # Detect keypoints and compute descriptors
        kp, desc = orb.detectAndCompute(self.img, None)
        
        # Assign a unique class_id to each keypoint
        for k in kp:
            # Increment the class-level counter
            Frame._keypoint_id_counter += 1
            # Assign the keypoint's class_id
            k.class_id = Frame._keypoint_id_counter
        
        self.keypoints = kp
        self.descriptors = desc        

    ############################################# LOGGING #############################################

    def log_keypoints(self):
        kpts_save_path = results_dir / "keypoints" / f"{self.id}_kpts.png"
        plot_keypoints(self.img, self.keypoints, kpts_save_path)

