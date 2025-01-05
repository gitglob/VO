from typing import List, Tuple, Dict
import numpy as np
from cv2 import DMatch


class Landmark():
    def __init__(self, 
                 position: np.ndarray, 
                 keypoint: np.ndarray, 
                 descriptors: np.ndarray, 
                 pose: np.ndarray,
                 landmark_id: int, 
                 frame_id: int):
        """
        Initializes a landmark
        
        Args:
            position: The xyz position of the landmark in the world frame (3,)
            keypoint: The u,v pixell coordinates of the landmark in the image plane
            descriptors: The associated frame descriptors (N, 32) 
            id: The landmark id 
            frame_id: The associated frame id
        """
        self.position = position
        self.pixel_coords = keypoint
        self.descriptors = descriptors
        self.pose = pose
        self.frame_id = frame_id
        self.id = landmark_id
        self.unique_id = 'f' + str(frame_id) + 'l' + str(landmark_id) # Unique landmark identifier

class Frame():
    def __init__(self, id: int, img: np.ndarray, keypoints, descriptors, bow = None):
        self.id: int = id                           # The frame id
        self.img: np.ndarray = img.copy()           # The rgb image
        self.bow = bow                              # The bag of words of that image

        self.keypoints: Tuple = keypoints           # The extracted ORB keypoints
        self.descriptors: np.ndarray = descriptors  # The extracted ORB descriptors

        self.landmarks_inditialized: bool = False   # Placeholder for whether landmarks have been initialized at this frame
                                                    # Landmarks are simply features that are chosen to be tracked across multiple consecutive frames
        
        self.points: np.ndarray = None              # Placeholder for the triangulated 3D points that correspond to some matched keypoints between 2 frames
        self.matches: Dict[List[DMatch]] = {}       # Placeholder for the matches between this frame's keypoints and others'
        self.triangulation_indices: Dict[List[int]] = {} # Placeholder for the triangulated keypoint indices between this frame and others

    def set_matches(self, with_frame_id: int, matches: List[DMatch]):
        """Sets matches with a specfic frame"""
        self.matches[with_frame_id] = np.array(matches, dtype=object)

    def set_triangulation_indices(self, with_frame_id: int, indices: List[int]):
        """Sets the indices of the keypoints that are kept during triangulation (and correspond to actual 3D points)"""
        self.triangulation_indices[with_frame_id] = indices

    def get_triangulation_matches(self, with_frame_id: int):
        """Get the valid indices. Valid indices are the ones that were actually used to generate 3D points during triangulation."""
        matches = self.matches[with_frame_id]
        triangulation_indices = self.triangulation_indices[with_frame_id]

        valid_matches = []
        for m in matches:
            if m.queryIdx in triangulation_indices:
                valid_matches.append(m)

        return valid_matches

    def set_pose(self, pose):
        self.pose = pose.copy()   # The robot pose at that frame

    def set_points(self, points: np.ndarray):
        print(f"\t\tSetting 3D points in frame #{self.id}")
        self.points = points

    def get_valid_points(self):
        """Returns only non-None points (triangulated points)"""
        mask = np.array([x is not None for x in self.points])
        return self.points[mask]

    def set_landmark_indices(self, indices: List):
        """Sets the indices of the keypoints that are tracked over time (found in consecutive frames)"""
        print(f"\t\tSetting landmarks in frame #{self.id}")
        self.landmark_indices = indices
        self.landmarks_initialized = True
        
    @property
    def landmark_points(self):
        if not self.landmarks_initialized:
            raise("No landmarks found!!")
        return self.points[self.landmark_indices]

    @property
    def landmark_keypoints(self):
        if not self.landmarks_initialized:
            raise("No landmarks found!!")
        return [self.keypoints[i] for i in self.landmark_indices]

    @property
    def landmark_descriptors(self):
        if not self.landmarks_initialized:
            raise("No landmarks found!!")
        return self.descriptors[self.landmark_indices]

    @property
    def landmark_pixels(self):
        if not self.landmarks_initialized:
            raise("No landmarks found!!")
        return np.float64([self.keypoints[i].pt for i in self.landmark_indices])
        
    def landmark_matches(self, with_frame_id: int):
        """Returns the matches that correspond to landmarks"""
        if not self.landmarks_initialized:
            raise("No landmarks found!!")
        matches = []
        for m in self.matches[with_frame_id]:
            if m.queryIdx in self.landmark_indices:
                matches.append(m)
        return matches