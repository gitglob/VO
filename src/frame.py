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
        
        self.points: np.ndarray = None              # Placeholder for the triangulated 3D points that correspond to some matched keypoints between 2 frames
        self.matches: Dict[List[DMatch]] = {}       # Placeholder for the matches between this frame's keypoints and others'
        self.valid_kpt_indices: Dict[List[int]] = {}# Placeholder for the valid keypoint indices of the matches between this frame's keypoints and others'
        self.landmarks: np.ndarray = None           # Placeholder for the pixel coordinates of the tracked features (aka landmarks)

    def set_matches(self, with_frame_id: int, matches: List[DMatch]):
        """Sets matches with a specfic frame"""
        self.matches[with_frame_id] = np.array(matches, dtype=object)

    def set_valid_kpt_indices(self, with_frame_id: int, indices: List[int]):
        """Sets the indices of the keypoints that are kept during triangulation (and correspond to actual 3D points)"""
        self.valid_kpt_indices[with_frame_id] = indices

    def get_valid_matches(self, with_frame_id: int):
        """Get the valid indices. Valid indices are the ones that were actually used to generate 3D points during triangulation."""
        matches = self.matches[with_frame_id]
        valid_kpt_indices = self.valid_kpt_indices[with_frame_id]

        valid_matches = []
        for m in matches:
            if m.queryIdx in valid_kpt_indices:
                valid_matches.append(m)

        return valid_matches

    def set_pose(self, pose):
        self.pose = pose.copy()   # The robot pose at that frame

    def set_points(self, points: np.ndarray):
        self.points = points

    def get_valid_points(self):
        """Returns only non-None points (triangulated points)"""
        mask = np.array([x is not None for x in self.points])
        return self.points[mask]

    def set_landmark_indices(self, indices: List):
        """Sets the indices of the keypoints that are tracked over time (found in consecutive frames)"""
        self.landmark_indices = indices
        
        self.landmark_keypoints = [self.keypoints[i] for i in indices]
        self.landmark_descriptors = self.descriptors[indices]
        self.landmark_pixels = np.float64([self.keypoints[i].pt for i in indices])

        # Only the initialization frames contain triangulated points  
        if self.points is not None:
            self.landmark_points = self.points[indices]

    def update_landmark_indices(self, indices: List):
        """Sets the indices of the keypoints that are tracked over time (found in consecutive frames)"""
        self.landmark_indices = [self.landmark_indices[i] for i in indices]
        
        self.landmark_keypoints = [self.landmark_keypoints[i] for i in indices]
        self.landmark_descriptors = self.landmark_descriptors[indices]
        self.landmark_pixels = self.landmark_pixels[indices]

        # Only the initialization frames contain triangulated points  
        self.landmark_points = self.landmark_points[indices]
