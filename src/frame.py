from typing import List, Tuple, Dict
import numpy as np
import cv2
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
        
        self.points: np.ndarray = None              # Placeholder for the triangulated 3D points that correspond to some matched keypoints
        self.matches: Dict[List[DMatch]] = {}       # Placeholder for the matches between this frame's keypoints and others'
        self.landmarks: List = []                   # Placeholder for the pixel coordinates of the tracked features (aka landmarks)

    def set_matches(self, with_frame_id: int, matches: List[DMatch]):
        """Sets matches with a specfic frame"""
        self.matches[with_frame_id] = np.array(matches, dtype=object)

    def get_filtered_matches(self, frame_id):
        matches = self.matches[frame_id]
        filtered_matches = matches[self.inlier_mask]

        return filtered_matches

    def set_pose(self, pose):
        self.pose = pose.copy()   # The robot pose at that frame

    def set_points(self, points: np.ndarray):
        self.points = points

    def set_inlier_mask(self, inlier_mask):
        self.inlier_mask = inlier_mask

    def set_landmark_indices(self, indices):
        """Sets the indices of the keypoints that are tracked over time (found in consecutive frames)"""
        self.landmark_indices = indices
        self.landmarks = np.float64([self.keypoints[i].pt for i in self.landmark_indices])
        self.landmark_descriptors = self.descriptors[self.landmark_indices]
