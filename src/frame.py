from typing import List
import numpy as np
import cv2
from src.frontend import extract_features


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
        self.id = id              # The frame id
        self.img = img.copy()     # The rgb image
        self.bow = bow            # The bag of words of that image
        self.keypoints = keypoints
        self.descriptors = descriptors
        self._landmarks = {}

    def set_pose(self, pose):
        self.pose = pose.copy()   # The robot pose at that frame

    # @property
    # def keypoints(self):
    #     if self._keypoints is None:
    #         self._extract_features()
    #     return self._keypoints

    # @property
    # def descriptors(self):
    #     if self._descriptors is None:
    #         self._extract_features()
    #     return self._descriptors

    # def _extract_features(self):
    #     """ Extracts the keypoints and descriptors of the associated image """
    #     self._keypoints, self._descriptors = extract_features(self.img) 

    def find_landmarks(self, matches, K, match_keypts, match_pose, query=True):
        """ Finds and initializes the frame landmarks """
        # Extract features from image
        self._extract_features()

        # Triangulate landmark_positions
        positions = self.triangulate_landmarks(matches, K, match_keypts, match_pose, query)
        
        # Initialize landmarks
        for i in range(len(positions)):
            # Create the new landmark
            landmark_id = matches[i].queryIdx if query else matches[i].trainIdx
            new_landmark = Landmark(positions[i], self.keypoints[i].pt, 
                                    self.descriptors, self.pose, 
                                    landmark_id, self.id)

            # Add it to the list
            self._landmarks[landmark_id] = new_landmark

    def triangulate_landmarks(self, matches: List, K: np.ndarray, 
                              prev_keypoints: List, 
                              match_pose: np.ndarray,
                              query=False) -> List[Landmark]:
        """
        Triangulates 3D landmarks from keypoints in two frames.

        Parameters:
            matches: The brute force matches between the previous and current descriptors
            K: Camera intrinsic matrix.
            curr_keypoints: List of keypoints from the current frame.
            prev_keypoints: List of keypoints from the previous frame.
            cur_pose: Pose of the current frame.
            match_pose: Pose of the previous frame.

        Returns:
            List: List of 3D landmarks (M, 3)
        """
        # Extract matched keypoints' pixel coordinates
        if query:
            prev_pts = np.float64([self.keypoints[m.queryIdx].pt for m in matches])
            curr_pts = np.float64([prev_keypoints[m.trainIdx].pt for m in matches])
        else:
            prev_pts = np.float64([self.keypoints[m.trainIdx].pt for m in matches])
            curr_pts = np.float64([prev_keypoints[m.queryIdx].pt for m in matches])

        # Triangulate points
        curr_pts_undistorted = cv2.undistortPoints(np.expand_dims(curr_pts, axis=1), K, None)
        prev_pts_undistorted = cv2.undistortPoints(np.expand_dims(prev_pts, axis=1), K, None)
        
        proj_matrix1 = np.hstack((self.pose[:3, :3], self.pose[:3, 3:]))
        proj_matrix2 = np.hstack((match_pose[:3, :3], match_pose[:3, 3:]))
        
        points_4d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, prev_pts_undistorted, curr_pts_undistorted)
        points_3d = points_4d[:3, :] / points_4d[3, :]

        # Initialize landmark objects
        landmark_positions = points_3d.T

        return landmark_positions

    def get_landmark(self, landmark_id):
        """ Returns a specific landmark """
        return self._landmarks[landmark_id]

    def landmarks(self):
        """ Return all the landmarks """
        return list(self._landmarks.values())