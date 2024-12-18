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
        
        self.matches_as_prev: List[DMatch] = None   # The keypoints that matched the previous keyframe
        self.matches_as_curr: List[DMatch] = None   # The keypoints that matched the next keyframe
        
        self.points: np.ndarray = None              # Placeholder for the triangulated 3D points that correspond to some matched keypoints
        self.matches: Dict[List[DMatch]] = {}       # Placeholder for the matches between this frame's keypoints and others'
        self._landmarks = {}                        # The landmarks

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

    def find_landmarks(self, matches, K, match_keypts, match_pose, query=True):
        """ Finds and initializes the frame landmarks """
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