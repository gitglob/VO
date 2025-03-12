from typing import List, Dict
import numpy as np
import cv2
from cv2 import DMatch
from src.visualize import plot_keypoints
from config import results_dir


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
    # This is a class-level (static) variable that all Frame instances share.
    _keypoint_id_counter = -1

    def __init__(self, id: int, img: np.ndarray, depth: np.ndarray, bow = None):
        self.id = id              # The frame id
        self.img = img.copy()     # The rgb image
        self.depth = depth.copy() # The depth image at that frame 
        self.bow = bow            # The bag of words of that image

        self.is_keyframe = False  # Whether the frame is a keyframe
        self.pose = None          # The robot pose at that frame
        self.keypoints = None     # The detected keypoints
        self.descriptors = None   # The computed descriptors
        
        self.match: Dict = {}     # The matches between this frame's keypoints and others'
        """
        The match dictionary looks like this:
        {
            frame_id: 
            {
                "matches": List[DMatch],     # The feature matches between the two frames
                "match_type": string,        # Whether the frame acted as query or train in the match

                "initialization": bool,      # Whether this frame was used to initialize the pose
                "use_homography": bool,      # Whether the homography/essential matrix was used to initialize the pose
                
                "pose": np.ndarray,          # The Transformation Matrix between the 2 frames
                "points": np.ndarray,        # The triangulated keypoint points
                "point_ids": np.ndarray,     # The triangulated keypoint identifiers
                
                "feature_mask": List[bool],     # Which keypoint/descriptors were used in the match 
                "triangulation_mask": List[int] # Which keypoint/descriptors were triangulated in the match
                
                "inlier_match_mask": List[int],       # Which matches were kept after Essential/Homography filtering in this match
                "triangulation_match_mask": List[int] # Which matches were triangulated in this match

                "stage": string              # The stage of the match (initialization, tracking)
            }
        }
        """

        self._extract_features()
        self._landmarks = {}

    def set_keyframe(self, is_keyframe: bool):
        self.is_keyframe = is_keyframe

    def set_pose(self, pose):
        self.pose = pose.copy()   # The robot pose at that frame

    def set_matches(self, with_frame_id: int, matches: List[DMatch], match_mask: np.ndarray, match_type: str):
        """Sets matches with another frame"""
        self.match[with_frame_id] = {}
        self.match[with_frame_id]["matches"] = np.array(matches, dtype=object)
        self.match[with_frame_id]["match_type"] = match_type
        self.match[with_frame_id]["match_mask"] = match_mask

        # Default values for the rest
        self.match[with_frame_id]["initialization"] = None
        self.match[with_frame_id]["use_homography"] = None
        self.match[with_frame_id]["inlier_match_mask"] = None
        self.match[with_frame_id]["pose"] = None
        self.match[with_frame_id]["points"] = None

    def get_matches(self, with_frame_id: int):
        """Returns matches with a specfic frame"""
        return self.match[with_frame_id]["matches"]

    def set_pose(self, pose: np.ndarray):
        self.pose = pose
    
    def _extract_features(self):
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
        orb = cv2.ORB_create(nfeatures=5000)
        
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
        print(f"\nframe #{self.id}")
        kpts_save_path = results_dir / "keypoints" / f"{self.id}_kpts.png"
        plot_keypoints(self.img, self.keypoints, kpts_save_path)

    ############################################# Landmarks #############################################

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
            curr_pts = np.float64([self.keypoints[m.queryIdx].pt for m in matches])
            prev_pts = np.float64([prev_keypoints[m.trainIdx].pt for m in matches])
        else:
            curr_pts = np.float64([self.keypoints[m.trainIdx].pt for m in matches])
            prev_pts = np.float64([prev_keypoints[m.queryIdx].pt for m in matches])

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
    