import numpy as np
import cv2
from src.visualize import plot_keypoints
from config import results_dir


class Frame():
    def __init__(self, id: int, img: np.ndarray, depth: np.ndarray):
        self.id = id              # The frame id
        self.img = img.copy()     # The rgb image
        self.depth = depth.copy() # The depth image at that frame 

        self.is_keyframe = False  # Whether the frame is a keyframe
        self.pose = None          # The robot pose at that frame
        self.keypoints = None     # The detected keypoints
        self.descriptors = None   # The computed descriptors

        self._extract_features()

    def set_keyframe(self, is_keyframe: bool):
        self.is_keyframe = is_keyframe

    def set_pose(self, pose: np.ndarray):
        self.pose = pose.copy()   # The robot pose at that frame

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
        orb = cv2.ORB_create(nfeatures=10000)
        
        # Detect keypoints and compute descriptors
        kp, desc = orb.detectAndCompute(self.img, None)
        
        self.keypoints = kp
        self.descriptors = desc        

    ############################################# LOGGING #############################################

    def log_keypoints(self):
        print(f"\nframe #{self.id}")
        kpts_save_path = results_dir / "keypoints" / f"{self.id}_kpts.png"
        plot_keypoints(self.img, self.keypoints, kpts_save_path)
    