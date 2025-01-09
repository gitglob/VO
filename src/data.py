import os
import glob
import pandas as pd
import cv2
import numpy as np
from src.utils import quat2rot_matrix


# Function to find the closest, timewise, depth images to the rgb ones
def find_closest(base_timestamps, timestamps):
    return timestamps.iloc[(timestamps - base_timestamps).abs().argmin()]


class Dataset:
    def __init__(self, data_dir, scene, use_dist=False):
        self.data_dir = data_dir
        self.use_dist = use_dist
        self.scene = scene

        self._current_index = 0 # Image counter

        # Placeholders
        self._K = None

        self._read()
        self._init_calibration()

    def _read(self):
        images_dir = self.data_dir / self.scene / "image_2"  
        calib_txt = self.data_dir / self.scene / "calib.txt"
        times_txt = self.data_dir / self.scene / "times.txt"

        ground_truth_txt = self.data_dir / "data_odometry_poses" / "dataset" / "poses" / (self.scene + ".txt")
        
        self._image_paths = glob.glob(os.path.join(images_dir, "*.png"))
        self._image_paths.sort()
        self._times = np.loadtxt(times_txt)
        self._calib = pd.read_csv(calib_txt, delimiter=' ', header=None, index_col=0)
        self._ground_truth = pd.read_csv(ground_truth_txt, delimiter=' ', header=None)
        
    def _init_calibration(self):
        """Intrinsics matrix """
        P2 = np.array(self._calib.loc['P2:']).reshape((3,4))
        self._K, self._R, self._t, _, _, _, _ = cv2.decomposeProjectionMatrix(P2)

        fx = self._K[0, 0]
        fy = self._K[1, 1]
        cx = self._K[0, 2]
        cy = self._K[1, 2]

    def get(self):
        """ Returns the next RGB image in terms of timestamp """
        timestamp = self._times[self._current_index]
        image_path = self._image_paths[self._current_index]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        gt_pose = np.eye(4)
        gt_pose[:3, :4] = np.array(self._ground_truth.iloc[self._current_index]).reshape((3, 4))

        return timestamp, image, gt_pose

    def ground_truth(self):
        return self._ground_truth

    def finished(self):
        return self._current_index >= len(self._image_paths)
    
    def get_intrinsics(self):
        return self._K
       
    def length(self):
        return len(self._image_paths)
