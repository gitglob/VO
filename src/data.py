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
    def __init__(self, data_dir, use_dist=False):
        self.data_dir = data_dir
        self.use_dist = use_dist

        self._current_index = 0 # Image counter

        # Placeholders
        self._K = None
        self._dist_coeffs = None

        self._read(data_dir)
        self._init_calibration()

    def _read(self, data_dir):      
        images_dir = data_dir / "images"  
        ground_truth_txt = data_dir / "groundtruthSync.txt"
        pcalib_txt = data_dir / "pcalib.txt"
        times_txt = data_dir / "times.txt"
        self.camera_txt = data_dir / "camera.txt"
        self.statistics_txt = data_dir / "statistics.txt"

        self._image_paths = glob.glob(os.path.join(images_dir, "*.jpg"))
        self._image_paths.sort()
        self._times = np.loadtxt(times_txt)[:,1]
        self._pcalib = np.loadtxt(pcalib_txt)
        self._ground_truth = pd.read_csv(ground_truth_txt, comment='#', sep='\s+', header=None, 
                                         names=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        
    def _init_calibration(self):
        """Intrinsics matrix, source: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect """
        fx = 0.535719308086809  # focal length x
        fy = 0.669566858850269  # focal length y
        cx = 0.493248545285398  # optical center x
        cy = 0.500408664348414  # optical center y
        omega = 0.897966326944875

        self._K = np.array([[fx,  0, cx],
                            [ 0, fy, cy],
                            [ 0,  0,  1]], dtype=np.float64)

    def get(self):
        """ Returns the next RGB image in terms of timestamp """
        timestamp = self._times[self._current_index]
        image_path = self._image_paths[self._current_index]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        gt_row = self._ground_truth.iloc[self._current_index]
        if not np.isclose(timestamp, gt_row["timestamp"], atol=1e-4):
            raise ValueError(f"Image and G.T. timestamps are different!! \n{timestamp} vs {gt_row['timestamp']}")

        t = [gt_row["tx"], gt_row["ty"], gt_row["tz"]]
        R = quat2rot_matrix(gt_row["qx"], gt_row["qy"], gt_row["qz"], gt_row["qw"])
        gt = np.eye(4)
        gt[:3, :3] = R
        gt[:3, 3] = t

        self._current_index += 1

        return timestamp, image, gt

    def ground_truth(self):
        return self._ground_truth

    def finished(self):
        return self._current_index >= len(self._image_paths)
    
    def get_intrinsics(self):
        return self._K, self._dist_coeffs
       
    def length(self):
        return len(self._image_paths)
