import pandas as pd
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


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
        self._combine_data()

    def _read(self, data_dir):        
        rgb_txt = data_dir / "rgb.txt"
        self._rgb = pd.read_csv(rgb_txt, comment='#', sep='\s+', header=None, names=["timestamp", "filename"])
        self._rgb['type'] = 'rgb'
        
        depth_txt = data_dir / "depth.txt"
        self._depth = pd.read_csv(depth_txt, comment='#', sep='\s+', header=None, names=["timestamp", "filename"])
        self._depth['type'] = 'depth'
        
        groundtruth_txt = data_dir / "groundtruth.txt"
        self._ground_truth = pd.read_csv(groundtruth_txt, comment='#', sep='\s+', header=None, names=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        
    def _init_calibration(self):
        """Intrinsics matrix, source: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect """
        if self.use_dist:
            fx = 520.9  # focal length x
            fy = 521.0  # focal length y
            cx = 325.1  # optical center x
            cy = 249.7  # optical center y
            
            # Distortion parameters
            d0 = 0.2312	
            d1 = -0.7849
            d2 = -0.0033
            d3 = -0.0001
            d4 = 0.9172
            self._dist_coeffs = np.array([d0, d1, d2, d3, d4], dtype=np.float64)
        else:
            fx = 525.0  # focal length x
            fy = 525.0  # focal length y
            cx = 319.5  # optical center x
            cy = 239.5  # optical center y

        self._K = np.array([[fx,  0, cx],
                            [ 0, fy, cy],
                            [ 0,  0,  1]], dtype=np.float64)

    def _combine_data(self):
        # Match the RGB data with the Depth data and Ground Truth
        rgb_copy = self._rgb.copy()
        
        rgb_copy["closest_depth"] = rgb_copy["timestamp"].apply(find_closest, timestamps=self._depth["timestamp"])
        rgb_copy = pd.merge(rgb_copy, self._depth, left_on="closest_depth", right_on="timestamp", suffixes=("_rgb", "_depth"))
        
        rgb_copy["closest_gt"] = rgb_copy["timestamp_rgb"].apply(find_closest, timestamps=self._ground_truth["timestamp"])
        rgb_copy = pd.merge(rgb_copy, self._ground_truth, left_on="closest_gt", right_on="timestamp", suffixes=("_rgb", "_gt"))
        
        # Instead of normalizing all poses now, we store the raw data.
        # Also, compute and store the first ground truth pose for normalization.
        t0 = np.array([rgb_copy.iloc[0]["tx"], rgb_copy.iloc[0]["ty"], rgb_copy.iloc[0]["tz"]])
        q0 = np.array([rgb_copy.iloc[0]["qx"], rgb_copy.iloc[0]["qy"], rgb_copy.iloc[0]["qz"], rgb_copy.iloc[0]["qw"]])
        r0 = R.from_quat(q0)  # initial rotation
        
        # Store the initial pose for later normalization in get()
        self._gt_offset = (t0, r0)
        
        # Save the combined dataframe without performing normalization now.
        self._data = rgb_copy.drop(columns=["closest_depth", "closest_gt", "type_rgb", "type_depth"])


    def get(self):
        """ Returns the next RGB image in terms of timestamp """
        row = self._data.iloc[self._current_index]
        timestamp = row["timestamp_rgb"]

        image_path = self.data_dir / row["filename_rgb"]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        depth_path = self.data_dir / row["filename_depth"]
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        # Get the raw ground truth pose from the row
        t = np.array([row["tx"], row["ty"], row["tz"]])
        q = np.array([row["qx"], row["qy"], row["qz"], row["qw"]])
        r_current = R.from_quat(q)

        # Retrieve the initial pose (t0, r0) stored earlier
        t0, r0 = self._gt_offset
        
        # Compute the relative (normalized) translation and rotation:
        # The new translation is the difference, rotated into the initial frame.
        t_rel = r0.inv().apply(t - t0)
        # The relative rotation is computed as the inverse of the initial rotation times the current rotation.
        r_rel = r0.inv() * r_current

        # Build the 4x4 ground truth transformation matrix.
        gt = np.eye(4)
        gt[:3, :3] = r_rel.as_matrix()
        gt[:3, 3] = t_rel

        self._current_index += 1

        return "rgbd", timestamp, image, depth, gt
        
    def frames(self, fraction=0.1):
        """ Returns x images from the dataset """
        # Sample the rgb dataframe
        interval = int(1 / fraction)
        sampled_rgb = self._rgb.iloc[::interval, :]

        # Extract the image filenames
        sampled_image_filenames = sampled_rgb["filename"].tolist()

        # Read all the images into a list        
        images = []
        for img_path in sampled_image_filenames:
            image = cv2.imread(self.data_dir / img_path, cv2.IMREAD_UNCHANGED)
            images.append(image)

        return images

    def ground_truth(self):
        return self._ground_truth

    def finished(self):
        return self._current_index >= len(self._data)
    
    def get_intrinsics(self):
        return self._K, self._dist_coeffs
       
    def length(self):
        return len(self._data)
