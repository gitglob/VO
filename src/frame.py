import numpy as np
import open3d as o3d
from src.utils import depth_to_3d_points


class Frame():
    # This is a class-level (static) variable that all Frame instances share.
    _keypoint_id_counter = -1

    def __init__(self, id: int, img: np.ndarray, depth: np.ndarray, bow = None):
        self.id = id              # The frame id
        self.img = img.copy()     # The rgb image
        self.depth = depth.copy() # The depth image at that frame 
        self.bow = bow            # The bag of words of that image

        self.points = None
        self.pcd = None
        self.pcd_down = None
        self._init()

        self.is_keyframe = False  # Whether the frame is a keyframe
        self.pose = None          # The robot pose at that frame

    def set_keyframe(self, is_keyframe: bool):
        self.is_keyframe = is_keyframe

    def set_pose(self, pose: np.ndarray):
        self.pose = pose
    
    def _init(self, normals=True, voxel_size = 0.05):
        self.points = depth_to_3d_points(self.depth) 
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.points)

        # Downsample the pcd
        self.pcd_down = self.pcd.voxel_down_sample(voxel_size)
        if normals:
            self.pcd_down.estimate_normals()
