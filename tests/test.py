import numpy as np
import cv2
from pathlib import Path
import glob
import os

# Dataset
main_dir = Path(__file__).parent
data_dir = Path.home() / "Documents" / "data" / "kitti"
scene = "06"
image_dir = data_dir / scene / "image_0"
image_paths = glob.glob(os.path.join(image_dir, "*.png"))
ground_truth_txt = data_dir / "data_odometry_poses" / "dataset" / "poses" / (scene + ".txt")

# Camera intrinsic matrix
fx, fy = 707.0912, 707.0912
cx, cy = 601.8873, 183.1104
K = np.array([[fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]])

def load_ground_truth(gt_file):
    """Load ground truth poses from a text file.
       Each line has 12 numbers representing a 3x4 transformation matrix.
    """
    gt_poses = []
    with open(gt_file, 'r') as f:
        for line in f:
            nums = list(map(float, line.split()))
            # Reshape the first 12 numbers into a 3x4 matrix.
            pose_3x4 = np.array(nums).reshape(3, 4)
            # Create a 4x4 homogeneous transformation matrix.
            T = np.eye(4)
            T[:3, :3] = pose_3x4[:, :3]
            T[:3, 3] = pose_3x4[:, 3]
            gt_poses.append(T)
    return gt_poses

def test_vo():
    # Load two consecutive frames (ensure they are in grayscale)
    img1 = cv2.imread(image_dir / '000000.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_dir / '000001.png', cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise IOError("Error loading images. Check the paths.")

    # Initialize ORB detector with a maximum number of features
    orb = cv2.ORB_create(nfeatures=1000)

    # Detect keypoints and compute descriptors for each image
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match descriptors using brute force matching with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance so that best matches come first
    matches = sorted(matches, key=lambda x: x.distance)

    # Optionally, select only the top N matches for a more robust estimate
    num_matches = 100  # You may adjust this number
    good_matches = matches[:num_matches]

    # Extract matched keypoint coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Compute the Essential Matrix using RANSAC to robustly handle outliers
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Recover the relative camera pose (rotation R and translation t)
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    print("Rotation Matrix:", R.round(2))
    print("\nTranslation Vector:", t.round(2))

if __name__ == '__main__':
    test_vo()
