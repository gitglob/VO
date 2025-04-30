import numpy as np
import cv2
from src.utils.linalg import invert_transform
from src.utils.linalg import skew_symmetric
from config import K, log


def print_mean_reprojection_error(q_kpts, t_kpts, q_frame, t_frame):
    """Prints the re-projection error between 2 frames with poses at specific keypoints"""
    q_pxs = np.array([kpt.pt for kpt in q_kpts], dtype=np.float64)
    t_pxs = np.array([kpt.pt for kpt in t_kpts], dtype=np.float64)
    T_q2t = invert_transform(t_frame.pose @ invert_transform(q_frame.pose))

    e = reprojection_error(q_pxs, t_pxs, T_q2t)
    log.info(f"Mean Reprojection Error: {np.mean(e):.2f}")

def reprojection_error(pxs1, pxs2, T):
    """
    Triangulate inlier correspondences, reproject them into the current frame, and filter matches by reprojection error.

    Args:
        pxs1 (N, 2): Pixel coordinates in frame 1.
        pxs2 (N, 2): Pixel coordinates in frame 2.
        T    (4, 4): relative pose from from O1 to O2.

    Returns:
        np.array: Updated boolean mask with matches having large reprojection errors filtered out.
    """
    R = T[:3, :3]
    t = T[:3, 3].reshape(-1,1)
    
    # Projection matrices
    M1 = K @ np.eye(3,4)        # Reference frame (identity)
    M2 = K @ np.hstack((R, t))  # Current frame

    # Triangulate points
    pts1_4d = cv2.triangulatePoints(M1, M2, pxs1.T, pxs2.T)
    pts1_3d = (pts1_4d[:3] / pts1_4d[3]).T

    # Reproject points into the second (current) camera
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    tvec = T[:3, 3] 
    pts1_proj2, _ = cv2.projectPoints(pts1_3d, rvec, tvec, K, None)
    pts1_proj2_px = pts1_proj2.reshape(-1, 2)

    # Compute reprojection errors
    errors = np.linalg.norm(pts1_proj2_px - pxs2, axis=1)

    return errors

def triang_points_reprojection_error(pts1_3d, pxs2, T):
    """
    Reproject triangulated points into the current frame and return reprojection error.

    Args:
        pts1_3d (N, 3): 3d points in frame 1.
        pxs2 (N, 2): Pixel coordinates in frame 2.
        T    (4, 4): relative pose from from O1 to O2.
    """
    # Reproject points into the second (current) camera
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    tvec = T[:3, 3] 
    pts1_proj2, _ = cv2.projectPoints(pts1_3d, rvec, tvec, K, None)
    pts1_proj2_px = pts1_proj2.reshape(-1, 2)

    # Compute reprojection errors
    errors = np.linalg.norm(pts1_proj2_px - pxs2, axis=1)

    return errors

def triangulation_angles(points1, points2, T):
    """Calculates the triangulation angles between 2 or more points given the transformation matrix of their origins."""
    # Extract rotation and translation
    R = T[:3, :3]
    t = T[:3, 3]

    # Camera centers in the coordinate system where camera1 is at the origin.
    C1 = np.zeros(3)                 # First camera at origin
    C2 = (-R.T @ t).reshape(3)       # Second camera center in the same coords

    # Vectors from camera centers to each 3D point
    # t_points is (N, 3), so the result is (N, 3)
    vec1 = points1 - C1[None, :]   # shape: (N, 3)
    vec2 = points2 - C2[None, :]   # shape: (N, 3)

    # Compute norms along axis=1 (per row) - distance of each point
    norms1 = np.linalg.norm(vec1, axis=1)  # shape: (N,)
    norms2 = np.linalg.norm(vec2, axis=1)  # shape: (N,)

    # Normalize vectors (element-wise division) - unit vectors
    vec1_norm = vec1 / (norms1[:, None] + 1e-8) # shape: (N, 3)
    vec2_norm = vec2 / (norms2[:, None] + 1e-8) # shape: (N, 3)

    # Compute dot product along axis=1 to get cos(theta)
    cos_angles = np.sum(vec1_norm * vec2_norm, axis=1)  # shape: (N,)

    # Clip to avoid numerical issues slightly outside [-1, 1]
    cos_angles = np.clip(cos_angles, -1.0, 1.0)

    # Convert to angles in degrees
    angles = np.degrees(np.arccos(cos_angles))  # shape: (N,)

    return angles

def triangulate(pxs1, pxs2, T):
    # Compute projection matrices for triangulation
    M1 = K @ np.eye(3,4)  # First camera at origin
    M2 = K @ T[:3, :]  # Second camera at R, t

    # Triangulate points
    if len(pxs1) != 2:
        points1_4d_hom = cv2.triangulatePoints(M1, M2, pxs1.T, pxs2.T)
    else:
        points1_4d_hom = cv2.triangulatePoints(M1, M2, pxs1, pxs2)

    # Convert homogeneous coordinates to 3D
    points1_3d = points1_4d_hom[:3] / points1_4d_hom[3]

    return points1_3d.T # (N, 3)

def compute_F12(frame1, frame2):
    """
    Compute the 3×3 fundamental matrix F12 mapping points in img1 → epipolar lines in img2.

    Parameters
    ----------
    frame1.pose, frame2.pose : 4×4 array
        Rigid transforms CAMERA→WORLD.

    Returns
    -------
    F12 : (3×3) ndarray
    """
    # 1) world→camera extrinsics
    Twc1 = invert_transform(frame1.pose)
    Twc2 = invert_transform(frame2.pose)
    R1, t1 = Twc1[:3, :3], Twc1[:3, 3]
    R2, t2 = Twc2[:3, :3], Twc2[:3, 3]

    # 2) relative motion: cam1 → cam2
    R12 = R2 @ R1.T
    t12 = t2 - R12 @ t1

    # 3) essential matrix
    E12 = skew_symmetric(t12) @ R12

    # 4) fundamental matrix
    F12 = np.linalg.inv(K).T @ E12 @ np.linalg.inv(K)
    return F12

def dist_epipolar_line(px1, px2, F12):
    """
    Squared distance of px2 to the epipolar line in img2 induced by px1 in img1.

    Parameters
    ----------
    px1 : array-like (2,) or tuple (x1, y1)
        Pixel coordinates in image 1.
    px2 : array-like (2,) or tuple (x2, y2)
        Pixel coordinates in image 2.
    F12 : (3×3) ndarray
        Fundamental matrix from img1 → img2.

    Returns
    -------
    float
        (a*x2 + b*y2 + c)**2 / (a² + b²), or np.inf if degenerate.
    """
    x1, y1 = px1
    x2, y2 = px2

    # epipolar line l = [a, b, c] in image2: l^T [x1, y1, 1] = 0
    l = F12 @ np.array([x1, y1, 1.0])
    a, b, c = l

    denom = a*a + b*b
    if denom < np.finfo(float).eps:
        return np.inf

    num = a*x2 + b*y2 + c
    return (num*num) / denom
