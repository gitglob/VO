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
    Computes the fundamental matrix (F12) between two keyframes based on their camera poses
    and intrinsic calibration matrices.
    
    The fundamental matrix is computed using the relative rotation and translation between
    the two keyframes. The steps are:
    
        1. Retrieve the rotations (R1w, R2w) and translations (t1w, t2w) for each keyframe.
        2. Compute the relative rotation:
               R12 = R1w * (R2w)^T
        3. Compute the relative translation:
               t12 = -R1w * (R2w)^T * t2w + t1w
        4. Construct the skew-symmetric matrix of t12 (denoted as t12x).
        5. Retrieve the calibration matrices (K1, K2) for the keyframes.
        6. Compute the fundamental matrix as:
               F12 = inv(K1.T) * t12x * R12 * inv(K2)
    
    Parameters
    ----------
    frame1 : KeyFrame
        The first keyframe object. Expected to have methods:
            - GetRotation(): returns a (3,3) numpy.ndarray.
            - GetTranslation(): returns a (3,) or (3,1) numpy.ndarray.
            - GetCalibrationMatrix(): returns a (3,3) numpy.ndarray.
    frame2 : KeyFrame
        The second keyframe object. Expected to have the same methods as frame1.
    
    Returns
    -------
    numpy.ndarray
        The 3x3 fundamental matrix (F12) relating the two keyframes.
    """
    # Retrieve rotation and translation for the first keyframe (from world to keyframe)
    R1w = frame1.R
    t1w = frame1.t
    
    # Retrieve rotation and translation for the second keyframe
    R2w = frame2.R
    t2w = frame2.t
    
    # Compute the relative rotation: R12 = R1w * (R2w).T
    R12 = R1w @ R2w.T
    
    # Compute the relative translation: t12 = -R1w * (R2w).T * t2w + t1w
    t12 = -R1w @ (R2w.T @ t2w) + t1w
    
    # Compute the skew-symmetric matrix of t12
    t12x = skew_symmetric(t12)
    
    # Compute the fundamental matrix:
    # F12 = inv(K1.T) * t12x * R12 * inv(K2)
    F12 = np.linalg.inv(K.T) @ t12x @ R12 @ np.linalg.inv(K)
    
    return F12

def dist_epipolar_line(px1, px2, F12):
    """
    Returns the squared distance from a pixel in the second image (kp2) to the epipolar line
    (computed from a pixel in the first image, kp1, and the fundamental matrix F12).
    
    The epipolar line in the second image is given by:
        l = [a, b, c]
    where:
        a = px1[0]*F12[0,0] + px1[1]*F12[1,0] + F12[2,0]
        b = px1[0]*F12[0,1] + px1[1]*F12[1,1] + F12[2,1]
        c = px1[0]*F12[0,2] + px1[1]*F12[1,2] + F12[2,2]
    
    The squared distance from kp2 to the epipolar line is computed as:
        dsqr = (a*px2[0] + b*px2[1] + c)^2 / (a^2 + b^2)
    
    Parameters
    ----------
    kp1 : cv2.KeyPoint
        Keypoint from the first image.
    kp2 : cv2.KeyPoint
        Keypoint from the second image.
    F12 : numpy.ndarray
        The 3x3 Fundamental matrix relating the two images.
    
    Returns
    -------
    float
        the distance kp2 -> epipolar line squared
    """
    # Compute the coefficients [a, b, c] of the epipolar line in the second image.
    a = px1[0]*F12[0, 0] + px1[1]*F12[1, 0] + F12[2, 0]
    b = px1[0]*F12[0, 1] + px1[1]*F12[1, 1] + F12[2, 1]
    c = px1[0]*F12[0, 2] + px1[1]*F12[1, 2] + F12[2, 2]

    # Compute the numerator of the distance formula.
    num = a * px2[0] + b * px2[1] + c

    # Compute the denominator as the squared norm of the line coefficients.
    den = a*a + b*b

    # If the denominator is zero, the line is degenerate, so return False.
    if den == 0:
        return False

    # Compute the squared distance from kp2 to the epipolar line.
    dsqr = (num * num) / den

    return dsqr