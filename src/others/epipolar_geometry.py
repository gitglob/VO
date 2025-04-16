import numpy as np
import cv2
from src.others.linalg import skew_symmetric
from config import K


def triangulate(q_frame_pixels, t_frame_pixels, T):
    # Compute projection matrices for triangulation
    q_M = K @ np.eye(3,4)  # First camera at origin
    t_M = K @ T[:3, :]  # Second camera at R, t

    # Triangulate points
    q_frame_points_4d_hom = cv2.triangulatePoints(q_M, t_M, q_frame_pixels.T, t_frame_pixels.T)

    # Convert homogeneous coordinates to 3D
    q_points_3d = q_frame_points_4d_hom[:3] / q_frame_points_4d_hom[3]

    return q_points_3d.T # (N, 3)

def compute_T12(frame1, frame2):
    """
    Computes the Transformation Matrix between 2 frames.
    
    Returns
    -------
    numpy.ndarray
        The 4x4 transformation matrix (T12) relating the two keyframes.
    """
    # Retrieve rotation and translation for the first keyframe (from world to keyframe)
    R1w = frame1.pose[:3, :3]
    t1w = frame1.pose[:3, 3]
    
    # Retrieve rotation and translation for the second keyframe
    R2w = frame2.pose[:3, :3]
    t2w = frame2.pose[:3, 3]
    
    # Compute the relative rotation: R12 = R1w * (R2w).T
    R12 = R1w @ R2w.T
    
    # Compute the relative translation: t12 = -R1w * (R2w).T * t2w + t1w
    t12 = -R1w @ (R2w.T @ t2w) + t1w
    
    T12 = np.eye(4)
    T12[:3, :3] = R12
    T12[:3, 3] = t12 
    return T12

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
    R1w = frame1.pose[:3, :3]
    t1w = frame1.pose[:3, 3]
    
    # Retrieve rotation and translation for the second keyframe
    R2w = frame2.pose[:3, :3]
    t2w = frame2.pose[:3, 3]
    
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