import numpy as np


def invert_transform(T: np.ndarray) -> np.ndarray:
    """
    Efficiently invert a 4x4 transformation matrix assuming it is composed of
    a 3x3 orthonormal rotation part (R) and a 3x1 translation part (t).

    Parameters
    ----------
    T : np.ndarray
        A 4x4 homogeneous transformation matrix of the form:
        [ R  t ]
        [ 0  1 ]

    Returns
    -------
    T_inv : np.ndarray
        The inverse of T, also a 4x4 homogeneous transformation matrix.
    """
    # Extract rotation (R) and translation (t)
    R = T[:3, :3]
    t = T[:3, 3]

    # Create an empty 4x4 identity matrix for the result
    T_inv = np.eye(4)

    # R^T goes in the top-left 3x3
    T_inv[:3, :3] = R.T

    # -R^T * t goes in the top-right 3x1
    T_inv[:3, 3] = -R.T @ t

    return T_inv

def transform_points(points_3d: np.ndarray, T: np.ndarray):
    """
    Apply a 4x4 transformation matrix T to a Nx3 array of 3D points.
    Returns a Nx3 array of transformed 3D points.

    Args:
        points_3d: Point with shape (N, 3)
        T: Transformation with shape (4, 4)
    """
    # 1. Convert Nx3 -> Nx4 (homogeneous)
    ones = np.ones((points_3d.shape[0], 1))
    points_hom = np.hstack([points_3d, ones]).T # (4, N)

    # 2. Multiply by the transform (assume row vectors)
    transformed_hom = (T @ points_hom).T        # (N, 4)

    # 3. Normalize back to 3D
    w = transformed_hom[:, 3]
    x = transformed_hom[:, 0] / w
    y = transformed_hom[:, 1] / w
    z = transformed_hom[:, 2] / w
    transformed_3d = np.column_stack((x, y, z)) # (N, 3)

    return transformed_3d

def skew_symmetric(v):
    """
    Creates a skew-symmetric matrix from a 3-element vector.
    
    For a vector v = [v1, v2, v3], the skew-symmetric matrix is:
        [  0, -v3,  v2 ]
        [ v3,   0, -v1 ]
        [-v2,  v1,   0 ]
    
    Parameters
    ----------
    v : numpy.ndarray
        A 3-element vector (either shape (3,) or (3,1)).
    
    Returns
    -------
    numpy.ndarray
        The 3x3 skew-symmetric matrix corresponding to v.
    """
    # Ensure v is a flat 1D array of length 3.
    v = np.squeeze(v)
    return np.array([[0,     -v[2],  v[1]],
                     [v[2],   0,    -v[0]],
                     [-v[1],  v[0],  0]])
