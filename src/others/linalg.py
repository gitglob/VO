import numpy as np



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
