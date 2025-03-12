import numpy as np
from pathlib import Path

data_dir = Path.home() / "Documents" / "data" / "vSLAM"
scene = "rgbd_dataset_freiburg2_pioneer_360"

main_dir = Path(__file__).parent
results_dir = main_dir / "results" / scene / "3d_2d"

use_distortion = False
if use_distortion:
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
    _dist_coeffs = np.array([d0, d1, d2, d3, d4], dtype=np.float64)
else:
    fx = 525.0  # focal length x
    fy = 525.0  # focal length y
    cx = 319.5  # optical center x
    cy = 239.5  # optical center y

_K = np.array([[fx,  0, cx],
                    [ 0, fy, cy],
                    [ 0,  0,  1]], dtype=np.float64)