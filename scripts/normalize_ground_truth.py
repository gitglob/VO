import os
from pathlib import Path
import numpy as np
import pandas as pd

def quaternion_conjugate(q):
    """ Returns the conjugate of a quaternion [qx, qy, qz, qw]. """
    qx, qy, qz, qw = q
    return np.array([-qx, -qy, -qz, qw])

def quaternion_multiply(q1, q2):
    """ Multiplies two quaternions q1 and q2. """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def normalize_trajectory(ground_truth):
    """ Normalizes the trajectory by subtracting the first pose and rotation from all subsequent poses. """
    # Normalize position (translation)
    first_tx = ground_truth['tx'].iloc[0]
    first_ty = ground_truth['ty'].iloc[0]
    first_tz = ground_truth['tz'].iloc[0]
    
    ground_truth['tx'] -= first_tx
    ground_truth['ty'] -= first_ty
    ground_truth['tz'] -= first_tz
    
    # Normalize rotation (quaternion)
    first_q = np.array([
        ground_truth['qx'].iloc[0],
        ground_truth['qy'].iloc[0],
        ground_truth['qz'].iloc[0],
        ground_truth['qw'].iloc[0]
    ])
    first_q_conjugate = quaternion_conjugate(first_q)

    # Apply the inverse of the first rotation to all quaternions
    for i in range(len(ground_truth)):
        q = np.array([
            ground_truth['qx'].iloc[i],
            ground_truth['qy'].iloc[i],
            ground_truth['qz'].iloc[i],
            ground_truth['qw'].iloc[i]
        ])
        normalized_q = quaternion_multiply(first_q_conjugate, q)
        ground_truth.at[i, 'qx'] = normalized_q[0]
        ground_truth.at[i, 'qy'] = normalized_q[1]
        ground_truth.at[i, 'qz'] = normalized_q[2]
        ground_truth.at[i, 'qw'] = normalized_q[3]
    
    return ground_truth

def normalize_and_save_trajectory(input_file, output_file):
    """ 
    Reads a ground truth trajectory file, normalizes the translation and rotation, 
    and writes the normalized trajectory to a new file.
    """
    # Read the input file, skipping comment lines
    ground_truth = pd.read_csv(input_file, comment='#', sep='\s+', header=None,
                               names=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])

    # Normalize the trajectory
    ground_truth = normalize_trajectory(ground_truth)

    # Write the normalized trajectory to the output file
    with open(output_file, 'w') as f_out:
        # Write header information
        f_out.write("# ground truth trajectory\n")
        f_out.write(f"# file: '{output_file}'\n")
        f_out.write("# timestamp tx ty tz qx qy qz qw\n")
        
        # Write normalized data with 4 decimal places
        for _, row in ground_truth.iterrows():
            f_out.write(f"{row['timestamp']:.4f} {row['tx']:.4f} {row['ty']:.4f} {row['tz']:.4f} "
                        f"{row['qx']:.4f} {row['qy']:.4f} {row['qz']:.4f} {row['qw']:.4f}\n")

def main():
    main_dir = Path(__file__).parent.parent
    scene = "rgbd_dataset_freiburg2_pioneer_slam2"
    gt_txt = main_dir / "data" / scene / "groundtruth_clipped.txt"
    gt_txt_norm = main_dir / "data" / scene / "groundtruth_norm.txt"

    normalize_and_save_trajectory(gt_txt, gt_txt_norm)

if __name__ == "__main__":
    main()
    