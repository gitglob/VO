import yaml
from pathlib import Path
import numpy as np
from src.utils.utils import setup_logger


main_dir = Path(__file__).parent
data_dir = Path.home() / "Documents" / "data" / "kitti"
scene = "00"
results_dir = main_dir / "results" / scene / "orb"

fx = 718.86
fy = 718.86
cx = 607.19
cy = 185.22
K = np.array([[fx, 0., cx],
              [0., fy, cy],
              [0., 0., 1.]])

np.set_printoptions(precision=2, suppress=True)

# Settings
settings_path = main_dir / "settings" / "config.yaml"
with open(settings_path, "r") as file:
    SETTINGS = yaml.safe_load(file)

log_dir = main_dir / "logs"
log = setup_logger(log_dir)
