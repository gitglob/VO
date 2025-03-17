import yaml
from pathlib import Path
import numpy as np


main_dir = Path(__file__).parent
data_dir = Path.home() / "Documents" / "data" / "kitti"
scene = "00"
results_dir = main_dir / "results" / scene / "orb"

image_width = 1243
image_height = 376

np.set_printoptions(precision=2, suppress=True)
debug = True


# Settings
settings_path = main_dir / "settings" / "config.yaml"
with open(settings_path, "r") as file:
    SETTINGS = yaml.safe_load(file)
