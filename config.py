from pathlib import Path

main_dir = Path(__file__).parent
data_dir = Path.home() / "Documents" / "data" / "kitti"
scene = "06"
results_dir = main_dir / "results" / scene / "2d_2d"
