# VO

Simple Visual Odometry using ORB features.

**Note**: The only purpose of this repository is to study and get some experience with Visual Odometry and its variants, as a first step to understanding how modern vSLAM systems work.

## Monocular Visual Odometry (2D-2D) on the KITTI Dataset

A simple 2D-2D Essential/Homography Matrix - based approach was used to calculate the trajectory in 2 Kitti scenes.

### 00

<img src="resources/00/4540.png" alt="kitti 00 2D Trajectory" width="500">
<img src="resources/00/final_trajectory.png" alt="kitti 00 3D Trajectory" width="500">

### 06

<img src="resources/06/1100.png" alt="kitti 06 2D Trajectory" width="500">
<img src="resources/06/final_trajectory.png" alt="kitti 06 3D Trajectory" width="500">

## RGB-D Visual Odometry on the TUM RGBD Dataset

A simple 3D-2D PnP - based approach was used to calculate the trajectory in one TUM RGB-D scene.

<img src="resources/rgbd_frei_360/1224.png" alt="TUM 2D Trajectory" width="500">
<img src="resources/rgbd_frei_360/final_trajectory.png" alt="TUM 3D Trajectory" width="500">

## Conclusion

The results indicate that this simple, bare-bones approach is not enough to provide accurate localization.
There is severe drift that makes the calculated trajectory deviate a lot from ground truth.

## Future Work

On branch "slam", exists an attempt for a simple, but complete vSLAM system.
An entire experiment, however, cannot be executed because creating new map points is too computationally expensive.

In the future, I plan to re-create it using C++.