# FAST-LIO ROS2 Copilot Instructions

## Project Overview

**FAST-LIO** (Fast LiDAR-Inertial Odometry) is a real-time LiDAR-Inertial SLAM system. It tightly fuses LiDAR feature points with IMU data using an iterated extended Kalman filter (IEKF) implemented via **IKFoM** (a custom on-manifold Kalman filter toolkit). The system maintains an incremental point cloud map using **ikd-Tree** (a dynamic KD-Tree optimized for fast 3D NN search).

## Architecture & Data Flow

### Core Components

1. **LaserMappingNode** (`src/laserMapping.cpp`): Main ROS2 node orchestrating the SLAM pipeline
   - **Timer callback** (100 Hz): Processes synchronized lidar-IMU measurement pairs via `timer_callback()`
   - **Sensor callbacks**: `livox_pcl_cbk()` (Livox format) and `standard_pcl_cbk()` (standard PointCloud2), `imu_cbk()` for buffering raw data

2. **ImuProcess** (`src/IMU_Processing.hpp`): Manages IMU preintegration and undistortion
   - Preintegrates IMU measurements between lidar scans using `imu_preintegration.hpp`
   - Undistorts lidar point clouds using IMU rotation/velocity estimates
   - Maintains IMU biases and covariances (gyroscope & accelerometer)

3. **Preprocess** (`src/preprocess.h/cpp`): Converts different lidar formats to unified PointType
   - Supports AVIA (Livox), VELO16 (Velodyne), OUST64 (Ouster), MID360 (Livox)
   - Handles point filtering, deskewing, and blind zone removal
   - Time unit conversion (SEC/MS/US/NS)

4. **ikd-Tree** (`include/ikd-Tree/`): Incremental KD-tree map representation
   - Fast point-to-map searches for scan-matching residuals
   - Dynamic insertion/deletion of points as the robot moves
   - Uses `lasermap_fov_segment()` to maintain local map around robot's field-of-view

5. **IKFoM** (`include/IKFoM_toolkit/esekfom/`): On-manifold Iterated Extended Kalman Filter
   - Optimizes pose, velocity, IMU biases, and lidar-IMU extrinsics
   - Custom manifold representation via `use-ikfom.hpp` and SO(3) math (`include/so3_math.h`)

### Data Pipeline

```
LiDAR Raw → Preprocess → Feature Extraction (optional) → Buffer
    ↓
IMU Raw → ImuProcess → Preintegration → Buffer
    ↓
[Sync when both buffers have data for lidar scan duration]
    ↓
ImuProcess::Process() → Undistort Points in Lidar Frame
    ↓
FOV Segmentation → Map Management (delete far points)
    ↓
Downsampling → Map Increments → ikd-Tree Updates
    ↓
Scan-to-Map Matching (residuals) + IMU Factors → IKFoM Optimization
    ↓
Publish: Odometry + Pointcloud (world/body frames)
```

## Build & Run

### Build
```bash
# From ROS2 workspace root
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

### Launch
```bash
# Default config: mid360.yaml
ros2 launch fast_lio mapping.launch.py

# Custom lidar (avia, horizon, ouster64, velodyne)
ros2 launch fast_lio mapping.launch.py config_file:=avia.yaml

# With RViz visualization
ros2 launch fast_lio mapping.launch.py rviz:=true
```

## Key Configuration Parameters

**`config/*.yaml`** controls all algorithm behavior:

- **preprocess.lidar_type**: 1=AVIA, 2=VELO16, 3=OUST64, 4=MID360
- **preprocess.scan_line**: Number of scan lines (6 for Livox, 16+ for Velodyne/Ouster)
- **mapping.fov_degree**: Lidar field-of-view (affects local map boundaries)
- **mapping.det_range**: Detection range for odometry (300-450m typical)
- **mapping.extrinsic_est_en**: Enable online estimation of lidar-IMU transformation
- **filter_size_surf/map**: Voxel downsample sizes (larger = faster but less accurate)
- **imu_propagation.enable_high_frequency**: High-frequency IMU odometry at 200Hz via `imu_highfreq_callback()`

## Critical Patterns & Conventions

### State Representation
- **state_ikfom**: Pose, velocity, IMU biases, gravity, lidar-IMU extrinsics (see `use-ikfom.hpp`)
- **PointType**: `pcl::PointXYZINormal` with intensity and curvature (see `preprocess.h`)
- **SO(3) rotations**: Use `MTK::SO3<double>` (manifold representation) not direct rotation matrices

### Undistortion
Points are motion-undistorted using IMU velocity/rotation between scan start/end times:
- `RGBpointBodyToWorld()` transforms body→world frame
- `pointBodyToWorld_ikfom()` is the extended Kalman filter version
- Lidar points assumed synchronized to IMU frame via `offset_T_L_I`, `offset_R_L_I`

### Matching & Residuals
- Scan-to-map uses nearest neighbor search: compute distance from undistorted point to nearest map point
- Residuals feed into IEKF cost function (see residual computation in timer_callback)
- Downsampling prevents map bloat: reject points with nearby map neighbors

### Multithreading
- ROS2 callbacks (`livox_pcl_cbk`, `imu_cbk`) buffer data with mutex `mtx_buffer`
- Main processing loop (`timer_callback`) runs synchronously at 100 Hz
- **NEW**: High-frequency IMU output (`imu_highfreq_callback`) independent thread for real-time state propagation

## Common Modifications

**Changing Lidar Support**: Update `preprocess.cpp::Preprocess::process()` method for new lidar formats

**Tuning Covariances**: Adjust `mapping.{gyr_cov, acc_cov, b_gyr_cov, b_acc_cov}` and `imu_propagation.{acc_n, gyr_n, acc_w, gyr_w}` in config files

**Output Formats**: Modify `publish_*()` functions in `src/laserMapping.cpp` to add custom messages

**Extrinsic Calibration**: Set `mapping.extrinsic_T` and `mapping.extrinsic_R` manually or enable `extrinsic_est_en: true` for online optimization

## Testing & Debugging

- **Time logging**: Enable `runtime_pos_log_enable: true` to log timing stats to `Log/fast_lio_time_log.csv`
- **PCD saving**: Set `pcd_save.pcd_save_en: true` to save maps to `PCD/` directory
- **Debug output**: Check `Log/{mat_pre.txt, mat_out.txt, dbg.txt}` for internal matrices
- **RViz**: Monitor `/cloud_registered` (map), `/cloud_registered_body` (lidar frame), `/high_freq_odom` (IMU propagation)

## Dependencies

- **ROS2** (Foxy+, Humble recommended)
- **PCL** >= 1.8, **Eigen** >= 3.3.4
- **livox_ros_driver2** (required even for non-Livox sensors, for message definitions)
- **OpenMP** for parallel processing (automatically detected and tuned based on CPU cores)
- **Python** headers for optional plotting/debugging with matplotlib

## Important Notes

- Lidar and IMU **must be hardware-synchronized** or calibrated via `time_offset_lidar_to_imu` / `time_sync_en`
- Initial scan (first INIT_TIME seconds) is used for system initialization; avoid rapid motion
- KD-tree memory grows unbounded if `filter_size_map_min` is too small; tune based on environment
- High-frequency IMU output requires continuous IMU data; gaps cause state propagation errors
