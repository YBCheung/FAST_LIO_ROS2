#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

class LidarCameraColorizerNode : public rclcpp::Node {
public:
  LidarCameraColorizerNode() : Node("lidar_camera_colorizer_node") {
     declare_parameter<std::string>("color_mapping.input_source", "livox");
    declare_parameter<std::string>("color_mapping.cloud_topic", "/Laser_map");
    declare_parameter<std::string>("color_mapping.livox_topic", "/livox/lidar");
    declare_parameter<std::string>("color_mapping.odom_topic", "/Odometry");
    declare_parameter<std::string>("color_mapping.rgb_topic", "/camera/color/image_raw/compressed");
    declare_parameter<std::string>("color_mapping.depth_topic", "/camera/depth/image_raw/compressedDepth");
    declare_parameter<std::string>("color_mapping.camera_info_topic", "/camera/color/camera_info");

    declare_parameter<std::string>("color_mapping.colored_scan_topic", "/cloud_registered_color");
    declare_parameter<std::string>("color_mapping.colored_map_topic", "/laser_map_color");

    declare_parameter<double>("color_mapping.fx", 0.0);
    declare_parameter<double>("color_mapping.fy", 0.0);
    declare_parameter<double>("color_mapping.cx", 0.0);
    declare_parameter<double>("color_mapping.cy", 0.0);
    declare_parameter<bool>("color_mapping.use_camera_info", true);

    declare_parameter<std::vector<double>>(
        "color_mapping.T_cam_lidar",
        std::vector<double>{
            0.0280838, -0.981278, 0.190538, -0.602142,
            -0.217406, -0.192047, -0.957002, 0.762459,
            0.975677, -0.0145479, -0.218729, 0.394083,
            0.0, 0.0, 0.0, 1.0});
    declare_parameter<bool>("color_mapping.invert_extrinsic", false);

    declare_parameter<std::vector<double>>("mapping.extrinsic_R",
                                           std::vector<double>{1.0, 0.0, 0.0,
                                                               0.0, 1.0, 0.0,
                                                               0.0, 0.0, 1.0});
    declare_parameter<std::vector<double>>("mapping.extrinsic_T",
                                           std::vector<double>{0.0, 0.0, 0.0});

    declare_parameter<double>("color_mapping.max_image_time_diff", 0.05);
    declare_parameter<double>("color_mapping.max_depth_time_diff", 0.05);
    declare_parameter<double>("color_mapping.max_odom_time_diff", 0.02);

    declare_parameter<bool>("color_mapping.use_depth_gate", true);
    declare_parameter<double>("color_mapping.depth_tolerance", 0.25);
    declare_parameter<double>("color_mapping.depth_scale", 0.001);
    declare_parameter<double>("color_mapping.min_depth", 0.1);
    declare_parameter<double>("color_mapping.max_depth", 80.0);

    declare_parameter<double>("color_mapping.map_voxel_size", 0.10);
    declare_parameter<int>("color_mapping.publish_map_every_n", 5);
    declare_parameter<int>("color_mapping.max_queue_size", 100);
    declare_parameter<bool>("color_mapping.filter_livox_tags", true);
    declare_parameter<std::string>("color_mapping.world_frame", "world");

    std::string input_source;
    get_parameter("color_mapping.input_source", input_source);
    get_parameter("color_mapping.cloud_topic", cloud_topic_);
    get_parameter("color_mapping.livox_topic", livox_topic_);
    get_parameter("color_mapping.odom_topic", odom_topic_);
    get_parameter("color_mapping.rgb_topic", rgb_topic_);
    get_parameter("color_mapping.depth_topic", depth_topic_);
    get_parameter("color_mapping.camera_info_topic", camera_info_topic_);
    get_parameter("color_mapping.colored_scan_topic", colored_scan_topic_);
    get_parameter("color_mapping.colored_map_topic", colored_map_topic_);
    get_parameter("color_mapping.fx", fx_);
    get_parameter("color_mapping.fy", fy_);
    get_parameter("color_mapping.cx", cx_);
    get_parameter("color_mapping.cy", cy_);
    get_parameter("color_mapping.use_camera_info", use_camera_info_);

    std::vector<double> t_cam_lidar;
    get_parameter("color_mapping.T_cam_lidar", t_cam_lidar);
    get_parameter("color_mapping.invert_extrinsic", invert_extrinsic_);

    std::vector<double> mapping_extrinsic_r;
    std::vector<double> mapping_extrinsic_t;
    get_parameter("mapping.extrinsic_R", mapping_extrinsic_r);
    get_parameter("mapping.extrinsic_T", mapping_extrinsic_t);

    get_parameter("color_mapping.max_image_time_diff", max_image_time_diff_);
    get_parameter("color_mapping.max_depth_time_diff", max_depth_time_diff_);
    get_parameter("color_mapping.max_odom_time_diff", max_odom_time_diff_);

    get_parameter("color_mapping.use_depth_gate", use_depth_gate_);
    get_parameter("color_mapping.depth_tolerance", depth_tolerance_);
    get_parameter("color_mapping.depth_scale", depth_scale_);
    get_parameter("color_mapping.min_depth", min_depth_);
    get_parameter("color_mapping.max_depth", max_depth_);

    get_parameter("color_mapping.map_voxel_size", map_voxel_size_);
    get_parameter("color_mapping.publish_map_every_n", publish_map_every_n_);
    get_parameter("color_mapping.max_queue_size", max_queue_size_);
    get_parameter("color_mapping.filter_livox_tags", filter_livox_tags_);
    get_parameter("color_mapping.world_frame", world_frame_);

    if (t_cam_lidar.size() != 16) {
      RCLCPP_ERROR(get_logger(), "color_mapping.T_cam_lidar must contain 16 values. Using identity.");
      t_cam_lidar = {1.0, 0.0, 0.0, 0.0,
                     0.0, 1.0, 0.0, 0.0,
                     0.0, 0.0, 1.0, 0.0,
                     0.0, 0.0, 0.0, 1.0};
    }

    Eigen::Matrix4d t_cl = Eigen::Matrix4d::Identity();
    for (int row = 0; row < 4; ++row) {
      for (int col = 0; col < 4; ++col) {
        t_cl(row, col) = t_cam_lidar[row * 4 + col];
      }
    }
    if (invert_extrinsic_) {
      t_cl = t_cl.inverse();
    }
    r_cl_ = t_cl.block<3, 3>(0, 0);
    t_cl_ = t_cl.block<3, 1>(0, 3);

    if (mapping_extrinsic_r.size() == 9) {
      r_li_ << mapping_extrinsic_r[0], mapping_extrinsic_r[1], mapping_extrinsic_r[2],
          mapping_extrinsic_r[3], mapping_extrinsic_r[4], mapping_extrinsic_r[5],
          mapping_extrinsic_r[6], mapping_extrinsic_r[7], mapping_extrinsic_r[8];
    } else {
      RCLCPP_WARN(get_logger(), "mapping.extrinsic_R must contain 9 values. Using identity.");
      r_li_.setIdentity();
    }

    if (mapping_extrinsic_t.size() == 3) {
      t_li_ << mapping_extrinsic_t[0], mapping_extrinsic_t[1], mapping_extrinsic_t[2];
    } else {
      RCLCPP_WARN(get_logger(), "mapping.extrinsic_T must contain 3 values. Using zero.");
      t_li_.setZero();
    }

    if (use_camera_info_) {
      camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
          camera_info_topic_, rclcpp::SensorDataQoS(),
          std::bind(&LidarCameraColorizerNode::cameraInfoCallback, this, std::placeholders::_1));
    } else {
      has_intrinsics_ = fx_ > 0.0 && fy_ > 0.0;
    }
    rgb_sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
      rgb_topic_, rclcpp::QoS(10).reliable(),
      std::bind(&LidarCameraColorizerNode::rgbCallback, this, std::placeholders::_1));

    depth_sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
      depth_topic_, rclcpp::QoS(10).reliable(),
      std::bind(&LidarCameraColorizerNode::depthCallback, this, std::placeholders::_1));
        
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, rclcpp::SensorDataQoS(),
        std::bind(&LidarCameraColorizerNode::odomCallback, this, std::placeholders::_1));

    if (input_source == "livox") {
      input_mode_ = InputMode::LIVOX_CUSTOM;
      livox_sub_ = create_subscription<livox_ros_driver2::msg::CustomMsg>(
          livox_topic_, rclcpp::SensorDataQoS(),
          std::bind(&LidarCameraColorizerNode::livoxCallback, this, std::placeholders::_1));
    } else {
      input_mode_ = InputMode::WORLD_POINTCLOUD2;
      cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
          cloud_topic_, rclcpp::SensorDataQoS(),
          std::bind(&LidarCameraColorizerNode::cloudCallback, this, std::placeholders::_1));
      RCLCPP_WARN(get_logger(), "Using /Laser_map (global map) for coloring can cause stale-color artifacts.");
    }

    colored_scan_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(colored_scan_topic_, 10);
    colored_map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(colored_map_topic_, 2);

    RCLCPP_INFO(get_logger(), "Lidar-camera colorizer started.");
      RCLCPP_INFO(get_logger(), "Node initialization complete. Waiting for messages...");
    if (input_mode_ == InputMode::LIVOX_CUSTOM) {
      RCLCPP_INFO(get_logger(), "Sub topics: livox=%s odom=%s rgb=%s depth=%s",
                  livox_topic_.c_str(), odom_topic_.c_str(), rgb_topic_.c_str(), depth_topic_.c_str());
    } else {
      RCLCPP_INFO(get_logger(), "Sub topics: cloud=%s odom=%s rgb=%s depth=%s",
                  cloud_topic_.c_str(), odom_topic_.c_str(), rgb_topic_.c_str(), depth_topic_.c_str());
    }
    RCLCPP_INFO(get_logger(), "Pub topics: colored_scan=%s colored_map=%s",
                colored_scan_topic_.c_str(), colored_map_topic_.c_str());

    if (rgb_topic_.find("compressed") == std::string::npos) {
      RCLCPP_WARN(get_logger(),
                  "RGB subscriber expects sensor_msgs/msg/CompressedImage, but rgb_topic '%s' does not look like a compressed topic.",
                  rgb_topic_.c_str());
    }
    if (depth_topic_.find("compressed") == std::string::npos) {
      RCLCPP_WARN(get_logger(),
                  "Depth subscriber expects sensor_msgs/msg/CompressedImage, but depth_topic '%s' does not look like a compressed topic.",
                  depth_topic_.c_str());
    }
  }

private:
  enum class InputMode {
    LIVOX_CUSTOM,
    WORLD_POINTCLOUD2,
  };

  struct VoxelKey {
    int x;
    int y;
    int z;

    bool operator==(const VoxelKey &other) const {
      return x == other.x && y == other.y && z == other.z;
    }
  };

  struct VoxelKeyHasher {
    std::size_t operator()(const VoxelKey &key) const {
      std::size_t h1 = std::hash<int>{}(key.x);
      std::size_t h2 = std::hash<int>{}(key.y);
      std::size_t h3 = std::hash<int>{}(key.z);
      return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
  };

  struct VoxelValue {
    double sum_x{0.0};
    double sum_y{0.0};
    double sum_z{0.0};
    double sum_r{0.0};
    double sum_g{0.0};
    double sum_b{0.0};
    std::uint32_t count{0};
  };

  template <typename MsgT>
  typename MsgT::ConstSharedPtr findClosestByStamp(
      const std::deque<typename MsgT::ConstSharedPtr> &queue,
      const rclcpp::Time &stamp,
      double max_dt) {
    if (queue.empty()) {
      return nullptr;
    }

    typename MsgT::ConstSharedPtr best_msg;
    double best_dt = std::numeric_limits<double>::max();

    for (auto it = queue.rbegin(); it != queue.rend(); ++it) {
      const auto &msg = *it;
      const rclcpp::Time msg_stamp(msg->header.stamp);
      const double dt = std::abs((msg_stamp - stamp).seconds());
      if (dt < best_dt) {
        best_dt = dt;
        best_msg = msg;
      } else if (msg_stamp <= stamp) {
        break;
      }
    }

    if (!best_msg || best_dt > max_dt) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Closest message dt=%.6f exceeds max_dt=%.6f", best_dt, max_dt);
      return nullptr;
    }
    return best_msg;
  }

  template <typename MsgT>
  void pruneQueue(std::deque<typename MsgT::ConstSharedPtr> &queue) {
    while (static_cast<int>(queue.size()) > max_queue_size_) {
      queue.pop_front();
    }
  }

  bool getSyncedInputs(const rclcpp::Time &stamp,
                       sensor_msgs::msg::CompressedImage::ConstSharedPtr &rgb_msg,
                       sensor_msgs::msg::CompressedImage::ConstSharedPtr &depth_msg,
                       nav_msgs::msg::Odometry::ConstSharedPtr &odom_msg) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (rgb_queue_.empty() || depth_queue_.empty() || odom_queue_.empty()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Input queue empty near lidar stamp %.3f: rgb=%zu depth=%zu odom=%zu. "
        "Check topic names/types. Current topics: rgb=%s depth=%s odom=%s",
        stamp.seconds(), rgb_queue_.size(), depth_queue_.size(), odom_queue_.size(),
        rgb_topic_.c_str(), depth_topic_.c_str(), odom_topic_.c_str());
    }

    rgb_msg = findClosestByStamp<sensor_msgs::msg::CompressedImage>(rgb_queue_, stamp, max_image_time_diff_);
    depth_msg = findClosestByStamp<sensor_msgs::msg::CompressedImage>(depth_queue_, stamp, max_depth_time_diff_);
    odom_msg = findClosestByStamp<nav_msgs::msg::Odometry>(odom_queue_, stamp, max_odom_time_diff_);
    int mask = 0;
    if (!rgb_msg) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
               "No RGB image found for timestamp %.3f", stamp.seconds());
    mask |= 1;
    }
    if (!depth_msg) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
               "No depth image found for timestamp %.3f", stamp.seconds());
    mask |= 2;
    }
    if (!odom_msg) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
               "No odometry found for timestamp %.3f", stamp.seconds());
    mask |= 4;
    }
    return (mask == 0);
}

  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr &msg) {
    fx_ = msg->k[0];
    fy_ = msg->k[4];
    cx_ = msg->k[2];
    cy_ = msg->k[5];
    has_intrinsics_ = fx_ > 0.0 && fy_ > 0.0;
  }

  void rgbCallback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &msg) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    rgb_queue_.push_back(msg);
    pruneQueue<sensor_msgs::msg::CompressedImage>(rgb_queue_);
  }

  void depthCallback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &msg) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    depth_queue_.push_back(msg);
    pruneQueue<sensor_msgs::msg::CompressedImage>(depth_queue_);
  }

  void odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr &msg) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    odom_queue_.push_back(msg);
    pruneQueue<nav_msgs::msg::Odometry>(odom_queue_);
  }

  bool readDepthMeters(const cv::Mat &depth_image, int u, int v,
                       const std::string &encoding, double &depth_meters) const {
    if (u < 0 || v < 0 || u >= depth_image.cols || v >= depth_image.rows) {
      return false;
    }

    if (encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
      const std::uint16_t value = depth_image.at<std::uint16_t>(v, u);
      if (value == 0) {
        return false;
      }
      depth_meters = static_cast<double>(value) * depth_scale_;
      return std::isfinite(depth_meters);
    }

    if (encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
      const float value = depth_image.at<float>(v, u);
      if (!std::isfinite(value) || value <= 0.0f) {
        return false;
      }
      depth_meters = static_cast<double>(value);
      return true;
    }

    return false;
  }

  void updateVoxelMap(const pcl::PointXYZRGB &point) {
    const int vx = static_cast<int>(std::floor(point.x / map_voxel_size_));
    const int vy = static_cast<int>(std::floor(point.y / map_voxel_size_));
    const int vz = static_cast<int>(std::floor(point.z / map_voxel_size_));

    VoxelKey key{vx, vy, vz};
    VoxelValue &voxel = voxel_map_[key];
    voxel.sum_x += point.x;
    voxel.sum_y += point.y;
    voxel.sum_z += point.z;
    voxel.sum_r += point.r;
    voxel.sum_g += point.g;
    voxel.sum_b += point.b;
    voxel.count += 1;
  }

  void publishColoredMap(const rclcpp::Time &stamp) {
    pcl::PointCloud<pcl::PointXYZRGB> map_cloud;
    map_cloud.reserve(voxel_map_.size());

    for (const auto &it : voxel_map_) {
      const VoxelValue &voxel = it.second;
      if (voxel.count == 0) {
        continue;
      }
      pcl::PointXYZRGB point;
      const double inv_count = 1.0 / static_cast<double>(voxel.count);
      point.x = static_cast<float>(voxel.sum_x * inv_count);
      point.y = static_cast<float>(voxel.sum_y * inv_count);
      point.z = static_cast<float>(voxel.sum_z * inv_count);
      point.r = static_cast<std::uint8_t>(std::clamp(voxel.sum_r * inv_count, 0.0, 255.0));
      point.g = static_cast<std::uint8_t>(std::clamp(voxel.sum_g * inv_count, 0.0, 255.0));
      point.b = static_cast<std::uint8_t>(std::clamp(voxel.sum_b * inv_count, 0.0, 255.0));
      map_cloud.push_back(point);
    }

    sensor_msgs::msg::PointCloud2 map_msg;
    pcl::toROSMsg(map_cloud, map_msg);
    map_msg.header.stamp = stamp;
    map_msg.header.frame_id = world_frame_;
    colored_map_pub_->publish(map_msg);
  }

  bool projectAndColorPoint(const Eigen::Vector3d &p_l,
                            const cv::Mat &rgb_img,
                            const cv::Mat &depth_img,
                            const std::string &depth_encoding,
                            std::uint8_t &r,
                            std::uint8_t &g,
                            std::uint8_t &b) const {
    const Eigen::Vector3d p_c = r_cl_ * p_l + t_cl_;
    if (p_c.z() <= 0.0) {
      return false;
    }

    const double u_float = fx_ * (p_c.x() / p_c.z()) + cx_;
    const double v_float = fy_ * (p_c.y() / p_c.z()) + cy_;
    const int u = static_cast<int>(std::lround(u_float));
    const int v = static_cast<int>(std::lround(v_float));

    if (u < 0 || v < 0 || u >= rgb_img.cols || v >= rgb_img.rows) {
      return false;
    }

    if (use_depth_gate_) {
      double depth_m = 0.0;
      if (!readDepthMeters(depth_img, u, v, depth_encoding, depth_m)) {
        return false;
      }
      if (depth_m < min_depth_ || depth_m > max_depth_) {
        return false;
      }
      if (std::abs(depth_m - p_c.z()) > depth_tolerance_) {
        return false;
      }
    }

    const cv::Vec3b bgr = rgb_img.at<cv::Vec3b>(v, u);
    r = bgr[2];
    g = bgr[1];
    b = bgr[0];
    return true;
  }

  bool prepareSynchronizedData(
      const rclcpp::Time &stamp,
      cv_bridge::CvImagePtr &rgb_cv_ptr,
      cv_bridge::CvImageConstPtr &depth_cv_ptr,
      nav_msgs::msg::Odometry::ConstSharedPtr &odom_msg) {
    if (!has_intrinsics_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "No camera intrinsics available yet. Waiting for camera_info or fx/fy/cx/cy params.");
      return false;
    }

    sensor_msgs::msg::CompressedImage::ConstSharedPtr rgb_msg;
    sensor_msgs::msg::CompressedImage::ConstSharedPtr depth_msg;
    if (!getSyncedInputs(stamp, rgb_msg, depth_msg, odom_msg)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "Missing synchronized rgb/depth/odom around lidar timestamp. ");
      return false;
    }

    // Decode compressed RGB image
    cv::Mat rgb_img;
    try {
      rgb_img = cv::imdecode(rgb_msg->data, cv::IMREAD_COLOR);
      if (rgb_img.empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Failed to decode RGB compressed image in prepareSynchronizedData.");
        return false;
      }
        rgb_cv_ptr = std::make_shared<cv_bridge::CvImage>(
          rgb_msg->header, sensor_msgs::image_encodings::BGR8, rgb_img);
    } catch (const std::exception &e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                            "Exception decoding RGB compressed image in prepareSynchronizedData: %s", e.what());
      return false;
    }

    // Decode compressed depth image
    cv::Mat depth_img;
    try {
      const auto &depth_data = depth_msg->data;
      if (depth_data.empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Depth compressed image payload is empty.");
        return false;
      }

      depth_img = cv::imdecode(depth_data, cv::IMREAD_UNCHANGED);

      const bool is_compressed_depth =
          depth_msg->format.find("compressedDepth") != std::string::npos;

      if (depth_img.empty() && is_compressed_depth) {
        constexpr unsigned char png_magic[] = {0x89, 0x50, 0x4E, 0x47};
        auto png_it = std::search(depth_data.begin(), depth_data.end(),
                                  std::begin(png_magic), std::end(png_magic));

        if (png_it != depth_data.end()) {
          const auto png_offset = static_cast<std::size_t>(
              std::distance(depth_data.begin(), png_it));
          std::vector<uint8_t> png_payload(depth_data.begin() + png_offset,
                                           depth_data.end());
          depth_img = cv::imdecode(png_payload, cv::IMREAD_UNCHANGED);
        }
      }

      if (depth_img.empty()) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Failed to decode depth compressed image. format='%s', bytes=%zu",
            depth_msg->format.c_str(), depth_data.size());
        return false;
      }

      std::string depth_encoding;
      if (depth_img.type() == CV_16UC1) {
        depth_encoding = sensor_msgs::image_encodings::TYPE_16UC1;
      } else if (depth_img.type() == CV_32FC1) {
        depth_encoding = sensor_msgs::image_encodings::TYPE_32FC1;
      } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Unsupported decoded depth type: %d", depth_img.type());
        return false;
      }

        depth_cv_ptr = std::make_shared<cv_bridge::CvImage>(
          depth_msg->header, depth_encoding, depth_img);
    } catch (const std::exception &e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                            "Exception decoding depth compressed image in prepareSynchronizedData: %s", e.what());
      return false;
    }

    return true;
  }

  bool computeWorldFromLidar(const Eigen::Vector3d &p_l,
                             const nav_msgs::msg::Odometry::ConstSharedPtr &odom_msg,
                             Eigen::Vector3d &p_w,
                             Eigen::Matrix3d &r_wi,
                             Eigen::Vector3d &t_wi) const {
    const auto &pose = odom_msg->pose.pose;
    Eigen::Quaterniond q_wi(pose.orientation.w, pose.orientation.x,
                            pose.orientation.y, pose.orientation.z);
    if (q_wi.norm() < 1e-6) {
      return false;
    }
    q_wi.normalize();
    r_wi = q_wi.toRotationMatrix();
    t_wi = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);

    const Eigen::Vector3d p_i = r_li_ * p_l + t_li_;
    p_w = r_wi * p_i + t_wi;
    return true;
  }

  void publishColoredScan(const pcl::PointCloud<pcl::PointXYZRGB> &scan, const rclcpp::Time &stamp) {
    sensor_msgs::msg::PointCloud2 colored_scan_msg;
    pcl::toROSMsg(scan, colored_scan_msg);
    colored_scan_msg.header.stamp = stamp;
    colored_scan_msg.header.frame_id = world_frame_;
    colored_scan_pub_->publish(colored_scan_msg);
  }

  void livoxCallback(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr &msg) {
    cv_bridge::CvImagePtr rgb_cv_ptr;
    cv_bridge::CvImageConstPtr depth_cv_ptr;
    nav_msgs::msg::Odometry::ConstSharedPtr odom_msg;
    const rclcpp::Time stamp(msg->header.stamp);

    if (!prepareSynchronizedData(stamp, rgb_cv_ptr, depth_cv_ptr, odom_msg)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "prepareSynchronizedData failed for Livox msg.");
      return;
    }

    const cv::Mat &rgb_img = rgb_cv_ptr->image;
    const cv::Mat &depth_img = depth_cv_ptr->image;

    pcl::PointCloud<pcl::PointXYZRGB> colored_scan;
    colored_scan.reserve(msg->point_num);

    Eigen::Matrix3d r_wi;
    Eigen::Vector3d t_wi;

    int accepted_points = 0;
    for (const auto &pt : msg->points) {
      if (filter_livox_tags_) {
        const bool valid_tag = ((pt.tag & 0x30) == 0x10) || ((pt.tag & 0x30) == 0x00);
        if (!valid_tag) {
          continue;
        }
      }

      const Eigen::Vector3d p_l(pt.x, pt.y, pt.z);

      std::uint8_t r = 0;
      std::uint8_t g = 0;
      std::uint8_t b = 0;
      if (!projectAndColorPoint(p_l, rgb_img, depth_img, depth_cv_ptr->encoding, r, g, b)) {
        continue;
      }

      Eigen::Vector3d p_w;
      if (!computeWorldFromLidar(p_l, odom_msg, p_w, r_wi, t_wi)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Invalid odometry quaternion.");
        return;
      }

      pcl::PointXYZRGB colored_point;
      colored_point.x = static_cast<float>(p_w.x());
      colored_point.y = static_cast<float>(p_w.y());
      colored_point.z = static_cast<float>(p_w.z());
      colored_point.r = r;
      colored_point.g = g;
      colored_point.b = b;

      colored_scan.push_back(colored_point);
      updateVoxelMap(colored_point);
      ++accepted_points;
    }

    publishColoredScan(colored_scan, stamp);

    if (accepted_points > 0) {
      ++cloud_count_;
      if (cloud_count_ % std::max(1, publish_map_every_n_) == 0) {
        publishColoredMap(stamp);
      }
    }
  }

  void cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &cloud_msg) {
    cv_bridge::CvImagePtr rgb_cv_ptr;
    cv_bridge::CvImageConstPtr depth_cv_ptr;
    nav_msgs::msg::Odometry::ConstSharedPtr odom_msg;
    const rclcpp::Time stamp(cloud_msg->header.stamp);

    if (!prepareSynchronizedData(stamp, rgb_cv_ptr, depth_cv_ptr, odom_msg)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "prepareSynchronizedData failed for PointCloud2 msg.");
      return;
    }

    const auto &pose = odom_msg->pose.pose;
    Eigen::Quaterniond q_wi(pose.orientation.w, pose.orientation.x,
                            pose.orientation.y, pose.orientation.z);
    if (q_wi.norm() < 1e-6) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "Invalid odometry quaternion.");
      return;
    }
    q_wi.normalize();

    const Eigen::Matrix3d r_wi = q_wi.toRotationMatrix();
    const Eigen::Matrix3d r_iw = r_wi.transpose();
    const Eigen::Vector3d t_wi(pose.position.x, pose.position.y, pose.position.z);

    pcl::PointCloud<pcl::PointXYZI> cloud_world;
    pcl::fromROSMsg(*cloud_msg, cloud_world);

    const cv::Mat &rgb_img = rgb_cv_ptr->image;
    const cv::Mat &depth_img = depth_cv_ptr->image;

    pcl::PointCloud<pcl::PointXYZRGB> colored_scan;
    colored_scan.reserve(cloud_world.size());

    int accepted_points = 0;
    for (const auto &pw : cloud_world.points) {
      const Eigen::Vector3d p_w(pw.x, pw.y, pw.z);
      const Eigen::Vector3d p_i = r_iw * (p_w - t_wi);
      const Eigen::Vector3d p_l = r_li_.transpose() * (p_i - t_li_);

      std::uint8_t r = 0;
      std::uint8_t g = 0;
      std::uint8_t b = 0;
      if (!projectAndColorPoint(p_l, rgb_img, depth_img, depth_cv_ptr->encoding, r, g, b)) {
        continue;
      }

      pcl::PointXYZRGB colored_point;
      colored_point.x = pw.x;
      colored_point.y = pw.y;
      colored_point.z = pw.z;
      colored_point.r = r;
      colored_point.g = g;
      colored_point.b = b;

      colored_scan.push_back(colored_point);
      updateVoxelMap(colored_point);
      ++accepted_points;
    }

    publishColoredScan(colored_scan, stamp);

    if (accepted_points > 0) {
      ++cloud_count_;
      if (cloud_count_ % std::max(1, publish_map_every_n_) == 0) {
        publishColoredMap(stamp);
      }
    }
  }

  InputMode input_mode_{InputMode::LIVOX_CUSTOM};

  std::string cloud_topic_;
  std::string livox_topic_;
  std::string odom_topic_;
  std::string rgb_topic_;
  std::string depth_topic_;
  std::string camera_info_topic_;
  std::string colored_scan_topic_;
  std::string colored_map_topic_;
  std::string world_frame_;

  double fx_{0.0};
  double fy_{0.0};
  double cx_{0.0};
  double cy_{0.0};

  double max_image_time_diff_{0.05};
  double max_depth_time_diff_{0.05};
  double max_odom_time_diff_{0.02};

  bool use_camera_info_{true};
  bool has_intrinsics_{false};
  bool use_depth_gate_{true};
  bool invert_extrinsic_{false};
  bool filter_livox_tags_{true};

  double depth_tolerance_{0.5};
  double depth_scale_{0.001};
  double min_depth_{0.1};
  double max_depth_{80.0};

  double map_voxel_size_{0.10};
  int publish_map_every_n_{5};
  int max_queue_size_{100};
  int cloud_count_{0};

  Eigen::Matrix3d r_cl_{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d t_cl_{Eigen::Vector3d::Zero()};

  Eigen::Matrix3d r_li_{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d t_li_{Eigen::Vector3d::Zero()};

  std::mutex data_mutex_;
  std::deque<sensor_msgs::msg::CompressedImage::ConstSharedPtr> rgb_queue_;
  std::deque<sensor_msgs::msg::CompressedImage::ConstSharedPtr> depth_queue_;
  std::deque<nav_msgs::msg::Odometry::ConstSharedPtr> odom_queue_;

  std::unordered_map<VoxelKey, VoxelValue, VoxelKeyHasher> voxel_map_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr livox_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr rgb_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr colored_scan_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr colored_map_pub_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LidarCameraColorizerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
