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
#include <std_msgs/msg/int32_multi_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

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
    declare_parameter<std::string>("color_mapping.yolo_detections_topic", "/camera/color/image_raw/hailo_yolo/detections");

    declare_parameter<std::string>("color_mapping.colored_scan_topic", "/cloud_registered_color");
    declare_parameter<std::string>("color_mapping.colored_map_topic", "/laser_map_color");

    declare_parameter<double>("color_mapping.fx", 0.0);
    declare_parameter<double>("color_mapping.fy", 0.0);
    declare_parameter<double>("color_mapping.cx", 0.0);
    declare_parameter<double>("color_mapping.cy", 0.0);
    declare_parameter<double>("color_mapping.depth_fx", 0.0);
    declare_parameter<double>("color_mapping.depth_fy", 0.0);
    declare_parameter<double>("color_mapping.depth_cx", 0.0);
    declare_parameter<double>("color_mapping.depth_cy", 0.0);
    declare_parameter<bool>("color_mapping.use_camera_info", true);
    declare_parameter<bool>("color_mapping.use_depth_to_color_extrinsics", true);
    declare_parameter<std::vector<double>>(
      "color_mapping.depth_to_color_rotation",
      std::vector<double>{
        0.999988317489624, -0.004466008860617876, -0.0018516527488827705,
        0.004464889410883188, 0.9999898672103882, -0.0006083904881961644,
        0.0018543510232120752, 0.0006001159781590104, 0.9999980926513672});
    declare_parameter<std::vector<double>>(
      "color_mapping.depth_to_color_translation",
      std::vector<double>{
        -0.023821550369262694,
        -0.00010778414458036422,
        0.0001281014233827591});

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
    declare_parameter<int>("color_mapping.debug_log_every_n_frames", 10);
    declare_parameter<bool>("color_mapping.filter_livox_tags", true);
    declare_parameter<std::string>("color_mapping.world_frame", "world");
    declare_parameter<bool>("color_mapping.enable_yolo_lidar_fusion", true);
    declare_parameter<double>("color_mapping.max_yolo_time_diff", 0.1);
    declare_parameter<int>("color_mapping.min_points_per_3d_bbox", 8);
    declare_parameter<int>("color_mapping.log_3d_target_every_n_frames", 5);
    declare_parameter<bool>("color_mapping.publish_yolo_3d_markers", true);
    declare_parameter<std::string>("color_mapping.yolo_3d_markers_topic", "/yolo_3d/markers");
    declare_parameter<double>("color_mapping.yolo_marker_lifetime_sec", 0.25);

    std::string input_source;
    get_parameter("color_mapping.input_source", input_source);
    get_parameter("color_mapping.cloud_topic", cloud_topic_);
    get_parameter("color_mapping.livox_topic", livox_topic_);
    get_parameter("color_mapping.odom_topic", odom_topic_);
    get_parameter("color_mapping.rgb_topic", rgb_topic_);
    get_parameter("color_mapping.depth_topic", depth_topic_);
    get_parameter("color_mapping.camera_info_topic", camera_info_topic_);
    get_parameter("color_mapping.yolo_detections_topic", yolo_detections_topic_);
    get_parameter("color_mapping.colored_scan_topic", colored_scan_topic_);
    get_parameter("color_mapping.colored_map_topic", colored_map_topic_);
    get_parameter("color_mapping.fx", fx_);
    get_parameter("color_mapping.fy", fy_);
    get_parameter("color_mapping.cx", cx_);
    get_parameter("color_mapping.cy", cy_);
    get_parameter("color_mapping.depth_fx", depth_fx_);
    get_parameter("color_mapping.depth_fy", depth_fy_);
    get_parameter("color_mapping.depth_cx", depth_cx_);
    get_parameter("color_mapping.depth_cy", depth_cy_);
    get_parameter("color_mapping.use_camera_info", use_camera_info_);
    get_parameter("color_mapping.use_depth_to_color_extrinsics", use_depth_to_color_extrinsics_);

    std::vector<double> depth_to_color_r;
    std::vector<double> depth_to_color_t;
    get_parameter("color_mapping.depth_to_color_rotation", depth_to_color_r);
    get_parameter("color_mapping.depth_to_color_translation", depth_to_color_t);

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
    get_parameter("color_mapping.debug_log_every_n_frames", debug_log_every_n_frames_);
    get_parameter("color_mapping.filter_livox_tags", filter_livox_tags_);
    get_parameter("color_mapping.world_frame", world_frame_);
    get_parameter("color_mapping.enable_yolo_lidar_fusion", enable_yolo_lidar_fusion_);
    get_parameter("color_mapping.max_yolo_time_diff", max_yolo_time_diff_);
    get_parameter("color_mapping.min_points_per_3d_bbox", min_points_per_3d_bbox_);
    get_parameter("color_mapping.log_3d_target_every_n_frames", log_3d_target_every_n_frames_);
    get_parameter("color_mapping.publish_yolo_3d_markers", publish_yolo_3d_markers_);
    get_parameter("color_mapping.yolo_3d_markers_topic", yolo_3d_markers_topic_);
    get_parameter("color_mapping.yolo_marker_lifetime_sec", yolo_marker_lifetime_sec_);

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

    if (depth_to_color_r.size() == 9) {
      r_cd_ << depth_to_color_r[0], depth_to_color_r[1], depth_to_color_r[2],
          depth_to_color_r[3], depth_to_color_r[4], depth_to_color_r[5],
          depth_to_color_r[6], depth_to_color_r[7], depth_to_color_r[8];
    } else {
      RCLCPP_WARN(get_logger(), "color_mapping.depth_to_color_rotation must contain 9 values. Using identity.");
      r_cd_.setIdentity();
    }

    if (depth_to_color_t.size() == 3) {
      t_cd_ << depth_to_color_t[0], depth_to_color_t[1], depth_to_color_t[2];
    } else {
      RCLCPP_WARN(get_logger(), "color_mapping.depth_to_color_translation must contain 3 values. Using zero.");
      t_cd_.setZero();
    }

    r_dc_ = r_cd_.transpose();
    t_dc_ = -r_dc_ * t_cd_;

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

    if (enable_yolo_lidar_fusion_) {
      yolo_detections_sub_ = create_subscription<std_msgs::msg::Int32MultiArray>(
        yolo_detections_topic_, rclcpp::QoS(10).best_effort(),
        std::bind(&LidarCameraColorizerNode::yoloDetectionsCallback, this, std::placeholders::_1));
    }

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
    if (publish_yolo_3d_markers_) {
      yolo_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
          yolo_3d_markers_topic_, 10);
    }

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
    if (enable_yolo_lidar_fusion_) {
      RCLCPP_INFO(get_logger(), "YOLO-LiDAR fusion enabled, detections topic: %s",
                  yolo_detections_topic_.c_str());
    } else {
      RCLCPP_INFO(get_logger(), "YOLO-LiDAR fusion disabled.");
    }
    if (publish_yolo_3d_markers_) {
      RCLCPP_INFO(get_logger(), "YOLO 3D marker topic: %s (lifetime=%.2fs)",
                  yolo_3d_markers_topic_.c_str(), yolo_marker_lifetime_sec_);
    }

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

  enum class ProjectionRejectReason {
    NONE = 0,
    BEHIND_CAMERA,
    OUT_OF_IMAGE,
    DEPTH_READ_FAIL,
    DEPTH_OUT_OF_RANGE,
    DEPTH_MISMATCH,
  };

  enum class DepthReadRejectReason {
    NONE = 0,
    OUT_OF_BOUNDS,
    BEHIND_DEPTH,
    ZERO_DEPTH_16U,
    INVALID_DEPTH_16U,
    INVALID_DEPTH_32F,
    UNSUPPORTED_ENCODING,
  };

  struct PointDropStats {
    std::uint64_t input_points{0};
    std::uint64_t filtered_by_tag{0};
    std::uint64_t reject_behind_camera{0};
    std::uint64_t reject_behind_depth{0};
    std::uint64_t reject_out_of_image{0};
    std::uint64_t reject_depth_read{0};
    std::uint64_t reject_depth_read_oob{0};
    std::uint64_t reject_depth_read_zero_16u{0};
    std::uint64_t reject_depth_read_invalid_16u{0};
    std::uint64_t reject_depth_read_invalid_32f{0};
    std::uint64_t reject_depth_read_unsupported_encoding{0};
    std::uint64_t reject_depth_range{0};
    std::uint64_t reject_depth_mismatch{0};
    std::uint64_t accepted_points{0};
  };

  struct YoloDetection2D {
    int class_id{-1};
    int score_pct{0};
    int x1{0};
    int y1{0};
    int x2{0};
    int y2{0};
  };

  struct YoloFrameDetections {
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
    std::vector<YoloDetection2D> detections;
  };

  struct BBox3DAccumulator {
    bool initialized{false};
    std::size_t count{0};
    Eigen::Vector3d min_pt{Eigen::Vector3d::Zero()};
    Eigen::Vector3d max_pt{Eigen::Vector3d::Zero()};
    Eigen::Vector3d sum_pt{Eigen::Vector3d::Zero()};

    void add(const Eigen::Vector3d &p_w) {
      if (!initialized) {
        initialized = true;
        min_pt = p_w;
        max_pt = p_w;
      } else {
        min_pt = min_pt.cwiseMin(p_w);
        max_pt = max_pt.cwiseMax(p_w);
      }
      sum_pt += p_w;
      ++count;
    }
  };

  void accumulatePointDropStats(const PointDropStats &frame_stats,
                                PointDropStats &sum_stats) const {
    sum_stats.input_points += frame_stats.input_points;
    sum_stats.filtered_by_tag += frame_stats.filtered_by_tag;
    sum_stats.reject_behind_camera += frame_stats.reject_behind_camera;
    sum_stats.reject_behind_depth += frame_stats.reject_behind_depth;
    sum_stats.reject_out_of_image += frame_stats.reject_out_of_image;
    sum_stats.reject_depth_read += frame_stats.reject_depth_read;
    sum_stats.reject_depth_read_oob += frame_stats.reject_depth_read_oob;
    sum_stats.reject_depth_read_zero_16u += frame_stats.reject_depth_read_zero_16u;
    sum_stats.reject_depth_read_invalid_16u += frame_stats.reject_depth_read_invalid_16u;
    sum_stats.reject_depth_read_invalid_32f += frame_stats.reject_depth_read_invalid_32f;
    sum_stats.reject_depth_read_unsupported_encoding += frame_stats.reject_depth_read_unsupported_encoding;
    sum_stats.reject_depth_range += frame_stats.reject_depth_range;
    sum_stats.reject_depth_mismatch += frame_stats.reject_depth_mismatch;
    sum_stats.accepted_points += frame_stats.accepted_points;
  }

  void logPointDropStats(const char *mode,
                         std::uint64_t frame_index,
                         const PointDropStats &frame_stats,
                         const PointDropStats &sum_stats) const {
    const std::uint64_t frame_proj_drop =
      frame_stats.reject_behind_camera + frame_stats.reject_behind_depth + frame_stats.reject_out_of_image +
        frame_stats.reject_depth_read + frame_stats.reject_depth_range +
        frame_stats.reject_depth_mismatch;

    const std::uint64_t sum_proj_drop =
      sum_stats.reject_behind_camera + sum_stats.reject_behind_depth + sum_stats.reject_out_of_image +
        sum_stats.reject_depth_read + sum_stats.reject_depth_range +
        sum_stats.reject_depth_mismatch;

    const std::uint64_t dominant_drop_count = std::max(
        {sum_stats.filtered_by_tag,
         sum_stats.reject_behind_camera,
       sum_stats.reject_behind_depth,
         sum_stats.reject_out_of_image,
         sum_stats.reject_depth_read,
         sum_stats.reject_depth_range,
         sum_stats.reject_depth_mismatch});

    const char *dominant_drop_stage = "none";
    if (dominant_drop_count > 0) {
      if (dominant_drop_count == sum_stats.filtered_by_tag) {
        dominant_drop_stage = "tag_filter";
      } else if (dominant_drop_count == sum_stats.reject_behind_camera) {
        dominant_drop_stage = "behind_camera";
      } else if (dominant_drop_count == sum_stats.reject_behind_depth) {
        dominant_drop_stage = "behind_depth";
      } else if (dominant_drop_count == sum_stats.reject_out_of_image) {
        dominant_drop_stage = "out_of_image";
      } else if (dominant_drop_count == sum_stats.reject_depth_read) {
        dominant_drop_stage = "depth_read_fail";
      } else if (dominant_drop_count == sum_stats.reject_depth_range) {
        dominant_drop_stage = "depth_out_of_range";
      } else if (dominant_drop_count == sum_stats.reject_depth_mismatch) {
        dominant_drop_stage = "depth_mismatch";
      }
    }

    RCLCPP_INFO(
        get_logger(),
        "[ColorDebug][%s] frame=%llu frame_in=%llu frame_tag_drop=%llu frame_proj_drop=%llu frame_accept=%llu | "
        "sum_in=%llu sum_tag_drop=%llu sum_behind=%llu sum_oob=%llu sum_color_passed=%llu sum_depth_read=%llu "
        "(oob=%llu behind_depth=%llu zero16=%llu invalid16=%llu invalid32=%llu unsup=%llu) "
        "sum_depth_range=%llu sum_depth_mismatch=%llu sum_proj_drop=%llu sum_accept=%llu dominant_drop=%s(%llu)",
        mode,
        static_cast<unsigned long long>(frame_index),
        static_cast<unsigned long long>(frame_stats.input_points),
        static_cast<unsigned long long>(frame_stats.filtered_by_tag),
        static_cast<unsigned long long>(frame_proj_drop),
        static_cast<unsigned long long>(frame_stats.accepted_points),
        static_cast<unsigned long long>(sum_stats.input_points),
        static_cast<unsigned long long>(sum_stats.filtered_by_tag),
        static_cast<unsigned long long>(sum_stats.reject_behind_camera),
        static_cast<unsigned long long>(sum_stats.reject_out_of_image),
        static_cast<unsigned long long>(sum_stats.input_points - sum_stats.filtered_by_tag - sum_stats.reject_behind_camera - sum_stats.reject_out_of_image),
        static_cast<unsigned long long>(sum_stats.reject_depth_read),
        static_cast<unsigned long long>(sum_stats.reject_depth_read_oob),
        static_cast<unsigned long long>(sum_stats.reject_behind_depth),
        static_cast<unsigned long long>(sum_stats.reject_depth_read_zero_16u),
        static_cast<unsigned long long>(sum_stats.reject_depth_read_invalid_16u),
        static_cast<unsigned long long>(sum_stats.reject_depth_read_invalid_32f),
        static_cast<unsigned long long>(sum_stats.reject_depth_read_unsupported_encoding),
        static_cast<unsigned long long>(sum_stats.reject_depth_range),
        static_cast<unsigned long long>(sum_stats.reject_depth_mismatch),
        static_cast<unsigned long long>(sum_proj_drop),
        static_cast<unsigned long long>(sum_stats.accepted_points),
        dominant_drop_stage,
        static_cast<unsigned long long>(dominant_drop_count));
  }

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

    const bool need_depth = use_depth_gate_;
    if (rgb_queue_.empty() || odom_queue_.empty() || (need_depth && depth_queue_.empty())) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Input queue empty near lidar stamp %.3f: rgb=%zu depth=%zu odom=%zu. "
        "Check topic names/types. Current topics: rgb=%s depth=%s odom=%s",
        stamp.seconds(), rgb_queue_.size(), depth_queue_.size(), odom_queue_.size(),
        rgb_topic_.c_str(), depth_topic_.c_str(), odom_topic_.c_str());
    }

    rgb_msg = findClosestByStamp<sensor_msgs::msg::CompressedImage>(rgb_queue_, stamp, max_image_time_diff_);
    if (need_depth) {
      depth_msg = findClosestByStamp<sensor_msgs::msg::CompressedImage>(depth_queue_, stamp, max_depth_time_diff_);
    } else {
      depth_msg.reset();
    }
    odom_msg = findClosestByStamp<nav_msgs::msg::Odometry>(odom_queue_, stamp, max_odom_time_diff_);
    int mask = 0;
    if (!rgb_msg) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
               "No RGB image found for timestamp %.3f", stamp.seconds());
    mask |= 1;
    }
    if (need_depth && !depth_msg) {
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

  void yoloDetectionsCallback(const std_msgs::msg::Int32MultiArray::ConstSharedPtr &msg) {
    ++yolo_frames_received_;
    if (msg->data.size() < 3) {
      ++yolo_frames_malformed_;
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                           "[YOLO2D] malformed message: size=%zu (<3)", msg->data.size());
      return;
    }

    const int sec = msg->data[0];
    const std::uint32_t nanosec = static_cast<std::uint32_t>(std::max(0, msg->data[1]));
    int det_count = msg->data[2];
    if (det_count < 0) {
      det_count = 0;
    }

    const std::size_t expected = 3 + static_cast<std::size_t>(det_count) * 6;
    if (msg->data.size() < expected) {
      ++yolo_frames_malformed_;
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                           "[YOLO2D] malformed payload: size=%zu expected>=%zu det_count=%d",
                           msg->data.size(), expected, det_count);
      return;
    }

    YoloFrameDetections frame;
    frame.stamp = rclcpp::Time(sec, nanosec, RCL_ROS_TIME);
    frame.detections.reserve(static_cast<std::size_t>(det_count));

    std::size_t offset = 3;
    for (int i = 0; i < det_count; ++i) {
      YoloDetection2D det;
      det.class_id = msg->data[offset + 0];
      det.score_pct = std::clamp(msg->data[offset + 1], 0, 100);
      det.x1 = msg->data[offset + 2];
      det.y1 = msg->data[offset + 3];
      det.x2 = msg->data[offset + 4];
      det.y2 = msg->data[offset + 5];
      if (det.x2 > det.x1 && det.y2 > det.y1) {
        frame.detections.push_back(det);
      }
      offset += 6;
    }

    std::lock_guard<std::mutex> lock(data_mutex_);
    yolo_detections_queue_.push_back(std::move(frame));
    while (static_cast<int>(yolo_detections_queue_.size()) > max_queue_size_) {
      yolo_detections_queue_.pop_front();
    }

    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "[YOLO2D] recv=%llu malformed=%llu queue=%zu latest_det=%d",
        static_cast<unsigned long long>(yolo_frames_received_),
        static_cast<unsigned long long>(yolo_frames_malformed_),
        yolo_detections_queue_.size(), det_count);
  }

  bool getSyncedYoloDetections(const rclcpp::Time &stamp, YoloFrameDetections &out_frame) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (yolo_detections_queue_.empty()) {
      ++yolo_sync_miss_count_;
      return false;
    }

    double best_dt = std::numeric_limits<double>::max();
    const YoloFrameDetections *best = nullptr;
    for (auto it = yolo_detections_queue_.rbegin(); it != yolo_detections_queue_.rend(); ++it) {
      const double dt = std::abs((it->stamp - stamp).seconds());
      if (dt < best_dt) {
        best_dt = dt;
        best = &(*it);
      } else if (it->stamp <= stamp) {
        break;
      }
    }

    if (best == nullptr || best_dt > max_yolo_time_diff_) {
      ++yolo_sync_miss_count_;
      RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "[YOLO2D] sync miss: queue=%zu best_dt=%.3f max_dt=%.3f miss=%llu hit=%llu",
          yolo_detections_queue_.size(), best_dt, max_yolo_time_diff_,
          static_cast<unsigned long long>(yolo_sync_miss_count_),
          static_cast<unsigned long long>(yolo_sync_hit_count_));
      return false;
    }

    out_frame = *best;
    ++yolo_sync_hit_count_;
    return true;
  }

  std::size_t getYoloQueueSize() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return yolo_detections_queue_.size();
  }

  bool projectPointToImageNoDepth(const Eigen::Vector3d &p_l,
                                  int image_width,
                                  int image_height,
                                  int &u,
                                  int &v) const {
    const Eigen::Vector3d p_c = r_cl_ * p_l + t_cl_;
    if (p_c.z() <= 0.0) {
      return false;
    }

    const double u_float = fx_ * (p_c.x() / p_c.z()) + cx_;
    const double v_float = fy_ * (p_c.y() / p_c.z()) + cy_;
    u = static_cast<int>(std::lround(u_float));
    v = static_cast<int>(std::lround(v_float));

    if (u < 0 || v < 0 || u >= image_width || v >= image_height) {
      return false;
    }
    return true;
  }

  void logYolo3DTargets(const char *mode,
                        const YoloFrameDetections &frame,
                        const std::vector<BBox3DAccumulator> &accumulators,
                        const rclcpp::Time &stamp,
                        std::uint64_t frame_index) {
    if (log_3d_target_every_n_frames_ <= 0) {
      return;
    }
    if ((frame_index % static_cast<std::uint64_t>(log_3d_target_every_n_frames_)) != 0) {
      return;
    }

    const double dt = std::abs((frame.stamp - stamp).seconds());
    for (std::size_t i = 0; i < frame.detections.size() && i < accumulators.size(); ++i) {
      const auto &acc = accumulators[i];
      if (acc.count < static_cast<std::size_t>(std::max(1, min_points_per_3d_bbox_))) {
        continue;
      }

      const Eigen::Vector3d center = acc.sum_pt / static_cast<double>(acc.count);
      const Eigen::Vector3d size = acc.max_pt - acc.min_pt;
      const auto &det = frame.detections[i];
      RCLCPP_INFO(
          get_logger(),
          "[YOLO3D][%s] dt=%.3f class_id=%d score=%d%% points=%zu center=(%.3f, %.3f, %.3f) size=(%.3f, %.3f, %.3f) min=(%.3f, %.3f, %.3f) max=(%.3f, %.3f, %.3f)",
          mode,
          dt,
          det.class_id,
          det.score_pct,
          acc.count,
          center.x(), center.y(), center.z(),
          size.x(), size.y(), size.z(),
          acc.min_pt.x(), acc.min_pt.y(), acc.min_pt.z(),
          acc.max_pt.x(), acc.max_pt.y(), acc.max_pt.z());
    }
  }

  std::array<float, 3> classColor(int class_id) const {
    const int cid = std::max(0, class_id);
    const float r = static_cast<float>(55 + ((29 * cid + 151) % 200)) / 255.0f;
    const float g = static_cast<float>(55 + ((17 * cid + 71) % 200)) / 255.0f;
    const float b = static_cast<float>(55 + ((37 * cid + 23) % 200)) / 255.0f;
    return {r, g, b};
  }

  void publishYolo3DMarkers(const char *mode,
                            const YoloFrameDetections &frame,
                            const std::vector<BBox3DAccumulator> &accumulators,
                            const rclcpp::Time &stamp) {
    if (!publish_yolo_3d_markers_ || !yolo_marker_pub_) {
      return;
    }

    visualization_msgs::msg::MarkerArray marker_array;

    visualization_msgs::msg::Marker clear_marker;
    clear_marker.header.frame_id = world_frame_;
    clear_marker.header.stamp = stamp;
    clear_marker.ns = "yolo3d";
    clear_marker.id = 0;
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(clear_marker);

    int marker_id = 1;
    for (std::size_t i = 0; i < frame.detections.size() && i < accumulators.size(); ++i) {
      const auto &acc = accumulators[i];
      if (acc.count < static_cast<std::size_t>(std::max(1, min_points_per_3d_bbox_))) {
        continue;
      }

      const Eigen::Vector3d center = acc.sum_pt / static_cast<double>(acc.count);
      const Eigen::Vector3d size = acc.max_pt - acc.min_pt;
      const auto &det = frame.detections[i];
      const auto color = classColor(det.class_id);

      visualization_msgs::msg::Marker box;
      box.header.frame_id = world_frame_;
      box.header.stamp = stamp;
      box.ns = std::string("yolo3d_box_") + mode;
      box.id = marker_id++;
      box.type = visualization_msgs::msg::Marker::CUBE;
      box.action = visualization_msgs::msg::Marker::ADD;
      box.pose.position.x = center.x();
      box.pose.position.y = center.y();
      box.pose.position.z = center.z();
      box.pose.orientation.w = 1.0;
      box.scale.x = std::max(0.05, size.x());
      box.scale.y = std::max(0.05, size.y());
      box.scale.z = std::max(0.05, size.z());
      box.color.r = color[0];
      box.color.g = color[1];
      box.color.b = color[2];
      box.color.a = 0.35f;
      box.lifetime = rclcpp::Duration::from_seconds(std::max(0.05, yolo_marker_lifetime_sec_));
      marker_array.markers.push_back(box);

      visualization_msgs::msg::Marker center_marker;
      center_marker.header.frame_id = world_frame_;
      center_marker.header.stamp = stamp;
      center_marker.ns = std::string("yolo3d_center_") + mode;
      center_marker.id = marker_id++;
      center_marker.type = visualization_msgs::msg::Marker::SPHERE;
      center_marker.action = visualization_msgs::msg::Marker::ADD;
      center_marker.pose.position.x = center.x();
      center_marker.pose.position.y = center.y();
      center_marker.pose.position.z = center.z();
      center_marker.pose.orientation.w = 1.0;
      center_marker.scale.x = 0.12;
      center_marker.scale.y = 0.12;
      center_marker.scale.z = 0.12;
      center_marker.color.r = color[0];
      center_marker.color.g = color[1];
      center_marker.color.b = color[2];
      center_marker.color.a = 0.95f;
      center_marker.lifetime = rclcpp::Duration::from_seconds(std::max(0.05, yolo_marker_lifetime_sec_));
      marker_array.markers.push_back(center_marker);

      visualization_msgs::msg::Marker text;
      text.header.frame_id = world_frame_;
      text.header.stamp = stamp;
      text.ns = std::string("yolo3d_text_") + mode;
      text.id = marker_id++;
      text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text.action = visualization_msgs::msg::Marker::ADD;
      text.pose.position.x = center.x();
      text.pose.position.y = center.y();
      text.pose.position.z = center.z() + std::max(0.4, size.z() * 0.6);
      text.pose.orientation.w = 1.0;
      text.scale.z = 0.22;
      text.color.r = 1.0f;
      text.color.g = 1.0f;
      text.color.b = 1.0f;
      text.color.a = 0.95f;
      text.lifetime = rclcpp::Duration::from_seconds(std::max(0.05, yolo_marker_lifetime_sec_));
      text.text = "id=" + std::to_string(det.class_id) +
                  " s=" + std::to_string(det.score_pct) + "%" +
                  " n=" + std::to_string(acc.count) +
                  " c=(" + std::to_string(center.x()) + "," +
                  std::to_string(center.y()) + "," +
                  std::to_string(center.z()) + ")";
      marker_array.markers.push_back(text);
    }

    yolo_marker_pub_->publish(marker_array);
  }

  bool readDepthMeters(const cv::Mat &depth_image, int u, int v,
                       const std::string &encoding, double &depth_meters,
                       DepthReadRejectReason *reject_reason = nullptr) const {
    if (reject_reason) {
      *reject_reason = DepthReadRejectReason::NONE;
    }

    if (u < 0 || v < 0 || u >= depth_image.cols || v >= depth_image.rows) {
      if (reject_reason) {
        *reject_reason = DepthReadRejectReason::OUT_OF_BOUNDS;
      }
      return false;
    }

    if (encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
      const std::uint16_t value = depth_image.at<std::uint16_t>(v, u);
      if (value == 0) {
        return true;  // high zero proportion is common for valid depth images, so we don't treat it as an error by default
      }
      depth_meters = static_cast<double>(value) * depth_scale_;
      if (!std::isfinite(depth_meters)) {
        if (reject_reason) {
          *reject_reason = DepthReadRejectReason::INVALID_DEPTH_16U;
        }
        return false;
      }
      return true;
    }

    if (encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
      const float value = depth_image.at<float>(v, u);
      if (!std::isfinite(value) || value <= 0.0f) {
        if (reject_reason) {
          *reject_reason = DepthReadRejectReason::INVALID_DEPTH_32F;
        }
        return false;
      }
      depth_meters = static_cast<double>(value);
      return true;
    }

    if (reject_reason) {
      *reject_reason = DepthReadRejectReason::UNSUPPORTED_ENCODING;
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
                            std::uint8_t &b,
                            ProjectionRejectReason *reject_reason = nullptr,
                            DepthReadRejectReason *depth_read_reject_reason = nullptr) const {
    if (reject_reason) {
      *reject_reason = ProjectionRejectReason::NONE;
    }
    if (depth_read_reject_reason) {
      *depth_read_reject_reason = DepthReadRejectReason::NONE;
    }

    const Eigen::Vector3d p_c = r_cl_ * p_l + t_cl_;
    if (p_c.z() <= 0.0) {
      if (reject_reason) {
        *reject_reason = ProjectionRejectReason::BEHIND_CAMERA;
      }
      return false;
    }

    const double u_float = fx_ * (p_c.x() / p_c.z()) + cx_;
    const double v_float = fy_ * (p_c.y() / p_c.z()) + cy_;
    const int u = static_cast<int>(std::lround(u_float));
    const int v = static_cast<int>(std::lround(v_float));

    if (u < 0 || v < 0 || u >= rgb_img.cols || v >= rgb_img.rows) {
      if (reject_reason) {
        *reject_reason = ProjectionRejectReason::OUT_OF_IMAGE;
      }
      return false;
    }

    if (use_depth_gate_) {
      const Eigen::Vector3d p_d = use_depth_to_color_extrinsics_ ? (r_dc_ * p_c + t_dc_) : p_c;
      if (p_d.z() <= 0.0) {
        if (reject_reason) {
          *reject_reason = ProjectionRejectReason::DEPTH_READ_FAIL;
        }
        if (depth_read_reject_reason) {
          *depth_read_reject_reason = DepthReadRejectReason::BEHIND_DEPTH;
        }
        return false;
      }

      const int depth_u = static_cast<int>(std::lround(fx_ * (p_d.x() / p_d.z()) + cx_));
      const int depth_v = static_cast<int>(std::lround(fy_ * (p_d.y() / p_d.z()) + cy_));

      double depth_m = 0.0;
      if (!readDepthMeters(depth_img, depth_u, depth_v, depth_encoding, depth_m, depth_read_reject_reason)) {
        if (reject_reason) {
          *reject_reason = ProjectionRejectReason::DEPTH_READ_FAIL;
        }
        return false;
      }

      if (depth_m == 0.0) {
        return true;  // treat zero depth as valid to avoid excessive rejections, especially for LiDAR points that often project to low-confidence depth pixels
      }
      if (depth_m > max_depth_) {
        if (reject_reason) {
          *reject_reason = ProjectionRejectReason::DEPTH_OUT_OF_RANGE;
        }
        return false;
      }
      if (std::abs(depth_m - p_d.z()) > depth_tolerance_) {
        if (reject_reason) {
          *reject_reason = ProjectionRejectReason::DEPTH_MISMATCH;
        }
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

    if (use_depth_gate_) {
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
    } else {
      depth_cv_ptr.reset();
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
    const cv::Mat depth_img = (depth_cv_ptr ? depth_cv_ptr->image : cv::Mat());
    const std::string depth_encoding = (depth_cv_ptr ? depth_cv_ptr->encoding : std::string());

    int depth_zeros = 0;
    int depth_total = 0;
    if (use_depth_gate_ && !depth_img.empty()) {
      depth_zeros = cv::countNonZero(depth_img == 0);
      depth_total = depth_img.rows * depth_img.cols;
    }
    // RCLCPP_INFO(get_logger(), "Depth image zeros: %d / %d (%.2f%%)", depth_zeros, depth_total, 100.0 * depth_zeros / depth_total);

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
    const Eigen::Vector3d t_wi(pose.position.x, pose.position.y, pose.position.z);
    const Eigen::Matrix3d r_wl = r_wi * r_li_;
    const Eigen::Vector3d t_wl = r_wi * t_li_ + t_wi;

    YoloFrameDetections yolo_frame;
    const bool has_yolo = enable_yolo_lidar_fusion_ && getSyncedYoloDetections(stamp, yolo_frame) && !yolo_frame.detections.empty();
    const double yolo_dt = has_yolo ? std::abs((yolo_frame.stamp - stamp).seconds()) : -1.0;
    std::vector<BBox3DAccumulator> yolo_accumulators;
    if (has_yolo) {
      yolo_accumulators.resize(yolo_frame.detections.size());
    }

    pcl::PointCloud<pcl::PointXYZRGB> colored_scan;
    colored_scan.reserve(msg->point_num);

    PointDropStats frame_stats;
    frame_stats.input_points = static_cast<std::uint64_t>(msg->point_num);

    for (const auto &pt : msg->points) {
      if (filter_livox_tags_) {
        const bool valid_tag = ((pt.tag & 0x30) == 0x10) || ((pt.tag & 0x30) == 0x00);
        if (!valid_tag) {
          ++frame_stats.filtered_by_tag;
          continue;
        }
      }

      const Eigen::Vector3d p_l(pt.x, pt.y, pt.z);

      if (has_yolo) {
        int u = 0;
        int v = 0;
        if (projectPointToImageNoDepth(p_l, rgb_img.cols, rgb_img.rows, u, v)) {
          const Eigen::Vector3d p_w = r_wl * p_l + t_wl;
          for (std::size_t det_idx = 0; det_idx < yolo_frame.detections.size(); ++det_idx) {
            const auto &det = yolo_frame.detections[det_idx];
            if (u >= det.x1 && u <= det.x2 && v >= det.y1 && v <= det.y2) {
              yolo_accumulators[det_idx].add(p_w);
            }
          }
        }
      }

      std::uint8_t r = 0;
      std::uint8_t g = 0;
      std::uint8_t b = 0;
      ProjectionRejectReason reject_reason = ProjectionRejectReason::NONE;
      DepthReadRejectReason depth_read_reject_reason = DepthReadRejectReason::NONE;
      if (!projectAndColorPoint(p_l, rgb_img, depth_img, depth_encoding,
                                r, g, b, &reject_reason, &depth_read_reject_reason)) {
        if (reject_reason == ProjectionRejectReason::BEHIND_CAMERA) {
          ++frame_stats.reject_behind_camera;
        } else if (reject_reason == ProjectionRejectReason::OUT_OF_IMAGE) {
          ++frame_stats.reject_out_of_image;
        } else if (reject_reason == ProjectionRejectReason::DEPTH_READ_FAIL) {
          ++frame_stats.reject_depth_read;
          if (depth_read_reject_reason == DepthReadRejectReason::OUT_OF_BOUNDS) {
            ++frame_stats.reject_depth_read_oob;
          } else if (depth_read_reject_reason == DepthReadRejectReason::BEHIND_DEPTH) {
            ++frame_stats.reject_behind_depth;
          } else if (depth_read_reject_reason == DepthReadRejectReason::ZERO_DEPTH_16U) {
            ++frame_stats.reject_depth_read_zero_16u;
          } else if (depth_read_reject_reason == DepthReadRejectReason::INVALID_DEPTH_16U) {
            ++frame_stats.reject_depth_read_invalid_16u;
          } else if (depth_read_reject_reason == DepthReadRejectReason::INVALID_DEPTH_32F) {
            ++frame_stats.reject_depth_read_invalid_32f;
          } else if (depth_read_reject_reason == DepthReadRejectReason::UNSUPPORTED_ENCODING) {
            ++frame_stats.reject_depth_read_unsupported_encoding;
          }
        } else if (reject_reason == ProjectionRejectReason::DEPTH_OUT_OF_RANGE) {
          ++frame_stats.reject_depth_range;
        } else if (reject_reason == ProjectionRejectReason::DEPTH_MISMATCH) {
          ++frame_stats.reject_depth_mismatch;
        }
        continue;
      }

      const Eigen::Vector3d p_w = r_wl * p_l + t_wl;

      pcl::PointXYZRGB colored_point;
      colored_point.x = static_cast<float>(p_w.x());
      colored_point.y = static_cast<float>(p_w.y());
      colored_point.z = static_cast<float>(p_w.z());
      colored_point.r = r;
      colored_point.g = g;
      colored_point.b = b;

      colored_scan.push_back(colored_point);
      updateVoxelMap(colored_point);
      ++frame_stats.accepted_points;
    }

    ++livox_frame_count_;

    if (debug_log_every_n_frames_ > 0 &&
      (livox_frame_count_ % static_cast<std::uint64_t>(debug_log_every_n_frames_)) == 0) {
      const std::size_t yolo_queue_size = enable_yolo_lidar_fusion_ ? getYoloQueueSize() : 0;
      RCLCPP_INFO(
        get_logger(),
        "[FusionState][livox] frame=%llu has_yolo=%d yolo_det=%zu yolo_dt=%.3f queue=%zu sync_hit=%llu sync_miss=%llu",
        static_cast<unsigned long long>(livox_frame_count_), has_yolo ? 1 : 0,
        has_yolo ? yolo_frame.detections.size() : 0, yolo_dt, yolo_queue_size,
        static_cast<unsigned long long>(yolo_sync_hit_count_),
        static_cast<unsigned long long>(yolo_sync_miss_count_));
    }

    accumulatePointDropStats(frame_stats, livox_sum_stats_);
    if (debug_log_every_n_frames_ > 0 &&
        (livox_frame_count_ % static_cast<std::uint64_t>(debug_log_every_n_frames_)) == 0) {
      logPointDropStats("livox", livox_frame_count_, frame_stats, livox_sum_stats_);
    }

    if (has_yolo) {
      logYolo3DTargets("livox", yolo_frame, yolo_accumulators, stamp, livox_frame_count_);
    }
    publishYolo3DMarkers("livox", yolo_frame, yolo_accumulators, stamp);

    publishColoredScan(colored_scan, stamp);

    if (frame_stats.accepted_points > 0) {
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
    const cv::Mat depth_img = (depth_cv_ptr ? depth_cv_ptr->image : cv::Mat());
    const std::string depth_encoding = (depth_cv_ptr ? depth_cv_ptr->encoding : std::string());

    YoloFrameDetections yolo_frame;
    const bool has_yolo = enable_yolo_lidar_fusion_ && getSyncedYoloDetections(stamp, yolo_frame) && !yolo_frame.detections.empty();
    const double yolo_dt = has_yolo ? std::abs((yolo_frame.stamp - stamp).seconds()) : -1.0;
    std::vector<BBox3DAccumulator> yolo_accumulators;
    if (has_yolo) {
      yolo_accumulators.resize(yolo_frame.detections.size());
    }

    pcl::PointCloud<pcl::PointXYZRGB> colored_scan;
    colored_scan.reserve(cloud_world.size());

    PointDropStats frame_stats;
    frame_stats.input_points = static_cast<std::uint64_t>(cloud_world.size());

    for (const auto &pw : cloud_world.points) {
      const Eigen::Vector3d p_w(pw.x, pw.y, pw.z);
      const Eigen::Vector3d p_i = r_iw * (p_w - t_wi);
      const Eigen::Vector3d p_l = r_li_.transpose() * (p_i - t_li_);

      if (has_yolo) {
        int u = 0;
        int v = 0;
        if (projectPointToImageNoDepth(p_l, rgb_img.cols, rgb_img.rows, u, v)) {
          for (std::size_t det_idx = 0; det_idx < yolo_frame.detections.size(); ++det_idx) {
            const auto &det = yolo_frame.detections[det_idx];
            if (u >= det.x1 && u <= det.x2 && v >= det.y1 && v <= det.y2) {
              yolo_accumulators[det_idx].add(p_w);
            }
          }
        }
      }

      std::uint8_t r = 0;
      std::uint8_t g = 0;
      std::uint8_t b = 0;
      ProjectionRejectReason reject_reason = ProjectionRejectReason::NONE;
      DepthReadRejectReason depth_read_reject_reason = DepthReadRejectReason::NONE;
      if (!projectAndColorPoint(p_l, rgb_img, depth_img, depth_encoding,
                                r, g, b, &reject_reason, &depth_read_reject_reason)) {
        if (reject_reason == ProjectionRejectReason::BEHIND_CAMERA) {
          ++frame_stats.reject_behind_camera;
        } else if (reject_reason == ProjectionRejectReason::OUT_OF_IMAGE) {
          ++frame_stats.reject_out_of_image;
        } else if (reject_reason == ProjectionRejectReason::DEPTH_READ_FAIL) {
          ++frame_stats.reject_depth_read;
          if (depth_read_reject_reason == DepthReadRejectReason::OUT_OF_BOUNDS) {
            ++frame_stats.reject_depth_read_oob;
          } else if (depth_read_reject_reason == DepthReadRejectReason::BEHIND_DEPTH) {
            ++frame_stats.reject_behind_depth;
          } else if (depth_read_reject_reason == DepthReadRejectReason::ZERO_DEPTH_16U) {
            ++frame_stats.reject_depth_read_zero_16u;
          } else if (depth_read_reject_reason == DepthReadRejectReason::INVALID_DEPTH_16U) {
            ++frame_stats.reject_depth_read_invalid_16u;
          } else if (depth_read_reject_reason == DepthReadRejectReason::INVALID_DEPTH_32F) {
            ++frame_stats.reject_depth_read_invalid_32f;
          } else if (depth_read_reject_reason == DepthReadRejectReason::UNSUPPORTED_ENCODING) {
            ++frame_stats.reject_depth_read_unsupported_encoding;
          }
        } else if (reject_reason == ProjectionRejectReason::DEPTH_OUT_OF_RANGE) {
          ++frame_stats.reject_depth_range;
        } else if (reject_reason == ProjectionRejectReason::DEPTH_MISMATCH) {
          ++frame_stats.reject_depth_mismatch;
        }
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
      ++frame_stats.accepted_points;
    }

    ++cloud_frame_count_;

    if (debug_log_every_n_frames_ > 0 &&
      (cloud_frame_count_ % static_cast<std::uint64_t>(debug_log_every_n_frames_)) == 0) {
      const std::size_t yolo_queue_size = enable_yolo_lidar_fusion_ ? getYoloQueueSize() : 0;
      RCLCPP_INFO(
        get_logger(),
        "[FusionState][cloud] frame=%llu has_yolo=%d yolo_det=%zu yolo_dt=%.3f queue=%zu sync_hit=%llu sync_miss=%llu",
        static_cast<unsigned long long>(cloud_frame_count_), has_yolo ? 1 : 0,
        has_yolo ? yolo_frame.detections.size() : 0, yolo_dt, yolo_queue_size,
        static_cast<unsigned long long>(yolo_sync_hit_count_),
        static_cast<unsigned long long>(yolo_sync_miss_count_));
    }

    accumulatePointDropStats(frame_stats, cloud_sum_stats_);
    if (debug_log_every_n_frames_ > 0 &&
        (cloud_frame_count_ % static_cast<std::uint64_t>(debug_log_every_n_frames_)) == 0) {
      logPointDropStats("cloud", cloud_frame_count_, frame_stats, cloud_sum_stats_);
    }

    if (has_yolo) {
      logYolo3DTargets("cloud", yolo_frame, yolo_accumulators, stamp, cloud_frame_count_);
    }
    publishYolo3DMarkers("cloud", yolo_frame, yolo_accumulators, stamp);

    publishColoredScan(colored_scan, stamp);

    if (frame_stats.accepted_points > 0) {
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
  std::string yolo_detections_topic_;
  std::string yolo_3d_markers_topic_;
  std::string colored_scan_topic_;
  std::string colored_map_topic_;
  std::string world_frame_;

  double fx_{0.0};
  double fy_{0.0};
  double cx_{0.0};
  double cy_{0.0};
  double depth_fx_{0.0};
  double depth_fy_{0.0};
  double depth_cx_{0.0};
  double depth_cy_{0.0};

  double max_image_time_diff_{0.05};
  double max_depth_time_diff_{0.05};
  double max_odom_time_diff_{0.02};
  double max_yolo_time_diff_{0.08};

  bool use_camera_info_{true};
  bool has_intrinsics_{false};
  bool use_depth_gate_{true};
  bool use_depth_to_color_extrinsics_{true};
  bool invert_extrinsic_{false};
  bool filter_livox_tags_{true};
  bool enable_yolo_lidar_fusion_{true};
  bool publish_yolo_3d_markers_{true};

  double depth_tolerance_{0.5};
  double depth_scale_{0.001};
  double min_depth_{0.1};
  double max_depth_{80.0};

  double map_voxel_size_{0.10};
  int publish_map_every_n_{5};
  int max_queue_size_{100};
  int debug_log_every_n_frames_{10};
  int min_points_per_3d_bbox_{8};
  int log_3d_target_every_n_frames_{5};
  double yolo_marker_lifetime_sec_{0.25};
  int cloud_count_{0};
  std::uint64_t livox_frame_count_{0};
  std::uint64_t cloud_frame_count_{0};
  std::uint64_t yolo_frames_received_{0};
  std::uint64_t yolo_frames_malformed_{0};
  std::uint64_t yolo_sync_hit_count_{0};
  std::uint64_t yolo_sync_miss_count_{0};

  PointDropStats livox_sum_stats_;
  PointDropStats cloud_sum_stats_;

  Eigen::Matrix3d r_cl_{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d t_cl_{Eigen::Vector3d::Zero()};

  Eigen::Matrix3d r_cd_{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d t_cd_{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d r_dc_{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d t_dc_{Eigen::Vector3d::Zero()};

  Eigen::Matrix3d r_li_{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d t_li_{Eigen::Vector3d::Zero()};

  std::mutex data_mutex_;
  std::deque<sensor_msgs::msg::CompressedImage::ConstSharedPtr> rgb_queue_;
  std::deque<sensor_msgs::msg::CompressedImage::ConstSharedPtr> depth_queue_;
  std::deque<nav_msgs::msg::Odometry::ConstSharedPtr> odom_queue_;
  std::deque<YoloFrameDetections> yolo_detections_queue_;

  std::unordered_map<VoxelKey, VoxelValue, VoxelKeyHasher> voxel_map_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr livox_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr rgb_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<std_msgs::msg::Int32MultiArray>::SharedPtr yolo_detections_sub_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr colored_scan_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr colored_map_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr yolo_marker_pub_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LidarCameraColorizerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
