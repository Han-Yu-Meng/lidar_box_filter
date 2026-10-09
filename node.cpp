/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

#include <fins/node.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#if __has_include(<tf2_eigen/tf2_eigen.hpp>)
  #include <tf2_eigen/tf2_eigen.hpp>
#else
  #include <tf2_eigen/tf2_eigen.h>
#endif

#include "livox_driver2/msg/custom_msg.hpp"
#include <Eigen/Dense>
#include <mutex>

using CustomMsg = livox_driver2::msg::CustomMsg;
using CustomPoint = livox_driver2::msg::CustomPoint;

class LidarBoxFilterNode : public fins::Node {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct Config {
        bool enable_box_filter = true;
        std::string filter_target_frame = "base_lidar";
        double filter_box_min_x = -1.0;
        double filter_box_max_x = 2.0;
        double filter_box_min_y = -1.0;
        double filter_box_max_y = 1.0;
        double filter_box_min_z = -0.5;
        double filter_box_max_z = 1.5;
        double lidar_blind_zone = 0.05;
        bool enable_z_filter = true;
        double filter_z_min = 0.0;
    };

    void define() override {
        set_name("LidarBoxFilter");
        set_category("Driver");

        register_input<CustomMsg>("lidar", &LidarBoxFilterNode::on_lidar_callback);
        register_input<geometry_msgs::msg::TransformStamped>("$T_{target}^{lidar}$", &LidarBoxFilterNode::on_tf_callback);

        register_output<CustomMsg>("filtered_cloud");
        register_output<visualization_msgs::msg::MarkerArray>("filter_box_marker");
    }

    void initialize() override {
        fins::ParamLoader config("LidarBoxFilter");

        config_.enable_box_filter = true;
        config_.filter_target_frame = std::string("base_lidar");
        config_.filter_box_min_x = -0.32;
        config_.filter_box_max_x = 0.32;
        config_.filter_box_min_y = -0.32;
        config_.filter_box_max_y =0.32;
        config_.filter_box_min_z = -0.13;
        config_.filter_box_max_z = 0.36;
        config_.lidar_blind_zone = 0.05;
        config_.enable_z_filter = false;
        config_.filter_z_min = 0.0;

        target_frame_id_ = config_.filter_target_frame;
    }

private:
    void on_lidar_callback(const fins::Msg<CustomMsg>& msg) {
        std::lock_guard<std::mutex> lock(data_mtx_);
        if (!tf_received_) {
            return;
        }
        do_filter(msg.ptr(), msg.acq_time);
    }

    void on_tf_callback(const fins::Msg<geometry_msgs::msg::TransformStamped>& msg) {
        std::lock_guard<std::mutex> lock(data_mtx_);
        tf_lidar_to_target_ = tf2::transformToEigen(*msg);
        target_frame_id_ = msg->header.frame_id;
        tf_received_ = true;
    }

    void do_filter(const CustomMsg::ConstSharedPtr& msg, fins::AcqTime ts) {
        if (msg->points.empty()) return;

        CustomMsg filtered;
        filtered.lidar_id = msg->lidar_id;
        filtered.points.reserve(msg->points.size());

        double blind_sq = config_.lidar_blind_zone * config_.lidar_blind_zone;
        bool check_blind = (config_.lidar_blind_zone > 1e-4);
        bool enable_box = config_.enable_box_filter;
        bool enable_z = config_.enable_z_filter;
        bool need_target_transform = enable_box || enable_z;
        double z_min = config_.filter_z_min;

        auto is_in_box = [&](const Eigen::Vector3d& pt_target) {
            return (pt_target.x() >= config_.filter_box_min_x && pt_target.x() <= config_.filter_box_max_x &&
                    pt_target.y() >= config_.filter_box_min_y && pt_target.y() <= config_.filter_box_max_y &&
                    pt_target.z() >= config_.filter_box_min_z && pt_target.z() <= config_.filter_box_max_z);
        };

        for (const auto& pt : msg->points) {
            if (check_blind && (pt.x * pt.x + pt.y * pt.y + pt.z * pt.z) < blind_sq) continue;

            if (need_target_transform) {
                Eigen::Vector3d pt_target = tf_lidar_to_target_ * Eigen::Vector3d(pt.x, pt.y, pt.z);
                if (enable_box && is_in_box(pt_target)) continue;
                if (enable_z && pt_target.z() < z_min) continue;
            }

            filtered.points.push_back(pt);
        }

        filtered.point_num = filtered.points.size();
        filtered.header.stamp = msg->header.stamp;
        filtered.header.frame_id = msg->header.frame_id;
        filtered.timebase = msg->timebase;

        send("filtered_cloud", filtered, ts);

        if (enable_box && required("filter_box_marker")) {
            publish_filter_box_marker(filtered.header.stamp, ts);
        }
    }

    void publish_filter_box_marker(const builtin_interfaces::msg::Time& stamp, fins::AcqTime ts) {
        visualization_msgs::msg::MarkerArray marker_array;
        visualization_msgs::msg::Marker marker;

        marker.header.frame_id = target_frame_id_;
        marker.header.stamp = stamp;
        marker.ns = "filter_box";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.action = visualization_msgs::msg::Marker::ADD;

        marker.pose.position.x = (config_.filter_box_min_x + config_.filter_box_max_x) / 2.0;
        marker.pose.position.y = (config_.filter_box_min_y + config_.filter_box_max_y) / 2.0;
        marker.pose.position.z = (config_.filter_box_min_z + config_.filter_box_max_z) / 2.0;
        marker.pose.orientation.w = 1.0;

        marker.scale.x = std::abs(config_.filter_box_max_x - config_.filter_box_min_x);
        marker.scale.y = std::abs(config_.filter_box_max_y - config_.filter_box_min_y);
        marker.scale.z = std::abs(config_.filter_box_max_z - config_.filter_box_min_z);

        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 0.4;

        marker.frame_locked = true;

        marker.lifetime.sec = 0;
        marker.lifetime.nanosec = 200000000; // 0.2s

        marker_array.markers.push_back(marker);
        send("filter_box_marker", marker_array, ts);
    }

    Config config_;
    std::mutex data_mtx_;

    std::string target_frame_id_;
    Eigen::Isometry3d tf_lidar_to_target_{Eigen::Isometry3d::Identity()};
    bool tf_received_{false};
};

EXPORT_NODE(LidarBoxFilterNode)
DEFINE_PLUGIN_ENTRY(fins::STATELESS)
