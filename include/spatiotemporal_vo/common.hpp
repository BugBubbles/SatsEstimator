#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <fstream>

namespace spatiotemporal_vo
{

  // 点云类型定义
  using PointT = pcl::PointXYZ;
  using PointCloud = pcl::PointCloud<PointT>;
  using PointCloudPtr = PointCloud::Ptr;
  using PointCloudConstPtr = PointCloud::ConstPtr;

  // 位姿类型定义 (Sophus)
  using SE3 = Sophus::SE3d;
  using SO3 = Sophus::SO3d;
  using Vec2 = Eigen::Vector2d;
  using Vec3 = Eigen::Vector3d;
  using Vec4 = Eigen::Vector4d;
  using Mat3 = Eigen::Matrix3d;
  using EigenMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

  struct RegistrationStats
  {
    double loss = 0.0;
    double time_taken = 0.0;
    int num_inliers = 0;
    int num_iterations = 0;
  };

  struct RegistrationOutput
  {
    Eigen::Vector3d twist = Eigen::Vector3d::Zero();
    RegistrationStats stats;
    std::pair<double, double> timespan = {0.0, 0.0};
  };
} // namespace spatiotemporal_vo
