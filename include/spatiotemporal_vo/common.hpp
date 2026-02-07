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
    Vec3 trans = Vec3::Zero();
    Vec3 twist = Vec3::Zero();
    RegistrationStats stats;
    std::pair<double, double> timespan = {0.0, 0.0};
  };
  
    /**
   * @brief 事件数据结构
   */
  struct EventPacket
  {
    std::vector<Vec2> xy; // 事件坐标
    std::vector<double> t;           // 时间戳
    std::vector<int> polarity;       // 极性 (1 或 -1)
    std::vector<double> depth;       // 深度值 (可选)
    std::vector<Vec3> coord_3d;     // 3D坐标

    size_t size() const { return xy.size(); }
    bool empty() const { return xy.empty(); }
    double timespan() const { return t.back() - t.front(); };
    void clear()
    {
      xy.clear();
      t.clear();
      polarity.clear();
      coord_3d.clear();
      depth.clear();
    }
    EventPacket copy() const
    {
      EventPacket packet;
      packet.xy = xy;
      packet.t = t;
      packet.polarity = polarity;
      packet.coord_3d = coord_3d;
      packet.depth = depth;
      return packet;
    }
    EventPacket operator+=(const EventPacket &other)
    {
      xy.insert(xy.end(), other.xy.begin(), other.xy.end());
      t.insert(t.end(), other.t.begin(), other.t.end());
      polarity.insert(polarity.end(), other.polarity.begin(), other.polarity.end());
      coord_3d.insert(coord_3d.end(), other.coord_3d.begin(), other.coord_3d.end());
      depth.insert(depth.end(), other.depth.begin(), other.depth.end());
      return *this;
    }
  };
} // namespace spatiotemporal_vo
