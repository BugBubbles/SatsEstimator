#pragma once

#include "spatiotemporal_vo/common.hpp"
#include "spatiotemporal_vo/config.hpp"
#include <vector>
#include <utility>
#include <Eigen/Core>
#include <pcl/kdtree/kdtree_flann.h>

namespace spatiotemporal_vo
{

  struct RegistrationStats
  {
    double loss = 0.0;
    int num_inliers = 0;
    int num_iterations = 0;
  };

  struct RegistrationOutput
  {
    Eigen::Vector3d twist;
    RegistrationStats stats;
    std::pair<double, double> timespan;

    RegistrationOutput() : twist(Eigen::Vector3d::Zero()), timespan(0.0, 0.0) {}
  };

  class SpatioTemporalRegistration
  {
  public:
    SpatioTemporalRegistration(double fx, double fy, double cx, double cy, const Config &cfg);

    RegistrationOutput run(
        const std::vector<Eigen::Vector2d> &events_xy,
        const std::vector<double> &events_t,
        const std::vector<int> &polarity,
        const std::vector<double> &depth = {});
    void optimizeWithCeres(
        const std::vector<Eigen::Vector3d> &pts_cam_a,
        const std::vector<Eigen::Vector3d> &pts_cam_b,
        Eigen::Vector3d &xi, double &final_loss);

    int count_ = 1;

  private:
    // 时空邻域结构
    struct TemporalNeighborhood
    {
      pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
      std::vector<int> original_indices;
      std::vector<Eigen::Vector3d> points_3d;
    };

    // 辅助函数
    Eigen::Vector2d projectPoint(const Eigen::Vector3d &p);
    Eigen::Vector3d backprojectPoint(const Eigen::Vector2d &uv, double d);
    std::vector<Eigen::Vector3d> backprojectPoints(
        const std::vector<Eigen::Vector2d> &uvs,
        const std::vector<double> &depths);
    Eigen::Matrix3d computeJacobian(const Eigen::Vector3d &p_transformed);
    std::vector<TemporalNeighborhood> buildTemporalNeighborhoods(
        const std::vector<double> &ts_a,
        const std::vector<double> &ts_b,
        const std::vector<Eigen::Vector3d> &pts_cam_b,
        double dt);

    // 相机内参
    double fx_, fy_, cx_, cy_;

    // 配置参数
    double epsilon_T_;
    int max_iters_;
    double lr_;
    double trim_ratio_;
    double convergence_thresh_;
  };
  
  template <typename T>
  Eigen::Matrix<T, 3, 3> skewSymmetric(const Eigen::Matrix<T, 3, 1> &v);
} // namespace spatiotemporal_vo
