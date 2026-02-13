#pragma once

#include "spatiotemporal_vo/common.hpp"
#include "spatiotemporal_vo/config.hpp"
#include <vector>
#include <utility>
#include <Eigen/Core>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>

namespace spatiotemporal_vo
{

  class SpatioTemporalRegistration
  {
  public:
    SpatioTemporalRegistration(
        double fx, double fy, double cx, double cy, const Config &cfg);

    RegistrationOutput<3> run(
        const std::vector<Eigen::Vector2d> &events_xy,
        const std::vector<double> &events_t,
        const std::vector<int> &polarity,
        const std::vector<double> &depth = {});

  void optimizeWithCeres(
      const std::vector<Eigen::Vector3d> &pts_cam_a,
      const std::vector<Eigen::Vector3d> &pts_cam_b,
      const std::vector<int> &correspondences,
      Eigen::Vector3d &xi);

    int count_ = 1;

  private:
    // 时空邻域结构
    struct TemporalNeighborhood
    {
      pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
      std::vector<int> original_indices;
      std::vector<Eigen::Vector3d> points_3d;
    };
    double runICP(
        const std::vector<Eigen::Vector3d> &data_points,
        const std::vector<Eigen::Vector3d> &model_points,
        std::vector<TemporalNeighborhood> &kdtree,
        Eigen::Matrix3d &R_result,
        int max_iterations,
        double trim_fraction);


    Eigen::Vector2d projectPoint(const Eigen::Vector3d &p);
    Eigen::Vector3d backprojectPoint(const Eigen::Vector2d &uv, double d);
    std::vector<Eigen::Vector3d> backprojectPoints(
        const std::vector<Eigen::Vector2d> &uvs,
        const std::vector<double> &depths);
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

} // namespace spatiotemporal_vo
