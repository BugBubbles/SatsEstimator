#include "spatiotemporal_vo/common.hpp"
#include "methods/cmax_estimator.hpp"
#include <sophus/so3.hpp>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <iostream>
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <cmath>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace spatiotemporal_vo
{
  // 继承 FirstOrderFunction，用于通用函数优化
  class CMaxFunction : public ceres::FirstOrderFunction
  {
  private:
    std::vector<Vec3> points_;
    int width_, height_;
    Eigen::Matrix3d intrinsic_;
    std::vector<double> delta_time_;
    std::vector<int> polarity_;

  public:
    CMaxFunction(const std::vector<Vec3> &points,
                 const std::vector<double> &delta_time,
                 const std::vector<int> &polarity,
                 int width, int height,
                 const Eigen::Matrix3d &K)
        : points_(points),
          delta_time_(delta_time),
          intrinsic_(K),
          width_(width),
          height_(height),
          polarity_(polarity)
    {
    }
    virtual ~CMaxFunction() = default;
    virtual bool Evaluate(const double *parameters,
                          double *cost,
                          double *gradient) const override
    {
      int n_points = points_.size();

      Eigen::Matrix<double, 3, 1> delta_points_T, delta_second_points_T;
      Eigen::Matrix<double, 3, 1> points_T;
      // 创建翘曲事件图像
      cv::Mat warped_image = cv::Mat::zeros(height_, width_, CV_64F);

      // 构建二维PCL点云
      pcl::PointCloud<pcl::PointXY>::Ptr cloud(new pcl::PointCloud<pcl::PointXY>());
      cloud->resize(n_points);
      for (int i = 0; i < n_points; ++i)
      {
        auto point_ = points_[i];
        // Taylor 二阶展开: points * skew_matrix * skew_matrix
        delta_points_T(0) = -parameters[2] * point_(1) + parameters[1] * point_(2);
        delta_points_T(1) = parameters[2] * point_(0) - parameters[0] * point_(2);
        delta_points_T(2) = -parameters[1] * point_(0) + parameters[0] * point_(1);

        delta_second_points_T(0) = -parameters[2] * delta_points_T(1) + parameters[1] * delta_points_T(2);
        delta_second_points_T(1) = parameters[2] * delta_points_T(0) - parameters[0] * delta_points_T(2);
        delta_second_points_T(2) = -parameters[1] * delta_points_T(0) + parameters[0] * delta_points_T(1);
        // 计算 warp 后的 3D 点
        double dt_early = delta_time_[i];
        double dt_sq = 0.5 * dt_early * dt_early;
        points_T(0) = point_(0) + delta_points_T(0) * dt_early + delta_second_points_T(0) * dt_sq;
        points_T(1) = point_(1) + delta_points_T(1) * dt_early + delta_second_points_T(1) * dt_sq;
        points_T(2) = point_(2) + delta_points_T(2) * dt_early + delta_second_points_T(2) * dt_sq;

        // 投影到 2D
        (*cloud)[i].x = points_T(0) / points_T(2) * intrinsic_(0, 0);
        (*cloud)[i].y = points_T(1) / points_T(2) * intrinsic_(1, 1);
        // 使用双线性插值将当前事件翘曲至图像
        int x = static_cast<int>((*cloud)[i].x + intrinsic_(0, 2));
        int y = static_cast<int>((*cloud)[i].y + intrinsic_(1, 2));
        double dx = ((*cloud)[i].x + intrinsic_(0, 2)) - x;
        double dy = ((*cloud)[i].y + intrinsic_(1, 2)) - y;

        if (x >= 0 && x < width_ - 1 && y >= 0 && y < height_ - 1)
        {
          double w1 = (1.0 - dx) * (1.0 - dy);
          double w2 = dx * (1.0 - dy);
          double w3 = (1.0 - dx) * dy;
          double w4 = dx * dy;
          warped_image.at<double>(y, x) += polarity_[i] * w1;
          warped_image.at<double>(y, x + 1) += polarity_[i] * w2;
          warped_image.at<double>(y + 1, x) += polarity_[i] * w3;
          warped_image.at<double>(y + 1, x + 1) += polarity_[i] * w4;
        }
      }
      // 平滑翘曲图像
      cv::GaussianBlur(warped_image, warped_image, cv::Size(3, 3), 1.0);
      // 计算残差（方差）
      double mean = cv::mean(warped_image)[0];
      double var = 0.0;
      for (int r = 0; r < warped_image.rows; ++r)
      {
        for (int c = 0; c < warped_image.cols; ++c)
        {
          double val = warped_image.at<double>(r, c);
          var += val * val;
        }
      }
      var = var / (width_ * height_) - mean * mean;
      cost[0] = -var;

      // 构建KD树
      pcl::KdTreeFLANN<pcl::PointXY> kdtree;
      kdtree.setInputCloud(cloud);

      const float radius = 3.0f;

      for (int k = 0; k < n_points; ++k)
      {
        std::vector<int> neighbor_indices;
        std::vector<float> neighbor_dists;
        pcl::PointXY search_point = (*cloud)[k];
        // 半径搜索
        kdtree.radiusSearch(search_point, radius, neighbor_indices, neighbor_dists);

        for (size_t idx = 0; idx < neighbor_indices.size(); ++idx)
        {
          int l = neighbor_indices[idx];
          if (l == k)
            continue;
          float dist2 = neighbor_dists[idx];
          double weight = std::exp(-dist2 * dist2 / 4.0) * delta_time_[k];
          auto k1 = points_[k](0);
          auto k2 = points_[k](1);
          auto l1 = points_[l](0);
          auto l2 = points_[l](1);
          gradient[0] += weight * ((intrinsic_(1, 1) + k2 * k2 / intrinsic_(1, 1)) * (-k2 + l2) + k1 * k2 * (-k1 + l1) / intrinsic_(1, 1));
          gradient[1] += weight * ((-intrinsic_(0, 0) - k1 * k1 / intrinsic_(0, 0)) * (-k1 + l1) - k1 * k2 * (-k2 + l2) / intrinsic_(0, 0));
          gradient[2] += weight * (-k1 * (-k2 + l2) + k2 * (-k1 + l1));
        }
      }
      gradient[0] /= n_points;
      gradient[1] /= n_points;
      gradient[2] /= n_points;
      return true;
    }

    virtual int NumParameters() const override { return 3; }
  };

  CMaxEstimator::CMaxEstimator(double fx, double fy, double cx, double cy,
                               int width, int height, const Config &config)
      : width_(width), height_(height), count_(1)
  {
    // 初始化相机内参矩阵
    K_ << fx, 0, cx,
        0, fy, cy,
        0, 0, 1;

    // 初始化估计状态
    est_angleAxis_.setZero();
    last_est_var_.setZero();
  }

  void CMaxEstimator::projectEventsTo3D(const std::vector<Vec2> &xy,
                                        std::vector<Vec3> &coord_3d,
                                        const std::vector<double> &depth)
  {
    int n = xy.size();
    coord_3d.resize(n);

    double fx = K_(0, 0), fy = K_(1, 1);
    double cx = K_(0, 2), cy = K_(1, 2);

    for (int i = 0; i < n; ++i)
    {
      double x = (xy[i](0) - cx) / fx;
      double y = (xy[i](1) - cy) / fy;
      coord_3d[i] = Vec3(x, y, depth.empty() ? 1.0 : depth[i]);
      coord_3d[i].normalize();
    }
  }

  void CMaxEstimator::getSampledIndices(std::vector<int> &indices, int count,
                                        double start_ratio, double end_ratio)
  {
    int start_idx = static_cast<int>(eventBundle_.size * start_ratio);
    int end_idx = static_cast<int>(eventBundle_.size * end_ratio);
    int range = end_idx - start_idx;

    if (range <= 0 || count <= 0)
    {
      indices.clear();
      return;
    }

    count = std::min(count, range);
    indices.resize(count);

    // 随机采样
    std::vector<int> all_indices(range);
    std::iota(all_indices.begin(), all_indices.end(), start_idx);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(all_indices.begin(), all_indices.end(), gen);

    std::copy(all_indices.begin(), all_indices.begin() + count, indices.begin());
    std::sort(indices.begin(), indices.end());
  }

  double CMaxEstimator::optimizeMotion()
  {
    // ... 参数初始化 ...
    double angleAxis[3] = {est_angleAxis_(0), est_angleAxis_(1), est_angleAxis_(2)};

    // 1. 构建 GradientProblem
    ceres::GradientProblem problem(new CMaxFunction(
        event_undis_Bundle_.coord_3d,
        eventBundle_.time_delta,
        eventBundle_.polarity,
        width_, height_,
        K_));

    // 2. 配置 Solver
    ceres::GradientProblemSolver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.line_search_direction_type = ceres::BFGS; // 推荐用于通用优化
    options.max_num_iterations = 100;

    ceres::GradientProblemSolver::Summary summary;

    // 3. 求解
    ceres::Solve(options, problem, angleAxis, &summary);

    std::cout << summary.FullReport() << std::endl;

    // ... 更新结果 ...
    est_angleAxis_(0) = angleAxis[0];
    est_angleAxis_(1) = angleAxis[1];
    est_angleAxis_(2) = angleAxis[2];
#ifdef DEBUG
    std::cout << "Final estimated angular velocity: " << est_angleAxis_.transpose() << std::endl;
#endif
    return summary.final_cost;
  }

  RegistrationOutput CMaxEstimator::run(const std::vector<Vec2> &xy,
                                        const std::vector<double> &timestamps,
                                        const std::vector<int> &polarity,
                                        const std::vector<double> &depth)
  {
    RegistrationOutput result;
    result.timespan = {timestamps.front(), timestamps.back()};

    if (xy.empty())
    {
      result.twist.setZero();
      result.stats.loss = 0;
      result.stats.num_iterations = 0;
      return result;
    }

    // 初始化事件数据
    eventBundle_.size = xy.size();
    eventBundle_.coord = xy;
    eventBundle_.polarity = polarity;
    eventBundle_.first_tstamp = timestamps.front();
    eventBundle_.last_tstamp = timestamps.back();

    // 计算相对时间
    eventBundle_.time_delta.resize(eventBundle_.size);
    for (int i = 0; i < eventBundle_.size; ++i)
    {
      eventBundle_.time_delta[i] = timestamps[i] - timestamps.front();
    }

    // 投影到 3D
    event_undis_Bundle_.size = eventBundle_.size;
    projectEventsTo3D(xy, event_undis_Bundle_.coord_3d, depth);

    std::cout << '[' << count_ << "] " << "CMax: Processing "
              << eventBundle_.size << " events, duration: "
              << (eventBundle_.last_tstamp - eventBundle_.first_tstamp) << "s" << std::endl;

    // 运行优化
    result.stats.loss = optimizeMotion();

    // 填充结果
    result.twist = est_angleAxis_;
    result.stats.num_iterations = count_;

    std::cout << '[' << count_ << "] " << "CMax: Estimated angular velocity: "
              << est_angleAxis_.transpose()
              << ", best loss: " << result.stats.loss << std::endl;
    count_++;
    return result;
  }

} // namespace spatiotemporal_vo