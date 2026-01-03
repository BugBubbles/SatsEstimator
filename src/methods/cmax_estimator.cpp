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
#include <omp.h>

namespace spatiotemporal_vo
{

  class CMaxOptimizer
  {
  private:
    // 你的数据成员（和原CMaxFunction一致）
    std::vector<Vec3> points_;
    int width_, height_;
    Eigen::Matrix3d intrinsic_;
    std::vector<double> delta_time_;
    std::vector<int> polarity_;
    pcl::PointCloud<pcl::PointXY>::Ptr cloud;
    int n_points;
    const float radius = 4.0f;
    // 添加 final_loss_ 成员
    double final_loss_ = 0.0;
    Vec3 est_angleAxis_;

  public:
    void getEstimatedParams(Vec3 &est_angleAxis) const { est_angleAxis = est_angleAxis_; }
    double getFinalLoss() const { return final_loss_; }
    CMaxOptimizer(const std::vector<Vec3> &points,
                  const std::vector<double> &delta_time,
                  const std::vector<int> &polarity,
                  int width, int height,
                  const Eigen::Matrix3d &K,
                  const Vec3 &initial_angleAxis)
        : points_(points), delta_time_(delta_time), polarity_(polarity),
          width_(width), height_(height), intrinsic_(K), est_angleAxis_(initial_angleAxis)
    {
      n_points = points_.size();
    }

    void compute_loss(const Vec3 &parameters, double &loss)
    {
      int padding = 100;
      Eigen::Matrix<double, 3, 1> delta_points_T, delta_second_points_T;
      Eigen::Matrix<double, 3, 1> points_T;
      // 创建翘曲事件图像
      cv::Mat warped_image = cv::Mat::zeros(height_ + 2 * padding, width_ + 2 * padding, CV_64F);
      // 构建二维PCL点云
      // 释放cloud旧内存
      cloud.reset(new pcl::PointCloud<pcl::PointXY>());
      cloud->resize(n_points);
      for (int i = 0; i < n_points; ++i)
      {
        auto point_ = points_[i];
        // Taylor 二阶展开: points * skew_matrix * skew_matrix
        delta_points_T(0) = -parameters[2] * point_(1) + parameters[1] * point_(2);
        delta_points_T(1) = parameters[2] * point_(0) - parameters[0] * point_(2);
        delta_points_T(2) = -parameters[1] * point_(0) + parameters[0] * point_(1);

        // delta_second_points_T(0) = -parameters[2] * delta_points_T(1) + parameters[1] * delta_points_T(2);
        // delta_second_points_T(1) = parameters[2] * delta_points_T(0) - parameters[0] * delta_points_T(2);
        // delta_second_points_T(2) = -parameters[1] * delta_points_T(0) + parameters[0] * delta_points_T(1);
        // 计算 warp 后的 3D 点
        double dt_early = delta_time_[i];
        // double dt_sq = 0.5 * dt_early * dt_early;
        // points_T(0) = point_(0) + delta_points_T(0) * dt_early + delta_second_points_T(0) * dt_sq;
        // points_T(1) = point_(1) + delta_points_T(1) * dt_early + delta_second_points_T(1) * dt_sq;
        // points_T(2) = point_(2) + delta_points_T(2) * dt_early + delta_second_points_T(2) * dt_sq;

        points_T(0) = point_(0) + delta_points_T(0) * dt_early;
        points_T(1) = point_(1) + delta_points_T(1) * dt_early;
        points_T(2) = point_(2) + delta_points_T(2) * dt_early;
        // 投影到 2D
        (*cloud)[i].x = points_T(0) / points_T(2) * intrinsic_(0, 0) + intrinsic_(0, 2);
        (*cloud)[i].y = points_T(1) / points_T(2) * intrinsic_(1, 1) + intrinsic_(1, 2);
        // 使用双线性插值将当前事件翘曲至图像
        int x = static_cast<int>((*cloud)[i].x);
        int y = static_cast<int>((*cloud)[i].y);
        double dx = (*cloud)[i].x - x;
        double dy = (*cloud)[i].y - y;

        if (x >= -padding && x < width_ + padding - 1 && y >= -padding && y < height_ + padding - 1)
        {
          double w4 = dx * dy;
          double w3 = (1.0 - dx) * dy;
          double w2 = dx * (1.0 - dy);
          double w1 = (1.0 - dx) * (1.0 - dy);

          warped_image.at<double>(y + padding, x + padding) += w1;
          warped_image.at<double>(y + padding, x + 1 + padding) += w2;
          warped_image.at<double>(y + padding + 1, x + padding) += w3;
          warped_image.at<double>(y + padding + 1, x + 1 + padding) += w4;
        }
      }
      // 平滑翘曲图像
      cv::GaussianBlur(warped_image, warped_image, cv::Size(3, 3), 1.0);
      // 计算残差（方差）
      cv::Scalar mean, stddev;
      cv::meanStdDev(warped_image, mean, stddev);
      loss = -stddev[0] * stddev[0];
    }
    void compute_gradient(const Vec3 &parameters, Vec3 &gradient, Eigen::Matrix3d &hessian) const
    {
      pcl::KdTreeFLANN<pcl::PointXY> kdtree;
      kdtree.setInputCloud(cloud);
      gradient.setZero();
      hessian.setZero();

      // 局部累加器
      int num_threads = omp_get_max_threads();
      std::vector<Vec3> grad_private(num_threads, Vec3::Zero());
      std::vector<Eigen::Matrix3d> hess_private(num_threads, Eigen::Matrix3d::Zero());

#pragma omp parallel for
      for (int k = 0; k < n_points; ++k)
      {
        int tid = omp_get_thread_num();
        std::vector<int> neighbor_indices;
        std::vector<float> neighbor_dists;

        pcl::PointXY search_point = (*cloud)[k];
        auto k1 = (*cloud)[k].x - intrinsic_(0, 2);
        auto k2 = (*cloud)[k].y - intrinsic_(1, 2);
        kdtree.radiusSearch(search_point, radius, neighbor_indices, neighbor_dists);

        for (size_t idx = 0; idx < neighbor_indices.size(); ++idx)
        {
          int l = neighbor_indices[idx];
          if (l == k)
            continue;
          float dist2 = neighbor_dists[idx];
          double weight = 0.079577 * std::exp(-dist2 * dist2 / 4.0);
          double tl = delta_time_[l];
          double tk = delta_time_[k];
          auto l1 = (*cloud)[l].x - intrinsic_(0, 2);
          auto l2 = (*cloud)[l].y - intrinsic_(1, 2);

          grad_private[tid][0] += weight * tk * ((intrinsic_(1, 1) + k2 * k2 / intrinsic_(1, 1)) * (-k2 + l2) + k1 * k2 * (-k1 + l1) / intrinsic_(1, 1));
          grad_private[tid][1] += weight * tk * ((-intrinsic_(0, 0) - k1 * k1 / intrinsic_(0, 0)) * (-k1 + l1) - k1 * k2 * (-k2 + l2) / intrinsic_(0, 0));
          grad_private[tid][2] += weight * tk * (-k1 * (-k2 + l2) + k2 * (-k1 + l1));

          hess_private[tid](0, 0) += weight * ((intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (pow(tk, 2) * (intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) - k1 * k2 * pow(tk, 2) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) / intrinsic_(1, 1)) + (intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (-tk * tl * (intrinsic_(1, 1) + pow(l2, 2) / intrinsic_(1, 1)) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) + l1 * l2 * tk * tl * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) / intrinsic_(1, 1)) + k1 * k2 * (-pow(tk, 2) * (intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) + k1 * k2 * pow(tk, 2) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) / intrinsic_(1, 1)) / intrinsic_(1, 1) + k1 * k2 * (tk * tl * (intrinsic_(1, 1) + pow(l2, 2) / intrinsic_(1, 1)) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) - l1 * l2 * tk * tl * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) / intrinsic_(1, 1)) / intrinsic_(1, 1));
          hess_private[tid](0, 1) += weight * ((-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (-pow(tk, 2) * (intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) + k1 * k2 * pow(tk, 2) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) / intrinsic_(1, 1)) + (-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (tk * tl * (intrinsic_(1, 1) + pow(l2, 2) / intrinsic_(1, 1)) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) - l1 * l2 * tk * tl * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) / intrinsic_(1, 1)) - k1 * k2 * (pow(tk, 2) * (intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) - k1 * k2 * pow(tk, 2) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) / intrinsic_(1, 1)) / intrinsic_(0, 0) - k1 * k2 * (-tk * tl * (intrinsic_(1, 1) + pow(l2, 2) / intrinsic_(1, 1)) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) + l1 * l2 * tk * tl * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) / intrinsic_(1, 1)) / intrinsic_(0, 0));
          hess_private[tid](0, 2) += weight * (-k1 * (pow(tk, 2) * (intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) - k1 * k2 * pow(tk, 2) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) / intrinsic_(1, 1)) - k1 * (-tk * tl * (intrinsic_(1, 1) + pow(l2, 2) / intrinsic_(1, 1)) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) + l1 * l2 * tk * tl * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) / intrinsic_(1, 1)) + k2 * (-pow(tk, 2) * (intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) + k1 * k2 * pow(tk, 2) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) / intrinsic_(1, 1)) + k2 * (tk * tl * (intrinsic_(1, 1) + pow(l2, 2) / intrinsic_(1, 1)) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) - l1 * l2 * tk * tl * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) / intrinsic_(1, 1)));
          // hess_private[tid](1, 0) += weight * ((intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (-pow(tk, 2) * (-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) - k1 * k2 * pow(tk, 2) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) / intrinsic_(0, 0)) + (intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (tk * tl * (-intrinsic_(0, 0) - pow(l1, 2) / intrinsic_(0, 0)) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) + l1 * l2 * tk * tl * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) / intrinsic_(0, 0)) + k1 * k2 * (pow(tk, 2) * (-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) + k1 * k2 * pow(tk, 2) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) / intrinsic_(0, 0)) / intrinsic_(1, 1) + k1 * k2 * (-tk * tl * (-intrinsic_(0, 0) - pow(l1, 2) / intrinsic_(0, 0)) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) - l1 * l2 * tk * tl * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) / intrinsic_(0, 0)) / intrinsic_(1, 1));
          hess_private[tid](1, 1) += weight * ((-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (pow(tk, 2) * (-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) + k1 * k2 * pow(tk, 2) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) / intrinsic_(0, 0)) + (-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (-tk * tl * (-intrinsic_(0, 0) - pow(l1, 2) / intrinsic_(0, 0)) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) - l1 * l2 * tk * tl * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) / intrinsic_(0, 0)) - k1 * k2 * (-pow(tk, 2) * (-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) - k1 * k2 * pow(tk, 2) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) / intrinsic_(0, 0)) / intrinsic_(0, 0) - k1 * k2 * (tk * tl * (-intrinsic_(0, 0) - pow(l1, 2) / intrinsic_(0, 0)) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) + l1 * l2 * tk * tl * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) / intrinsic_(0, 0)) / intrinsic_(0, 0));
          hess_private[tid](1, 2) += weight * (-k1 * (-pow(tk, 2) * (-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) - k1 * k2 * pow(tk, 2) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) / intrinsic_(0, 0)) - k1 * (tk * tl * (-intrinsic_(0, 0) - pow(l1, 2) / intrinsic_(0, 0)) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) + l1 * l2 * tk * tl * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) / intrinsic_(0, 0)) + k2 * (pow(tk, 2) * (-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) + k1 * k2 * pow(tk, 2) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) / intrinsic_(0, 0)) + k2 * (-tk * tl * (-intrinsic_(0, 0) - pow(l1, 2) / intrinsic_(0, 0)) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) - l1 * l2 * tk * tl * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) / intrinsic_(0, 0)));
          // hessian(2, 0) += (intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (-k1 * pow(tk, 2) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) - k2 * pow(tk, 2) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2)) + (intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (l1 * tk * tl * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) + l2 * tk * tl * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2)) + k1 * k2 * (k1 * pow(tk, 2) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) + k2 * pow(tk, 2) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5)) / intrinsic_(1, 1) + k1 * k2 * (-l1 * tk * tl * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) - l2 * tk * tl * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5)) / intrinsic_(1, 1);
          // hessian(2, 1) += (-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (k1 * pow(tk, 2) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) + k2 * pow(tk, 2) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5)) + (-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (-l1 * tk * tl * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) - l2 * tk * tl * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5)) - k1 * k2 * (-k1 * pow(tk, 2) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) - k2 * pow(tk, 2) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2)) / intrinsic_(0, 0) - k1 * k2 * (l1 * tk * tl * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) + l2 * tk * tl * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2)) / intrinsic_(0, 0);
          hess_private[tid](2, 2) += weight * (-k1 * (-k1 * pow(tk, 2) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) - k2 * pow(tk, 2) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2)) - k1 * (l1 * tk * tl * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) + l2 * tk * tl * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2)) + k2 * (k1 * pow(tk, 2) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) + k2 * pow(tk, 2) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5)) + k2 * (-l1 * tk * tl * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) - l2 * tk * tl * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5)));
        }
      }
      // 归约
      for (int t = 0; t < num_threads; ++t)
      {
        gradient += grad_private[t];
        hessian += hess_private[t];
      }
      gradient /= static_cast<double>(n_points);
      hessian(1, 0) = hessian(0, 1);
      hessian(2, 0) = hessian(0, 2);
      hessian(2, 1) = hessian(1, 2);
      hessian.array() *= (2.0 / static_cast<double>(n_points));
    }
    void MainOptimize()
    {
      // 使用牛顿法进行优化
      const int max_iterations = 40;
      const double tolerance = 1e-3;
      Eigen::Matrix3d hessian;
      Vec3 gradient;
      Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
      double loss = std::numeric_limits<double>::max();
      for (int iter = 0; iter < max_iterations; ++iter)
      {
        compute_loss(est_angleAxis_, loss);
        compute_gradient(est_angleAxis_, gradient, hessian);

        // 添加阻尼项以确保Hessian矩阵是正定的
        Eigen::Matrix3d damped_hessian = hessian;

        // 求解线性系统
        Eigen::Vector3d step = damped_hessian.ldlt().solve(-gradient);
        // 更新参数
        est_angleAxis_ += step;

        // 检查收敛性
        if (step.norm() < tolerance)
        {
          break;
        }
// #ifdef DEBUG
//         std::cout << "Iteration " << iter << ": Loss = " << loss << ", Step Norm = " << step.norm() << std::endl;
//         std::cout << "Estimated Parameters: " << est_angleAxis_.transpose() << std::endl;
// #endif
      }
      // 计算最终损失
      compute_loss(est_angleAxis_, final_loss_);
    }
  };
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
      int padding = 100;
      Eigen::Matrix<double, 3, 1> delta_points_T, delta_second_points_T;
      Eigen::Matrix<double, 3, 1> points_T;
      // 创建翘曲事件图像
      cv::Mat warped_image = cv::Mat::zeros(height_ + 2 * padding, width_ + 2 * padding, CV_64F);

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

        // delta_second_points_T(0) = -parameters[2] * delta_points_T(1) + parameters[1] * delta_points_T(2);
        // delta_second_points_T(1) = parameters[2] * delta_points_T(0) - parameters[0] * delta_points_T(2);
        // delta_second_points_T(2) = -parameters[1] * delta_points_T(0) + parameters[0] * delta_points_T(1);
        // 计算 warp 后的 3D 点
        double dt_early = delta_time_[i];
        // double dt_sq = 0.5 * dt_early * dt_early;
        // points_T(0) = point_(0) + delta_points_T(0) * dt_early + delta_second_points_T(0) * dt_sq;
        // points_T(1) = point_(1) + delta_points_T(1) * dt_early + delta_second_points_T(1) * dt_sq;
        // points_T(2) = point_(2) + delta_points_T(2) * dt_early + delta_second_points_T(2) * dt_sq;

        points_T(0) = point_(0) + delta_points_T(0) * dt_early;
        points_T(1) = point_(1) + delta_points_T(1) * dt_early;
        points_T(2) = point_(2) + delta_points_T(2) * dt_early;
        // 投影到 2D
        (*cloud)[i].x = points_T(0) / points_T(2) * intrinsic_(0, 0) + intrinsic_(0, 2);
        (*cloud)[i].y = points_T(1) / points_T(2) * intrinsic_(1, 1) + intrinsic_(1, 2);
        // 使用双线性插值将当前事件翘曲至图像
        int x = static_cast<int>((*cloud)[i].x);
        int y = static_cast<int>((*cloud)[i].y);
        double dx = (*cloud)[i].x - x;
        double dy = (*cloud)[i].y - y;

        if (x >= -padding && x < width_ + padding - 1 && y >= -padding && y < height_ + padding - 1)
        {
          double w4 = dx * dy;
          double w3 = (1.0 - dx) * dy;
          double w2 = dx * (1.0 - dy);
          double w1 = (1.0 - dx) * (1.0 - dy);

          warped_image.at<double>(y + padding, x + padding) += w1;
          warped_image.at<double>(y + padding, x + 1 + padding) += w2;
          warped_image.at<double>(y + padding + 1, x + padding) += w3;
          warped_image.at<double>(y + padding + 1, x + 1 + padding) += w4;
        }
      }
      // 平滑翘曲图像
      cv::GaussianBlur(warped_image, warped_image, cv::Size(3, 3), 1.0);
      // 计算图像梯度
      cv::Mat grad_x, grad_y;
      // cv::Sobel(warped_image, grad_x, CV_64F, 1, 0, 3);
      // cv::Sobel(warped_image, grad_y, CV_64F, 0, 1, 3);
      // 计算残差（方差）
      cv::Scalar mean, stddev;
      cv::meanStdDev(warped_image, mean, stddev);
      double var = stddev[0] * stddev[0];
      cost[0] = -var;

      // 构建KD树
      pcl::KdTreeFLANN<pcl::PointXY> kdtree;
      kdtree.setInputCloud(cloud);

      const float radius = 4.0f;
      gradient[0] = 0.0;
      gradient[1] = 0.0;
      gradient[2] = 0.0;

      // 局部累加器
      int num_threads = omp_get_max_threads();
      std::vector<Vec3> grad_private(num_threads, Vec3::Zero());

#pragma omp parallel for
      for (int k = 0; k < n_points; ++k)
      {
        int tid = omp_get_thread_num();
        std::vector<int> neighbor_indices;
        std::vector<float> neighbor_dists;

        pcl::PointXY search_point = (*cloud)[k];
        auto k1 = (*cloud)[k].x - intrinsic_(0, 2);
        auto k2 = (*cloud)[k].y - intrinsic_(1, 2);
        // 半径搜索
        kdtree.radiusSearch(search_point, radius, neighbor_indices, neighbor_dists);

        for (size_t idx = 0; idx < neighbor_indices.size(); ++idx)
        {
          int l = neighbor_indices[idx];
          if (l == k)
            continue;
          float dist2 = neighbor_dists[idx];
          double weight = 0.079577 * std::exp(-dist2 * dist2 / 4.0) * delta_time_[k];
          auto l1 = (*cloud)[l].x - intrinsic_(0, 2);
          auto l2 = (*cloud)[l].y - intrinsic_(1, 2);
          grad_private[tid][0] += weight * ((intrinsic_(1, 1) + k2 * k2 / intrinsic_(1, 1)) * (-k2 + l2) + k1 * k2 * (-k1 + l1) / intrinsic_(1, 1));
          grad_private[tid][1] += weight * ((-intrinsic_(0, 0) - k1 * k1 / intrinsic_(0, 0)) * (-k1 + l1) - k1 * k2 * (-k2 + l2) / intrinsic_(0, 0));
          grad_private[tid][2] += weight * (-k1 * (-k2 + l2) + k2 * (-k1 + l1));
        }
      }
      for (int t = 0; t < num_threads; ++t)
      {
        gradient[0] += grad_private[t][0];
        gradient[1] += grad_private[t][1];
        gradient[2] += grad_private[t][2];
      }
      // #pragma omp parallel for
      //       for (int k = 0; k < n_points; ++k)
      //       {
      //         double x = (*cloud)[k].x;
      //         double y = (*cloud)[k].y;
      //         if (x < 0 || x >= width_ || y < 0 || y >= height_)
      //           continue;
      //         double dx = grad_x.at<double>(static_cast<int>(y), static_cast<int>(x)) * 2 * delta_time_[k];
      //         double dy = grad_y.at<double>(static_cast<int>(y), static_cast<int>(x)) * 2 * delta_time_[k];
      //         auto k1 = x - intrinsic_(0, 2);
      //         auto k2 = y - intrinsic_(1, 2);
      //         gradient[0] += dx * k1 * k2 / intrinsic_(1, 1) + dy * (intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1));
      //         gradient[1] += dx * (-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) - dy * k1 * k2 / intrinsic_(0, 0);
      //         gradient[2] += dx * k2 - dy * k1;
      //       }
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

  double CMaxEstimator::optimizeMotionIpopt()
  {
    CMaxOptimizer optimizer(
        event_undis_Bundle_.coord_3d,
        eventBundle_.time_delta,
        eventBundle_.polarity,
        width_, height_,
        K_, est_angleAxis_);

    // 运行优化
    optimizer.MainOptimize();

    // 更新结果
    optimizer.getEstimatedParams(est_angleAxis_);
#ifdef DEBUG
    std::cout << "Final estimated angular velocity: " << est_angleAxis_.transpose() << std::endl;
#endif
    return optimizer.getFinalLoss();
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
    options.max_num_iterations = 20;

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
    result.stats.loss = optimizeMotionIpopt();
    // result.stats.loss = optimizeMotion();

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