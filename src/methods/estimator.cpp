#include "spatiotemporal_vo/common.hpp"
#include "methods/estimator.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <iostream>

namespace spatiotemporal_vo
{
    MotionEstimator::MotionEstimator(double fx, double fy, double cx, double cy,
                               int width, int height)
      : width_(width), height_(height), count_(1), fx_(fx), fy_(fy), cx_(cx), cy_(cy)
  {
    // 初始化梯度和海森矩阵图像容器
      warped_image_.resize(height_ + 2 * padding_, width_ + 2 * padding_);
      pixel_grad_.resize(height_ + 2 * padding_, width_ + 2 * padding_);
      pixel_hess_.resize(height_ + 2 * padding_, width_ + 2 * padding_);
      iwe_grad_.resize(3);
      iwe_hess_.resize(9);
      for (int i = 0; i < 3; ++i)
      {
        iwe_grad_[i].resize(height_ + 2 * padding_, width_ + 2 * padding_);
      }
      for (int i = 0; i < 9; ++i)
      {
        iwe_hess_[i].resize(height_ + 2 * padding_, width_ + 2 * padding_);
      }
  }
    /**
     * @brief 计算CMax函数对像素的一阶梯度
     */

    void MotionEstimator::assign_vec(const Vec3 &vec, std::vector<EigenMat> &mat_vec, int x, int y) const
    {
      for (int i = 0; i < 3; ++i)
      {
        mat_vec[i](y, x) += vec(i);
      }
    }
    void MotionEstimator::assign_mat(const Mat3 &mat, std::vector<EigenMat> &mat_vec, int x, int y) const
    {
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; ++j)
        {
          mat_vec[i * 3 + j](y, x) += mat(i, j);
        }
      }
    }

    void MotionEstimator::MainOptimize(const Config &config)
    {
      // 使用牛顿法进行优化
      const int max_iterations = config.solver.max_iters;
      const double tolerance = config.solver.convergence_thresh;
      Mat3 hessian;
      Vec3 gradient;
      double best_loss = std::numeric_limits<double>::max();
      double loss = 0.0, lambda = 1, prev_loss = best_loss;
      Vec3 best_params = est_angleAxis_;
      for (int iter = 0; iter < max_iterations; ++iter)
      {
        compute_gradient(est_angleAxis_, loss, gradient, hessian);

        // 添加阻尼项以确保Hessian矩阵是正定的
        // hessian.diagonal().array() += (lambda * hessian.diagonal().mean());

        // 求解线性系统
        Eigen::Vector3d step = hessian.ldlt().solve(-gradient);
        est_angleAxis_ += step;
        // 检查收敛性
        if (step.norm() < tolerance || std::abs(prev_loss - loss) < tolerance)
        {
          break;
        }
        // 更新参数
        if (best_loss > loss)
        {
          best_loss = loss;
          best_params = est_angleAxis_;
        }
        prev_loss = loss;
#ifdef DEBUG
        std::cout << "[DEBUG] Iteration " << iter << ": Loss = " << loss << ", Step Norm = " << step.norm() << std::endl;
        std::cout << "[DEBUG] Estimated Parameters: " << est_angleAxis_.transpose() << std::endl;
#endif
      }
      // 计算最终损失
      est_angleAxis_ = best_params;
      final_loss_ = best_loss;
    }


  RegistrationOutput MotionEstimator::run(const Config &config, const EventPacket &packet)
  {
    points_ = &packet.coord_3d;
    delta_time_ = &packet.t;
    polarity_ = &packet.polarity;
    n_points_ = points_->size();

    RegistrationOutput result;

    if (n_points_ < 10)
    {
      result.twist.setZero();
      result.stats.loss = 0;
      result.stats.time_taken = 0.0;
      result.stats.num_iterations = 0;
      return result;
    }

    std::cout << '[' << count_ << "] " <<  "<<" << getMethodName() << ">>" << ": Processing "
              << n_points_ << " events, duration: "
              << delta_time_->back() - delta_time_->front()<< "s" << std::endl;
    result.stats.time_taken = omp_get_wtime();

    // 运行优化
    MainOptimize(config);

    // 更新结果
#ifdef DEBUG
    std::cout << "[DEBUG] Final estimated parammeters: " << est_angleAxis_.transpose() << std::endl;
#endif
    // 运行优化
    result.stats.loss = final_loss_;
    result.stats.time_taken = omp_get_wtime() - result.stats.time_taken;
    std::cout << "<<" << getMethodName() << ">>" <<": Optimization took " << result.stats.time_taken << " seconds." << std::endl;

    // 填充结果
    result.twist = est_angleAxis_;
    result.stats.num_iterations = count_;

    std::cout << '[' << count_ << "] " <<   "<<" << getMethodName() << ">>" <<": Estimated parammeters: "
              << est_angleAxis_.transpose()
              << ", best loss: " << result.stats.loss << std::endl;
    count_++;
    return result;
  }

} // namespace spatiotemporal_vo