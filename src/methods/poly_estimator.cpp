#include "spatiotemporal_vo/common.hpp"
#include "methods/poly_estimator.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <iostream>

namespace spatiotemporal_vo
{

  class PolyOptimizer
  {
  private:
    // 你的数据成员（和原CMaxFunction一致）
    std::vector<Vec3> points_;
    int width_, height_;
    Eigen::Matrix3d intrinsic_;
    std::vector<EigenMat> iwe_grad_;
    std::vector<EigenMat> iwe_hess_;
    EigenMat warped_image_, pixel_grad_, pixel_hess_;
    std::vector<double> delta_time_;
    std::vector<int> polarity_;
    int n_points_;
    int padding_ = 100;
    double final_loss_ = 0.0;
    Vec3 est_angleAxis_;

  public:
    void getEstimatedParams(Vec3 &est_angleAxis) const { est_angleAxis = est_angleAxis_; }
    double getFinalLoss() const { return final_loss_; }
    PolyOptimizer(const EventPacket &packet,
                  int width, int height,
                  const Eigen::Matrix3d &K,
                  const Vec3 &initial_angleAxis)
        : points_(packet.coord_3d), delta_time_(packet.t), polarity_(packet.polarity),
          width_(width), height_(height), intrinsic_(K), est_angleAxis_(initial_angleAxis)
    {
      n_points_ = points_.size();
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
    void iwe_gradient(const EigenMat &iwe, double &loss, EigenMat &grad, EigenMat &hess) const
    {
      // Ours
      loss = (0.9416 * iwe.array() + (iwe.array() + 0.1).log()).sum() / iwe.sum();
      grad = 0.9416 + 1 / (iwe.array() + 0.1);
      hess = -1 / (iwe.array() + 0.1).square();
    }
    /**
     * @brief 计算运动参数对翘曲事件点的梯度
     * @param parameters 运动参数（角轴表示法）
     * @param xs, ys 翘曲后的事件点（2D坐标），注意，这里的坐标是减去u0 v0后的坐标
     * @param grad_warp 输出的梯度矩阵，大小为2x3
     */
    void warp_gradient(const Vec3 &parameters, const Vec3 &point, double dt, Eigen::Matrix<double, 2, 3> &grad_warp, Eigen::Matrix3d &hess_warp) const
    {
      double X_Z = point(0) / point(2);
      double Y_Z = point(1) / point(2);
      grad_warp << intrinsic_(0, 0) * X_Z * Y_Z, -intrinsic_(0, 0) * (1 + X_Z * X_Z), intrinsic_(0, 0) * Y_Z,
          intrinsic_(1, 1) * (1 + Y_Z * Y_Z), -intrinsic_(1, 1) * X_Z * Y_Z, -intrinsic_(1, 1) * X_Z;
      grad_warp *= dt;
      hess_warp = grad_warp.row(0).transpose() * grad_warp.row(1) + grad_warp.row(1).transpose() * grad_warp.row(0);
    }
    void assign_vec(const Vec3 &vec, std::vector<EigenMat> &mat_vec, int x, int y) const
    {
      for (int i = 0; i < 3; ++i)
      {
        mat_vec[i](y, x) += vec(i);
      }
    }
    void assign_mat(const Eigen::Matrix3d &mat, std::vector<EigenMat> &mat_vec, int x, int y) const
    {
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; ++j)
        {
          mat_vec[i * 3 + j](y, x) += mat(i, j);
        }
      }
    }
    void compute_gradient(const Vec3 &parameters, double &loss, Vec3 &gradient, Eigen::Matrix3d &hessian)
    {
      auto blur_eigen_mat = [&](EigenMat &mat)
      {
        // 注意：Eigen是列优先，所以rows是Eigen的cols，cols是Eigen的rows
        cv::Mat cv_mat(mat.cols(), mat.rows(), CV_64F, mat.data());
        // 使用与 compute_loss 中相同的模糊参数 (Size(5,5), sigma=1.0)
        // 这一步对于模拟第一种方法的"Radius Search"至关重要
        cv::GaussianBlur(cv_mat, cv_mat, cv::Size(5, 5), 1.0);
      };
      Eigen::Matrix<double, 3, 1> delta_points_T, delta_second_points_T;
      Eigen::Matrix<double, 3, 1> points_T;
      // 创建翘曲事件图像
      int pad_width = width_ + 2 * padding_;
      int pad_height = height_ + 2 * padding_;
      warped_image_.setZero();
      // 初始化梯度和海森矩阵图像
      for (int i = 0; i < 3; ++i)
      {
        iwe_grad_[i].setZero();
      }
      for (int i = 0; i < 9; ++i)
      {
        iwe_hess_[i].setZero();
      }
      double warped_x, warped_y, event_x, event_y;
      Eigen::Matrix<double, 2, 3> grad_warp;
      Eigen::Matrix3d hess_warp;

      // 提取内参，避免在循环中重复访问成员变量
      const double fx = intrinsic_(0, 0);
      const double cx = intrinsic_(0, 2);
      const double fy = intrinsic_(1, 1);
      const double cy = intrinsic_(1, 2);
      for (int i = 0; i < n_points_; ++i)
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
        double dt = delta_time_[i];
        // double dt_sq = 0.5 * dt * dt;
        points_T(0) = point_(0) + delta_points_T(0) * dt;
        points_T(1) = point_(1) + delta_points_T(1) * dt;
        points_T(2) = point_(2) + delta_points_T(2) * dt;

        // 投影到 2D
        warped_x = points_T(0) / points_T(2) * fx + cx + padding_;
        warped_y = points_T(1) / points_T(2) * fy + cy + padding_;
        // 使用双线性插值将当前事件翘曲至图像
        int x = static_cast<int>(floor(warped_x));
        int y = static_cast<int>(floor(warped_y));
        double dx = warped_x - x;
        double dy = warped_y - y;

        if (x >= 0 && x < pad_width - 1 && y >= 0 && y < pad_height - 1)
        {
          double w4 = dx * dy;
          double w3 = (1.0 - dx) * dy;
          double w2 = dx * (1.0 - dy);
          double w1 = (1.0 - dx) * (1.0 - dy);

          warped_image_(y, x) += w1;
          warped_image_(y, x + 1) += w2;
          warped_image_(y + 1, x) += w3;
          warped_image_(y + 1, x + 1) += w4;
          // 计算梯度
          warp_gradient(parameters, point_, dt, grad_warp, hess_warp);
          assign_vec((dy - 1) * grad_warp.row(0) + (dx - 1) * grad_warp.row(1), iwe_grad_, x, y);
          assign_vec((1 - dy) * grad_warp.row(0) + (-dx) * grad_warp.row(1), iwe_grad_, x + 1, y);
          assign_vec((-dy) * grad_warp.row(0) + (1 - dx) * grad_warp.row(1), iwe_grad_, x, y + 1);
          assign_vec(dy * grad_warp.row(0) + dx * grad_warp.row(1), iwe_grad_, x + 1, y + 1);
          // 计算海森矩阵
          assign_mat(hess_warp, iwe_hess_, x, y);
          assign_mat(-hess_warp, iwe_hess_, x + 1, y);
          assign_mat(-hess_warp, iwe_hess_, x, y + 1);
          assign_mat(hess_warp, iwe_hess_, x + 1, y + 1);
        }
      }
      blur_eigen_mat(warped_image_);

// 2. 模糊雅可比图 (最重要的改动!)
// 这实际上是在计算：参数微扰动会导致"模糊后的图像"发生怎样的变化
// 没有这一步，梯度就是针对"尖锐"图像的，与 Loss 不匹配
#pragma omp parallel for
      for (int k = 0; k < 3; ++k)
      {
        blur_eigen_mat(iwe_grad_[k]);
      }

// 3. 模糊海森图
#pragma omp parallel for
      for (int k = 0; k < 9; ++k)
      {
        blur_eigen_mat(iwe_hess_[k]);
      }
      // 计算最终的梯度与海森矩阵
      iwe_gradient(warped_image_, loss, pixel_grad_, pixel_hess_);
// 计算总梯度，各个像素梯度的加权和
// gradient 等于将 pixel_grad 与 iwe_grad_ 第一个通道按像素位置对应相乘后求和
#pragma omp parallel for
      for (int i = 0; i < 3; ++i)
      {
        gradient(i) = (pixel_grad_.array() * iwe_grad_[i].array()).sum();
      }
#pragma omp parallel for
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; ++j)
        {
          hessian(i, j) = (pixel_grad_.array() * iwe_hess_[3 * i + j].array()).sum() + (pixel_hess_.array() * iwe_grad_[i].array() * iwe_grad_[j].array()).sum();
        }
      }
    }
    void MainOptimize(const Config &config)
    {
      // 使用牛顿法进行优化
      const int max_iterations = config.solver.max_iters;
      const double tolerance = config.solver.epsilon_T;
      Eigen::Matrix3d hessian;
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
        std::cout << "Iteration " << iter << ": Loss = " << loss << ", Step Norm = " << step.norm() << std::endl;
        std::cout << "Estimated Parameters: " << est_angleAxis_.transpose() << std::endl;
#endif
      }
      // 计算最终损失
      est_angleAxis_ = best_params;
      final_loss_ = best_loss;
      // compute_loss(est_angleAxis_, final_loss_);
    }
  };
  PolyEstimator::PolyEstimator(double fx, double fy, double cx, double cy,
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

  RegistrationOutput PolyEstimator::run(const EventPacket &packet, const Config &config)
  {
    RegistrationOutput result;
    result.timespan = {packet.t.front(), packet.t.back()};

    if (packet.empty())
    {
      result.twist.setZero();
      result.stats.loss = 0;
      result.stats.time_taken = 0.0;
      result.stats.num_iterations = 0;
      return result;
    }

    std::cout << '[' << count_ << "] " << "Poly: Processing "
              << packet.size() << " events, duration: "
              << packet.timespan() << "s" << std::endl;
    result.stats.time_taken = omp_get_wtime();
    // 构造优化器
    PolyOptimizer optimizer(
        packet,
        width_, height_,
        K_, est_angleAxis_);

    // 运行优化
    optimizer.MainOptimize(config);

    // 更新结果
    optimizer.getEstimatedParams(est_angleAxis_);
#ifdef DEBUG
    std::cout << "[DEBUG] Final estimated angular velocity: " << est_angleAxis_.transpose() << std::endl;
#endif
    // 运行优化
    result.stats.loss = optimizer.getFinalLoss();
    result.stats.time_taken = omp_get_wtime() - result.stats.time_taken;
    std::cout << "Poly: Optimization took " << result.stats.time_taken << " seconds." << std::endl;

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