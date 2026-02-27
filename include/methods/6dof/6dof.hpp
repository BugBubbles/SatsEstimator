#pragma once

#include "methods/estimator.hpp"

namespace spatiotemporal_vo
{

  class SixDofEstimator : public MotionEstimator<6>
  {
    public:
    using MotionEstimator<6>::MotionEstimator; // 继承所有基类构造函数
    protected:
    using Vec6 = Eigen::Matrix<double, 6, 1>;
    using Mat6 = Eigen::Matrix<double, 6, 6>;
    void warp_gradient(const Vec6 &parameters, const Vec3 &point, double dt, Eigen::Matrix<double, 2, 6> &grad_warp, Mat6 &hess_warp) const
    {
      double X_Z = point(0) / point(2);
      double Y_Z = point(1) / point(2);
      grad_warp << -fx_ / point(2), 0, fx_ * X_Z / point(2), fx_ * X_Z * Y_Z, -fx_ * (1 + X_Z * X_Z), fx_ * Y_Z,
          0, -fy_ / point(2), fy_ * Y_Z / point(2), fy_ * (1 + Y_Z * Y_Z), -fy_ * X_Z * Y_Z, -fy_ * X_Z;
      grad_warp *= dt;
      hess_warp = grad_warp.row(0).transpose() * grad_warp.row(1) + grad_warp.row(1).transpose() * grad_warp.row(0);
    }

    void compute_gradient(const Vec6 &parameters, double &loss, Vec6 &gradient, Mat6 &hessian) override
    {
      auto huber_loss = [](double &x, double &loss, double &grad, double &sec_grad, double delta = 1.0)
      {
        if (std::abs(x) <= delta)
        {
          loss = 0.5 * x * x;
          grad = x;
          sec_grad = 1.0;
        }
        else
        {
          loss = delta * (std::abs(x) - 0.5 * delta);
          grad = delta * (x > 0 ? 1 : -1);
          sec_grad = 0.0;
        }
      };
      auto blur_eigen_mat = [&](EigenMat &mat)
      {
        // 注意：Eigen是列优先，所以rows是Eigen的cols，cols是Eigen的rows
        cv::Mat cv_mat(mat.cols(), mat.rows(), CV_64F, mat.data());
        // 使用与 compute_loss 中相同的模糊参数 (Size(5,5), sigma=1.0)
        // 这一步对于模拟第一种方法的"Radius Search"至关重要
        cv::GaussianBlur(cv_mat, cv_mat, cv::Size(5, 5), 1.0);
      };
      Vec3 delta_points_rot, delta_points_trans, points_T;
      // 创建翘曲事件图像
      int pad_width = width_ + 2 * padding_;
      int pad_height = height_ + 2 * padding_;
      warped_image_.setZero();
      // 初始化梯度和海森矩阵图像
      #pragma omp parallel for
      for (int i = 0; i < 6; ++i)
      {
        iwe_grad_[i].setZero();
      }
#ifdef ANALYTIC_HESSIAN
      #pragma omp parallel for
      for (int i = 0; i < 36; ++i)
      {
        iwe_hess_[i].setZero();
      }
#endif
      double warped_x, warped_y, event_x, event_y;
      Eigen::Matrix<double, 2, 6> grad_warp;
      Mat6 hess_warp;

      // 提取内参，避免在循环中重复访问成员变量
      for (int i = 0; i < n_points_; ++i)
      {
        auto point_ = points_->at(i);
        // Taylor 二阶展开: points * skew_matrix * skew_matrix
        delta_points_rot(0) = -parameters[5] * point_(1) + parameters[4] * point_(2);
        delta_points_rot(1) = parameters[5] * point_(0) - parameters[3] * point_(2);
        delta_points_rot(2) = -parameters[4] * point_(0) + parameters[3] * point_(1);

        delta_points_trans(0) = parameters[0];
        delta_points_trans(1) = parameters[1];
        delta_points_trans(2) = parameters[2];

        // 计算 warp 后的 3D 点
        double dt = delta_time_->at(i);
        points_T = point_ + delta_points_rot * dt + delta_points_trans * dt;

        // 投影到 2D
        warped_x = points_T(0) / points_T(2) * fx_ + cx_ + padding_;
        warped_y = points_T(1) / points_T(2) * fy_ + cy_ + padding_;

        // 退化正则项损失值
        double grad = dt / (std::fabs(points_T(2)) + 1e-6);
        double res = std::log(point_(2) / (std::fabs(points_T(2)) + 1e-6));
        double huber_val, huber_grad, huber_sec_grad;

        huber_loss(res, huber_val, huber_grad, huber_sec_grad);
        regular_loss_ += huber_val;
        regular_gradient_2_ += huber_grad * grad * (points_T(2) > 0 ? 1 : -1);
        regular_gradient_3_ += -huber_grad * grad * (points_T(2) > 0 ? 1 : -1) * point_(1);
        regular_gradient_4_ += huber_grad * grad * (points_T(2) > 0 ? 1 : -1) * point_(0);
        regular_hessian_2_2_ += (huber_grad + huber_sec_grad) * grad * grad;
        regular_hessian_3_3_ += (huber_grad + huber_sec_grad) * grad * grad * point_(1) * point_(1);
        regular_hessian_4_4_ += (huber_grad + huber_sec_grad) * grad * grad * point_(0) * point_(0);
        regular_hessian_2_3_ += -(huber_grad + huber_sec_grad) * grad * grad * point_(1);
        regular_hessian_2_4_ += (huber_grad + huber_sec_grad) * grad * grad * point_(0);
        regular_hessian_3_4_ += -(huber_grad + huber_sec_grad) * grad * grad * point_(0) * point_(1);

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
#ifdef ANALYTIC_HESSIAN
          assign_mat(hess_warp, iwe_hess_, x, y);
          assign_mat(-hess_warp, iwe_hess_, x + 1, y);
          assign_mat(-hess_warp, iwe_hess_, x, y + 1);
          assign_mat(hess_warp, iwe_hess_, x + 1, y + 1);
#endif
        }
      }
      blur_eigen_mat(warped_image_);

// 2. 模糊雅可比图
// 这实际上是在计算：参数微扰动会导致"模糊后的图像"发生怎样的变化
// 没有这一步，梯度就是针对"尖锐"图像的，与 Loss 不匹配
#pragma omp parallel for
      for (int k = 0; k < 6; ++k)
      {
        blur_eigen_mat(iwe_grad_[k]);
      }

// 3. 模糊海森图
#ifdef ANALYTIC_HESSIAN
#pragma omp parallel for
      for (int k = 0; k < 36; ++k)
      {
        blur_eigen_mat(iwe_hess_[k]);
      }
#endif
      // 计算最终的梯度与海森矩阵
      iwe_gradient(warped_image_, loss, pixel_grad_, pixel_hess_);
// 计算总梯度，各个像素梯度的加权和
// gradient 等于将 pixel_grad 与 iwe_grad_ 第一个通道按像素位置对应相乘后求和
#pragma omp parallel for
      for (int i = 0; i < 6; ++i)
      {
        gradient(i) = (pixel_grad_.array() * iwe_grad_[i].array()).sum();
      }
#pragma omp parallel for
      for (int i = 0; i < 6; ++i)
      {
        for (int j = 0; j < 6; ++j)
        {
#ifdef ANALYTIC_HESSIAN
          hessian(i, j) = (pixel_grad_.array() * iwe_hess_[6 * i + j].array()).sum() + (pixel_hess_.array() * iwe_grad_[i].array() * iwe_grad_[j].array()).sum();
#else
        hessian(i, j) = (pixel_hess_.array() * iwe_grad_[i].array() * iwe_grad_[j].array()).sum();
#endif
        }
      }
    }
    void MainOptimize(const Config &config) override
    {
      // 使用牛顿法进行优化
      const int max_iterations = config.solver.max_iters;
      const double tolerance = config.solver.convergence_thresh;
      const double regularization = config.solver.regularization;
      const double decay = config.solver.decay;
      double lr = config.solver.lr;
      Mat6 hessian;
      Vec6 gradient;
      double best_loss = std::numeric_limits<double>::max();
      double loss = 0.0, lambda = 1.5, prev_loss = best_loss;
      Vec6 step, best_params = est_params_;

      // // Adam优化器参数
      // Vec6 m = Vec6::Zero(), v = Vec6::Zero();
      // const double beta1 = 0.9, inv_beta1 = 1-beta1;
      // const double beta2= 0.999, inv_beta2 =1-beta2, epsilon = 1e-8;
      // double beta1_T = beta1, beta2_T = beta2;
      for (int iter = 0; iter < max_iterations; ++iter)
      {
        compute_gradient(est_params_, loss, gradient, hessian);

        // 添加阻尼项以确保Hessian矩阵是正定的
        // hessian.diagonal().array() += (lambda * hessian.diagonal().cwiseAbs().array());
#ifdef DEBUG
        std::cout << "[DEBUG] loss = " << loss << std::endl;
        std::cout << "[DEBUG] gradient = " << gradient.transpose() << std::endl;
        std::cout << "[DEBUG] hessian :" << hessian << std::endl;
        Eigen::SelfAdjointEigenSolver<Mat6> eigensolver(hessian);
        std::cout << "[DEBUG] Hessian eigenvalues: " << eigensolver.eigenvalues().transpose() << std::endl;
        std::cout << "[DEBUG] <<Regular>> loss = " << regularization * regular_loss_ / n_points_ << std::endl;
        std::cout << "[DEBUG] <<Regular>> gradien = " << regularization * regular_gradient_2_ << regularization * regular_gradient_3_ << regularization * regular_gradient_4_ << std::endl;
        std::cout << "[DEBUG] <<Regular>> hessian_2_2 = " << regularization * regular_hessian_2_2_ << std::endl;
        std::cout << "[DEBUG] <<Regular>> hessian_3_3 = " << regularization * regular_hessian_3_3_ << std::endl;
        std::cout << "[DEBUG] <<Regular>> hessian_4_4 = " << regularization * regular_hessian_4_4_ << std::endl;
        std::cout << "[DEBUG] <<Regular>> hessian_2_3 = " << regularization * regular_hessian_2_3_ << std::endl;
        std::cout << "[DEBUG] <<Regular>> hessian_2_4 = " << regularization * regular_hessian_2_4_ << std::endl;
        std::cout << "[DEBUG] <<Regular>> hessian_3_4 = " << regularization * regular_hessian_3_4_ << std::endl;
#endif

        loss += regularization * regular_loss_ / n_points_;
        gradient(2) += regularization * regular_gradient_2_;
        gradient(3) += regularization * regular_gradient_3_;
        gradient(4) += regularization * regular_gradient_4_;
        hessian(2, 2) += regularization * regular_hessian_2_2_;
        hessian(3, 3) += regularization * regular_hessian_3_3_;
        hessian(4, 4) += regularization * regular_hessian_4_4_;
        hessian(2, 3) += regularization * regular_hessian_2_3_;
        hessian(3, 2) += regularization * regular_hessian_2_3_;
        hessian(2, 4) += regularization * regular_hessian_2_4_;
        hessian(4, 2) += regularization * regular_hessian_2_4_;
        hessian(3, 4) += regularization * regular_hessian_3_4_;
        hessian(4, 3) += regularization * regular_hessian_3_4_;

        // 求解线性系统，LM算法
        step = hessian.ldlt().solve(-gradient) * lr;
        // Adam优化器更新
        // m = beta1 * m + inv_beta1 * gradient;
        // v = beta2 * v + inv_beta2 * gradient.array().square().matrix();
        // beta1_T *= beta1;
        // beta2_T *= beta2;
        // step = lr * m.array() / (v.array().sqrt() + epsilon) * (std::sqrt(1 - beta2_T) / (1 - beta1_T));
        est_params_ += step;
        // 检查收敛性
        if (step.norm() < tolerance || std::abs(prev_loss - loss) < tolerance)
        {
          break;
        }
        // 更新参数
        if (best_loss > loss)
        {
          best_loss = loss;
          best_params = est_params_;
        }
        prev_loss = loss;
        lr *= decay;
#ifdef DEBUG
        std::cout << "[DEBUG] Iteration " << iter << ": Loss = " << loss << ", Step Norm = " << step.norm() << std::endl;
        std::cout << "[DEBUG] Estimated translation velocity: " << est_params_.head<3>().transpose() << std::endl;
        std::cout << "[DEBUG] Estimated rotation velocity: " << est_params_.tail<3>().transpose() << std::endl;
#endif
      }
      // 计算最终损失
      est_params_ = best_params;
      final_loss_ = best_loss;
    }
    private:
    double regular_loss_, regular_gradient_2_, regular_gradient_3_, regular_gradient_4_;
    double regular_hessian_2_2_, regular_hessian_3_3_, regular_hessian_4_4_;
    double regular_hessian_2_3_, regular_hessian_2_4_, regular_hessian_3_4_;
  };
};