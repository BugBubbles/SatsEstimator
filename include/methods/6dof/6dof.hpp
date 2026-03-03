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
      regular_loss_ = 0;
      regular_gradient_.setZero();
      regular_hessian_.setZero();
#ifdef SEPARATE_POLARITY
      warped_image_pos_.setZero();
      warped_image_neg_.setZero();
#else
      warped_image_.setZero();
#endif
// 初始化梯度和海森矩阵图像
#pragma omp parallel for
      for (int i = 0; i < 6; ++i)
      {
#ifdef SEPARATE_POLARITY
        iwe_grad_pos_[i].setZero();
        iwe_grad_neg_[i].setZero();
#else
        iwe_grad_[i].setZero();
#endif
      }
#ifdef ANALYTIC_HESSIAN
#pragma omp parallel for
      for (int i = 0; i < 36; ++i)
      {
#ifdef SEPARATE_POLARITY
        iwe_hess_pos_[i].setZero();
        iwe_hess_neg_[i].setZero();
#else
        iwe_hess_[i].setZero();
#endif
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
        double grad, res, huber_val, huber_grad, huber_sec_grad;

        // res = std::log(std::fabs(point_(0) + 1e-6) / (std::fabs(points_T(0)) + 1e-6));
        // grad = dt / (std::fabs(points_T(0)) + 1e-6);
        // huber_loss(res, huber_val, huber_grad, huber_sec_grad);
        // regular_loss_ += huber_val;
        // regular_gradient_(0) += huber_grad * grad * (points_T(0) > 0 ? 1 : -1);
        // regular_hessian_(0, 0) += (huber_grad + huber_sec_grad) * grad * grad;

        // res = std::log(std::fabs(point_(1) + 1e-6) / (std::fabs(points_T(1)) + 1e-6));
        // grad = dt / (std::fabs(points_T(1)) + 1e-6);
        // huber_loss(res, huber_val, huber_grad, huber_sec_grad);
        // regular_loss_ += huber_val;
        // regular_gradient_(1) += huber_grad * grad * (points_T(1) > 0 ? 1 : -1);
        // regular_hessian_(1, 1) += (huber_grad + huber_sec_grad) * grad * grad;

        // res = std::log(std::fabs(point_(2) + 1e-6) / (std::fabs(points_T(2)) + 1e-6));
        // grad = dt / (std::fabs(points_T(2)) + 1e-6);
        // huber_loss(res, huber_val, huber_grad, huber_sec_grad);
        // regular_loss_ += huber_val;
        // regular_gradient_(2) += huber_grad * grad * (points_T(2) > 0 ? 1 : -1);
        // regular_hessian_(2, 2) += (huber_grad + huber_sec_grad) * grad * grad;

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
#ifdef SEPARATE_POLARITY
          switch (polarity_->at(i))
          {
          case 1:
            warped_image_pos_(y, x) += w1;
            warped_image_pos_(y, x + 1) += w2;
            warped_image_pos_(y + 1, x) += w3;
            warped_image_pos_(y + 1, x + 1) += w4;
            // 计算梯度
            warp_gradient(parameters, point_, dt, grad_warp, hess_warp);
            assign_vec((dy - 1) * grad_warp.row(0) + (dx - 1) * grad_warp.row(1), iwe_grad_pos_, x, y);
            assign_vec((1 - dy) * grad_warp.row(0) + (-dx) * grad_warp.row(1), iwe_grad_pos_, x + 1, y);
            assign_vec((-dy) * grad_warp.row(0) + (1 - dx) * grad_warp.row(1), iwe_grad_pos_, x, y + 1);
            assign_vec(dy * grad_warp.row(0) + dx * grad_warp.row(1), iwe_grad_pos_, x + 1, y + 1);
            // 计算海森矩阵
#ifdef ANALYTIC_HESSIAN
            assign_mat(hess_warp, iwe_hess_pos_, x, y);
            assign_mat(-hess_warp, iwe_hess_pos_, x + 1, y);
            assign_mat(-hess_warp, iwe_hess_pos_, x, y + 1);
            assign_mat(hess_warp, iwe_hess_pos_, x + 1, y + 1);
#endif
            break;
          case -1:
            warped_image_neg_(y, x) += w1;
            warped_image_neg_(y, x + 1) += w2;
            warped_image_neg_(y + 1, x) += w3;
            warped_image_neg_(y + 1, x + 1) += w4;
            // 计算梯度
            warp_gradient(parameters, point_, dt, grad_warp, hess_warp);
            assign_vec((dy - 1) * grad_warp.row(0) + (dx - 1) * grad_warp.row(1), iwe_grad_neg_, x, y);
            assign_vec((1 - dy) * grad_warp.row(0) + (-dx) * grad_warp.row(1), iwe_grad_neg_, x + 1, y);
            assign_vec((-dy) * grad_warp.row(0) + (1 - dx) * grad_warp.row(1), iwe_grad_neg_, x, y + 1);
            assign_vec(dy * grad_warp.row(0) + dx * grad_warp.row(1), iwe_grad_neg_, x + 1, y + 1);
            // 计算海森矩阵
#ifdef ANALYTIC_HESSIAN
            assign_mat(hess_warp, iwe_hess_neg_, x, y);
            assign_mat(-hess_warp, iwe_hess_neg_, x + 1, y);
            assign_mat(-hess_warp, iwe_hess_neg_, x, y + 1);
            assign_mat(hess_warp, iwe_hess_neg_, x + 1, y + 1);
#endif
            break;
          }
#else
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
#endif
        }
      }
#ifdef SEPARATE_POLARITY
      blur_eigen_mat(warped_image_pos_);
      blur_eigen_mat(warped_image_neg_);
#else
      blur_eigen_mat(warped_image_);
#endif
// 2. 模糊雅可比图
// 这实际上是在计算：参数微扰动会导致"模糊后的图像"发生怎样的变化
// 没有这一步，梯度就是针对"尖锐"图像的，与 Loss 不匹配
#pragma omp parallel for
      for (int k = 0; k < 6; ++k)
      {
#ifdef SEPARATE_POLARITY
        blur_eigen_mat(iwe_grad_pos_[k]);
        blur_eigen_mat(iwe_grad_neg_[k]);
#else
        blur_eigen_mat(iwe_grad_[k]);
#endif
      }

// 3. 模糊海森图
#ifdef ANALYTIC_HESSIAN
#pragma omp parallel for
      for (int k = 0; k < 36; ++k)
      {
#ifdef SEPARATE_POLARITY
        blur_eigen_mat(iwe_hess_pos_[k]);
        blur_eigen_mat(iwe_hess_neg_[k]);
#else
        blur_eigen_mat(iwe_hess_[k]);
#endif
      }
#endif
// 计算最终的梯度与海森矩阵
#ifdef SEPARATE_POLARITY
      double loss_pos, loss_neg;
      iwe_gradient(warped_image_pos_, loss_pos, pixel_grad_pos_, pixel_hess_pos_);
      iwe_gradient(warped_image_neg_, loss_neg, pixel_grad_neg_, pixel_hess_neg_);
      loss = loss_pos + loss_neg;
#else
      iwe_gradient(warped_image_, loss, pixel_grad_, pixel_hess_);
#endif
// 计算总梯度，各个像素梯度的加权和
// gradient 等于将 pixel_grad 与 iwe_grad_ 第一个通道按像素位置对应相乘后求和
#pragma omp parallel for
      for (int i = 0; i < 6; ++i)
      {
#ifdef SEPARATE_POLARITY
        gradient(i) = (pixel_grad_pos_.array() * iwe_grad_pos_[i].array()).sum() + (pixel_grad_neg_.array() * iwe_grad_neg_[i].array()).sum();
#else
        gradient(i) = (pixel_grad_.array() * iwe_grad_[i].array()).sum();
#endif
      }
#pragma omp parallel for
      for (int i = 0; i < 6; ++i)
      {
#ifdef ANALYTIC_HESSIAN
#ifdef SEPARATE_POLARITY
        hessian(i, i) = (pixel_grad_pos_.array() * iwe_hess_pos_[6 * i + i].array()).sum() + (pixel_hess_pos_.array() * iwe_grad_pos_[i].array() * iwe_grad_pos_[i].array()).sum() + (pixel_grad_neg_.array() * iwe_hess_neg_[6 * i + i].array()).sum() + (pixel_hess_neg_.array() * iwe_grad_neg_[i].array() * iwe_grad_neg_[i].array()).sum();
#else
        hessian(i, i) = (pixel_grad_.array() * iwe_hess_[6 * i + i].array()).sum() + (pixel_hess_.array() * iwe_grad_[i].array() * iwe_grad_[i].array()).sum();
#endif
#else
#ifdef SEPARATE_POLARITY
        hessian(i, i) = (pixel_hess_pos_.array() * iwe_grad_pos_[i].array() * iwe_grad_pos_[i].array()).sum() + (pixel_hess_neg_.array() * iwe_grad_neg_[i].array() * iwe_grad_neg_[i].array()).sum();
#else
        hessian(i, i) = (pixel_hess_.array() * iwe_grad_[i].array() * iwe_grad_[i].array()).sum();
#endif
#endif
        for (int j = i + 1; j < 6; ++j)
        {
#ifdef ANALYTIC_HESSIAN
#ifdef SEPARATE_POLARITY
          hessian(i, j) = (pixel_grad_pos_.array() * iwe_hess_pos_[6 * i + j].array()).sum() + (pixel_hess_pos_.array() * iwe_grad_pos_[i].array() * iwe_grad_pos_[j].array()).sum() + (pixel_grad_neg_.array() * iwe_hess_neg_[6 * i + j].array()).sum() + (pixel_hess_neg_.array() * iwe_grad_neg_[i].array() * iwe_grad_neg_[j].array()).sum();
#else
          hessian(i, j) = (pixel_grad_.array() * iwe_hess_[6 * i + j].array()).sum() + (pixel_hess_.array() * iwe_grad_[i].array() * iwe_grad_[j].array()).sum();
#endif
#else
#ifdef SEPARATE_POLARITY
          hessian(i, j) = (pixel_hess_pos_.array() * iwe_grad_pos_[i].array() * iwe_grad_pos_[j].array()).sum() + (pixel_hess_neg_.array() * iwe_grad_neg_[i].array() * iwe_grad_neg_[j].array()).sum();
#else
          hessian(i, j) = (pixel_hess_.array() * iwe_grad_[i].array() * iwe_grad_[j].array()).sum();
#endif
#endif
          hessian(j, i) = hessian(i, j); // 对称矩阵
        }
      }
#ifdef DEBUG
// 将warped_image_保存为图像，便于调试
#ifdef SEPARATE_POLARITY
      cv::Mat debug_img_pos(pad_width, pad_height, CV_64F, warped_image_pos_.data());
      cv::Mat debug_img_neg(pad_width, pad_height, CV_64F, warped_image_neg_.data());
      debug_img_pos /= warped_image_pos_.maxCoeff();
      debug_img_neg /= warped_image_pos_.maxCoeff();
      cv::imwrite("debug_warped_pos.png", debug_img_pos * 255);
      cv::imwrite("debug_warped_neg.png", debug_img_neg * 255);
#else
      cv::Mat debug_img(pad_width, pad_height, CV_64F, warped_image_.data());
      debug_img /= warped_image_.maxCoeff();
      cv::rotate(debug_img, debug_img, cv::ROTATE_90_CLOCKWISE); 
      cv::imwrite("debug_warped.png", debug_img * 255);
#endif
#endif
      regular_loss_ = parameters.head<3>().norm() + 1e-6;
      regular_gradient_.head<3>() = parameters.head<3>() / regular_loss_;
      regular_hessian_ = -regular_gradient_ * regular_gradient_.transpose();
      regular_hessian_(0, 0) += 1.0;
      regular_hessian_(1, 1) += 1.0;
      regular_hessian_(2, 2) += 1.0;
      regular_gradient_ /= regular_loss_;
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
        std::cout << "[DEBUG] hessian :\n"
                  << hessian << std::endl;
        std::cout << "[DEBUG] <<Regular>> loss = " << regularization * regular_loss_ / n_points_ << std::endl;
        std::cout << "[DEBUG] <<Regular>> gradien = " << regularization * regular_gradient_.transpose() << std::endl;
        std::cout << "[DEBUG] <<Regular>> hessian :\n"
                  << regularization * regular_hessian_ << std::endl;
#endif

        loss += regularization * regular_loss_;
        gradient += regularization * regular_gradient_;
        hessian += regularization * regular_hessian_;

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
    double regular_loss_;
    Vec6 regular_gradient_;
    Mat6 regular_hessian_;
  };
};