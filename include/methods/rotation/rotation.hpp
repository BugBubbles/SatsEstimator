#pragma once

#include "methods/estimator.hpp"

namespace spatiotemporal_vo
{

  /**
   * @brief CMax (Contrast Maximization) 估计器
   * 基于时间表面的对比度最大化方法
   */
  class RotationEstimator : public MotionEstimator
  {
    public:
    using MotionEstimator::MotionEstimator; // 继承所有基类构造函数
    protected:
    void warp_gradient(const Vec3 &parameters, const Vec3 &point, double dt, Eigen::Matrix<double, 2, 3> &grad_warp, Mat3 &hess_warp) const
    {
      double X_Z = point(0) / point(2);
      double Y_Z = point(1) / point(2);
      grad_warp << fx_ * X_Z * Y_Z, -fx_ * (1 + X_Z * X_Z), fx_ * Y_Z,
          fy_ * (1 + Y_Z * Y_Z), -fy_ * X_Z * Y_Z, -fy_ * X_Z;
      grad_warp *= dt;
      hess_warp = grad_warp.row(0).transpose() * grad_warp.row(1) + grad_warp.row(1).transpose() * grad_warp.row(0);
    }
    void compute_gradient(const Vec3 &parameters, double &loss, Vec3 &gradient, Mat3 &hessian) override
    {
      auto blur_eigen_mat = [&](EigenMat &mat)
      {
        // 注意：Eigen是列优先，所以rows是Eigen的cols，cols是Eigen的rows
        cv::Mat cv_mat(mat.cols(), mat.rows(), CV_64F, mat.data());
        // 使用与 compute_loss 中相同的模糊参数 (Size(5,5), sigma=1.0)
        // 这一步对于模拟第一种方法的"Radius Search"至关重要
        cv::GaussianBlur(cv_mat, cv_mat, cv::Size(5, 5), 1.0);
      };
      Vec3 delta_points_T, delta_second_points_T, points_T;
      // 创建翘曲事件图像
      int pad_width = width_ + 2 * padding_;
      int pad_height = height_ + 2 * padding_;
      warped_image_.setZero();
      // 初始化梯度和海森矩阵图像
      #pragma omp parallel for
      for (int i = 0; i < 3; ++i)
      {
        iwe_grad_[i].setZero();
      }
      #pragma omp parallel for
      for (int i = 0; i < 9; ++i)
      {
        iwe_hess_[i].setZero();
      }
      double warped_x, warped_y, event_x, event_y;
      Eigen::Matrix<double, 2, 3> grad_warp;
      Mat3 hess_warp;

      // 提取内参，避免在循环中重复访问成员变量
      for (int i = 0; i < n_points_; ++i)
      {
        auto point_ = points_->at(i);
        // Taylor 二阶展开: points * skew_matrix * skew_matrix
        delta_points_T(0) = -parameters[2] * point_(1) + parameters[1] * point_(2);
        delta_points_T(1) = parameters[2] * point_(0) - parameters[0] * point_(2);
        delta_points_T(2) = -parameters[1] * point_(0) + parameters[0] * point_(1);

        // delta_second_points_T(0) = -parameters[2] * delta_points_T(1) + parameters[1] * delta_points_T(2);
        // delta_second_points_T(1) = parameters[2] * delta_points_T(0) - parameters[0] * delta_points_T(2);
        // delta_second_points_T(2) = -parameters[1] * delta_points_T(0) + parameters[0] * delta_points_T(1);
        // 计算 warp 后的 3D 点
        double dt = delta_time_->at(i);
        // double dt_sq = 0.5 * dt * dt;
        points_T(0) = point_(0) + delta_points_T(0) * dt;
        points_T(1) = point_(1) + delta_points_T(1) * dt;
        points_T(2) = point_(2) + delta_points_T(2) * dt;

        // 投影到 2D
        warped_x = points_T(0) / points_T(2) * fx_ + cx_ + padding_;
        warped_y = points_T(1) / points_T(2) * fy_ + cy_ + padding_;
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
          // hessian(i, j) = (pixel_grad_.array() * iwe_hess_[3 * i + j].array()).sum() + (pixel_hess_.array() * iwe_grad_[i].array() * iwe_grad_[j].array()).sum();
        hessian(i, j) = (pixel_hess_.array() * iwe_grad_[i].array() * iwe_grad_[j].array()).sum();  
        }
      }
    }
  };
};