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
    void warp_gradient(const Vec3 &parameters, const Vec3 &point, double dt, Eigen::Matrix<double, 2, 3> &grad_warp, Eigen::Matrix3d &hess_warp) const
    {
      double X_Z = point(0) / point(2);
      double Y_Z = point(1) / point(2);
      grad_warp << fx_ * X_Z * Y_Z, -fx_ * (1 + X_Z * X_Z), fx_ * Y_Z,
          fy_ * (1 + Y_Z * Y_Z), -fy_ * X_Z * Y_Z, -fy_ * X_Z;
      grad_warp *= dt;
      hess_warp = grad_warp.row(0).transpose() * grad_warp.row(1) + grad_warp.row(1).transpose() * grad_warp.row(0);
    }
  };
};