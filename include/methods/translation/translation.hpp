#pragma once

#include "methods/estimator.hpp"

namespace spatiotemporal_vo
{

  class TranslationEstimator : public MotionEstimator
  {
    public:
    using MotionEstimator::MotionEstimator; // 继承所有基类构造函数
    protected:
    void warp_gradient(const Vec3 &parameters, const Vec3 &point, double dt, Eigen::Matrix<double, 2, 3> &grad_warp, Eigen::Matrix3d &hess_warp) const
    {
      double X_Z = point(0) / point(2);
      double Y_Z = point(1) / point(2);
      grad_warp << -fx_ / point(2), 0, fx_ * X_Z / point(2),
          0, -fy_ / point(2), fy_ * Y_Z / point(2);
      grad_warp *= dt;
      hess_warp = grad_warp.row(0).transpose() * grad_warp.row(1) + grad_warp.row(1).transpose() * grad_warp.row(0);
    }
  };
};