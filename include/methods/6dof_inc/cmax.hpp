#pragma once

#include "methods/6dof_inc/6dof.hpp"

namespace spatiotemporal_vo
{

  class CMaxSixDofInc : public SixDofIncEstimator
  {
  public:
    using SixDofIncEstimator::SixDofIncEstimator; // 继承所有基类构造函数
  protected:
    std::string getMethodName() const override { return "CMaxSixDofInc"; }
    void iwe_gradient(const EigenMat &iwe, double &loss, EigenMat &grad, EigenMat &hess) const
    {
      // CMax
      loss = -(iwe.array() - iwe.mean()).square().mean();
      grad = -2 * (iwe.array() - iwe.mean());
      hess = -2 * EigenMat::Ones(iwe.rows(), iwe.cols());
    }
  };
};