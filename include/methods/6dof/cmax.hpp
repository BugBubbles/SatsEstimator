#pragma once

#include "methods/6dof/6dof.hpp"

namespace spatiotemporal_vo
{

  class CMaxSixDof : public SixDofEstimator
  {
  public:
    using SixDofEstimator::SixDofEstimator; // 继承所有基类构造函数
  protected:
    std::string getMethodName() const override { return "CMaxSixDof"; }
    void iwe_gradient(const EigenMat &iwe, double &loss, EigenMat &grad, EigenMat &hess) const
    {
      // CMax
      loss = -(iwe.array() - iwe.mean()).square().mean();
      grad = -2 * (iwe.array() - iwe.mean());
      hess = -2 * EigenMat::Ones(iwe.rows(), iwe.cols());
    }
  };
};