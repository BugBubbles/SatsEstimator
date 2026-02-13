#pragma once

#include "methods/6dof/6dof.hpp"

namespace spatiotemporal_vo
{

  class PolySixDof : public SixDofEstimator
  {
  public:
    using SixDofEstimator::SixDofEstimator; // 继承所有基类构造函数
  protected:
    std::string getMethodName() const override { return "PolySixDof"; }
    void iwe_gradient(const EigenMat &iwe, double &loss, EigenMat &grad, EigenMat &hess) const
    {
      // Ours
      loss = (0.9416 * iwe.array() + (iwe.array() + 0.1).log() + std::log(10)).sum() / iwe.sum();
      grad = 0.9416 + 1 / (iwe.array() + 0.1);
      hess = -1 / (iwe.array() + 0.1).square();
    }
  };
};