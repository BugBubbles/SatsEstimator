#pragma once

#include "methods/6dof/6dof.hpp"

namespace spatiotemporal_vo
{

  class CITESixDof : public SixDofEstimator
  {
  public:
    using SixDofEstimator::SixDofEstimator; // 继承所有基类构造函数
  protected:
    std::string getMethodName() const override { return "CITESixDof"; }
    void iwe_gradient(const EigenMat &iwe, double &loss, EigenMat &grad, EigenMat &hess) const override
    {
      // Ours
      loss = ((iwe.array() + 0.1).log() + std::log(10)).sum() / iwe.sum();
      grad = 1 / (iwe.array() + 0.1);
      hess = -1 / (iwe.array() + 0.1).square();
    }
  };
};