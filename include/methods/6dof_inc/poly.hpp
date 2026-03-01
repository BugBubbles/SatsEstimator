#pragma once

#include "methods/6dof_inc/6dof.hpp"

namespace spatiotemporal_vo
{

  class PolySixDofInc : public SixDofIncEstimator
  {
  public:
    using SixDofIncEstimator::SixDofIncEstimator; // 继承所有基类构造函数
  protected:
    std::string getMethodName() const override { return "PolySixDofInc"; }
    void iwe_gradient(const EigenMat &iwe, double &loss, EigenMat &grad, EigenMat &hess) const
    {
      // Ours
      // loss = (0.9416 * iwe.array() + (iwe.array() + 0.1).log() + std::log(10) - 0.5 * iwe.array().pow(2)).sum() / iwe.sum();
      // grad = 0.9416 + 1 / (iwe.array() + 0.1) - iwe.array();
      // hess = -1 / (iwe.array() + 0.1).square() - 1.0;
      // 用均值平衡偏移点的版本
      EigenMat balanced_iwe = (iwe.array() - iwe.mean()).matrix();
      EigenMat balanced_iwe_power = balanced_iwe.array().square().matrix();
      loss = -(0.9416 * balanced_iwe_power.array() + (balanced_iwe_power.array() + 0.1).log() + std::log(10)).mean();
      grad = -(0.9416 + 1 / (balanced_iwe_power.array() + 0.1)) * 2 * balanced_iwe.array();
      hess = -(-1 / (balanced_iwe_power.array() + 0.1).square()) * 2 * balanced_iwe_power.array() - (0.9416 + 1 / (balanced_iwe_power.array() + 0.1)) * 2;
    }
  };
};