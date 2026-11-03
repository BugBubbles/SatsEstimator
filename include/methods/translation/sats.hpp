#pragma once

#include "methods/translation/translation.hpp"

namespace spatiotemporal_vo
{

  class SATSTranslation : public TranslationEstimator
  {
  public:
    using TranslationEstimator::TranslationEstimator; // 继承所有基类构造函数
  protected:
    std::string getMethodName() const override { return "SATSTranslation"; }
    void iwe_gradient(const EigenMat &iwe, double &loss, EigenMat &grad, EigenMat &hess) const override
    {
      loss = ((iwe.array() + 0.1).log() + std::log(10)).sum() / iwe.sum();
      grad = 1 / (iwe.array() + 0.1);
      hess = -1 / (iwe.array() + 0.1).square();
    }
  };
};