#pragma once

#include "methods/translation/translation.hpp"

namespace spatiotemporal_vo
{

  class PolyTranslation : public TranslationEstimator
  {
  public:
    using TranslationEstimator::TranslationEstimator; // 继承所有基类构造函数
  protected:
    std::string getMethodName() const override { return "PolyTranslation"; }
    void iwe_gradient(const EigenMat &iwe, double &loss, EigenMat &grad, EigenMat &hess) const
    {
      loss = ((iwe.array() + 0.1).log() + std::log(10)).sum() / iwe.sum();
      grad = 1 / (iwe.array() + 0.1);
      hess = -1 / (iwe.array() + 0.1).square();
    }
  };
};