#pragma once

#include "methods/translation/translation.hpp"

namespace spatiotemporal_vo
{

  class CMaxTranslation : public TranslationEstimator
  {
    public:
    using TranslationEstimator::TranslationEstimator; // 继承所有基类构造函数
    protected:
    std::string getMethodName() const override { return "CMaxTranslation"; }
    void iwe_gradient(const EigenMat &iwe, double &loss, EigenMat &grad, EigenMat &hess) const override {
      // CMax
      loss = -(iwe.array() - iwe.mean()).square().mean();
      grad = -2 * (iwe.array() - iwe.mean());
      hess = -2 * EigenMat::Ones(iwe.rows(), iwe.cols());
    };
  };
};