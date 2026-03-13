#pragma once

#include "methods/translation/translation.hpp"
#ifdef BOOST_AVAILABLE
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#endif
#include <cmath>
namespace spatiotemporal_vo
{
#ifndef BOOST_AVAILABLE
  double digamma(double x)
  {
    double result = 0.0;
    while (x < 6.0)
    {
      result -= 1.0 / x;
      x += 1.0;
    }
    double xx = 1.0 / x;
    double xx2 = xx * xx;
    double xx4 = xx2 * xx2;
    result += std::log(x) - 0.5 * xx - xx2 / 12.0 + xx4 / 120.0;
    return result;
  };
  double trigamma(double x)
  {
    double result = 0.0;
    while (x < 6.0)
    {
      result += 1.0 / (x * x);
      x += 1.0;
    }
    double xx = 1.0 / x;
    double xx2 = xx * xx;
    double xx4 = xx2 * xx2;
    result += xx + 0.5 * xx2 + (1.0 / 6.0) * xx2 * xx - (1.0 / 30.0) * xx4 * xx;
    return result;
  }
#endif
  class STPPPTranslation : public TranslationEstimator
  {
  public:
    using TranslationEstimator::TranslationEstimator; // 继承所有基类构造函数
  protected:
    std::string getMethodName() const override { return "STPPPTranslation"; }
    void iwe_gradient(const EigenMat &iwe, double &loss, EigenMat &grad, EigenMat &hess) const override
    {
      // STPPP
      auto stppp_loss = [](double x, double r = 0.1, double q = 0.39)
      {
#ifdef BOOST_AVAILABLE
        return -boost::math::lgamma(x + r) + boost::math::lgamma(x + 1) + boost::math::lgamma(r) - r * std::log(q) - x * std::log(1 - q);
#else
        return -std::lgamma(x + r) + std::lgamma(x + 1) + std::lgamma(r) - r * std::log(q) - x * std::log(1 - q);
#endif
      };
      auto stppp_grad = [](double x, double r = 0.1, double q = 0.39)
      {
#ifdef BOOST_AVAILABLE
        return -boost::math::digamma(x + r) + boost::math::digamma(x + 1) - std::log(1 - q);
#else
        return -digamma(x + r) + digamma(x + 1) - std::log(1 - q);
#endif
      };
      auto stppp_hess = [](double x, double r = 0.1, double q = 0.39)
      {
#ifdef BOOST_AVAILABLE
        return -boost::math::trigamma(x + r) + boost::math::trigamma(x + 1);
#else
        return -trigamma(x + r) + trigamma(x + 1);
#endif
      };
      loss = iwe.unaryExpr(stppp_loss).sum() / iwe.sum();
      grad = iwe.unaryExpr(stppp_grad);
      hess = iwe.unaryExpr(stppp_hess);
    }
  };

};