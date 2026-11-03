#include "spatiotemporal_vo/common.hpp"
#include "methods/peme_estimator.hpp"

namespace spatiotemporal_vo
{
  cv::Mat PEMEEstimator::getWarpedEventImage(const Vec3 &transVelocity,
                                              const Vec3 &angularVelocity,
                                              EventPacket &warped_bundle,
                                              std::vector<bool> &isInner,
                                              bool use_later_time)
  {
    warped_bundle.clear();
    isInner.assign(eventBundle_.size(), false);
    // 创建三通道16位无符号整型图像
    cv::Mat image = cv::Mat::zeros(height_ + 2 * padding_, width_ + 2 * padding_, CV_16U);
    int n = eventBundle_.size();
    warped_bundle.resize(n);

    double ref_time = use_later_time ? eventBundle_.t.back() : 0.0;
    Vec3 point_first, p3d, rotation_vec, trans_vec;
    for (int i = 0; i < n; ++i)
    {
      double dt = eventBundle_.t[i] - ref_time;
      trans_vec = transVelocity * dt;
      rotation_vec = angularVelocity * dt;
      p3d = eventBundle_.coord_3d[i];
      point_first(0) = trans_vec(0) - rotation_vec(2) * p3d(1) + rotation_vec(1) * p3d(2);
      point_first(1) = trans_vec(1) + rotation_vec(2) * p3d(0) - rotation_vec(0) * p3d(2);
      point_first(2) = trans_vec(2) - rotation_vec(1) * p3d(0) + rotation_vec(0) * p3d(1);

      p3d += point_first;

      // 投影到 2D
      double x = p3d(0) / p3d(2) * fx_ + cx_;
      double y = p3d(1) / p3d(2) * fy_ + cy_;

      warped_bundle.xy[i] = Vec2(x, y);
      warped_bundle.coord_3d[i] = p3d;

      int sampled_x = std::round(x);
      int sampled_y = std::round(y);

      // 检查是否在图像内
      if (sampled_x >= -padding_ && sampled_x <= width_ + padding_ - 1 &&
          sampled_y >= -padding_ && sampled_y <= height_ + padding_ - 1)
      {
        isInner[i] = true;
        image.at<unsigned short>(sampled_y + padding_, sampled_x + padding_) += 1;
      }
      else
      {
        isInner[i] = false;
      }
    }

    return image;
  }
  double PEMEEstimator::overlapRatio(const Vec3 &transVelocity,
                                     const Vec3 &angularVelocity,
                                     const EventPacket &warped_bundle,
                                     const EventPacket &next_warped_bundle)
  {
    // 创建三通道64位浮点型图像
    cv::Mat timesurface = cv::Mat::zeros(height_ + 2 * padding_, width_ + 2 * padding_, CV_64F);
    timesurface.setTo(warped_bundle.t.back() * params_.default_value_factor);
    int n = warped_bundle.size();
    double t_ref = warped_bundle.t.front();

    Vec3 point_first, p3d, rotation_vec, trans_vec;
    // 倒序填充，以获取最小值时间戳
    for (int i = n - 1; i >= 0; i--)
    {
      double dt = warped_bundle.t[i] - t_ref;
      rotation_vec = angularVelocity * dt;
      trans_vec = transVelocity * dt;
      p3d = warped_bundle.coord_3d[i];
      point_first(0) = trans_vec(0) - rotation_vec(2) * p3d(1) + rotation_vec(1) * p3d(2);
      point_first(1) = trans_vec(1) + rotation_vec(2) * p3d(0) - rotation_vec(0) * p3d(2);
      point_first(2) = trans_vec(2) - rotation_vec(1) * p3d(0) + rotation_vec(0) * p3d(1);

      p3d += point_first;

      // 投影到 2D
      double x = p3d(0) / p3d(2) * fx_ + cx_;
      double y = p3d(1) / p3d(2) * fy_ + cy_;

      int sampled_x = std::round(x);
      int sampled_y = std::round(y);
      // 检查是否在图像内
      if (sampled_x >= -padding_ && sampled_x <= width_ + padding_ - 1 &&
          sampled_y >= -padding_ && sampled_y <= height_ + padding_ - 1)
        timesurface.at<double>(sampled_y + padding_, sampled_x + padding_) = dt;
    }
    // 计算重叠度
    int count = 0;
    double ts, mid_ts = warped_bundle.t[n / 2];
    for (int i = 0; i < next_warped_bundle.size(); ++i)
    {
      double dt = next_warped_bundle.t[i] - t_ref;
      rotation_vec = angularVelocity * dt;
      trans_vec = transVelocity * dt;
      p3d = next_warped_bundle.coord_3d[i];
      point_first(0) = trans_vec(0) - rotation_vec(2) * p3d(1) + rotation_vec(1) * p3d(2);
      point_first(1) = trans_vec(1) + rotation_vec(2) * p3d(0) - rotation_vec(0) * p3d(2);
      point_first(2) = trans_vec(2) - rotation_vec(1) * p3d(0) + rotation_vec(0) * p3d(1);

      p3d += point_first;
      // 投影到 2D
      double x = p3d(0) / p3d(2) * fx_ + cx_;
      double y = p3d(1) / p3d(2) * fy_ + cy_;

      int sampled_x = std::round(x);
      int sampled_y = std::round(y);
      // 检查是否在图像内
      if (sampled_x >= -padding_ && sampled_x <= width_ + padding_ - 1 &&
          sampled_y >= -padding_ && sampled_y <= height_ + padding_ - 1)
      {
        ts = timesurface.at<double>(sampled_y + padding_, sampled_x + padding_);
        if (ts < mid_ts)
        {
          count++;
        }
      }
    }

    return static_cast<double>(count) / next_warped_bundle.size();
  }
}
