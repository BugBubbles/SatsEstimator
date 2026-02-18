#include "spatiotemporal_vo/common.hpp"
#include "methods/peme_estimator.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <ctime>
#include <iostream>
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

namespace spatiotemporal_vo
{
  /**
   * @brief Ceres 残差函数
   */
  struct PEMERotation
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PEMERotation(
        const Vec3 &point,
        const int padding,
        double delta_time_early,
        double delta_time_later,
        double fx, double fy, double cx, double cy,
        ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> *interpolator_early_ptr,
        ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> *interpolator_later_ptr)
        : point_(point),
          padding_(padding),
          delta_time_early_(delta_time_early),
          delta_time_later_(delta_time_later),
          fx_(fx), fy_(fy), cx_(cx), cy_(cy),
          interpolator_early_ptr_(interpolator_early_ptr),
          interpolator_later_ptr_(interpolator_later_ptr)
    {
    }

    template <typename T>
    bool operator()(const T *const ag, T *residual) const
    {
      Eigen::Matrix<T, 3, 1> delta_points_T, points_early_T, points_later_T;
      Eigen::Matrix<T, 2, 1> points_2D_early_T, points_2D_later_T;

      // Taylor 二阶展开: points * skew_matrix * skew_matrix
      delta_points_T(0) = -ag[2] * T(point_(1)) + ag[1] * T(point_(2));
      delta_points_T(1) = ag[2] * T(point_(0)) - ag[0] * T(point_(2));
      delta_points_T(2) = -ag[1] * T(point_(0)) + ag[0] * T(point_(1));

      // 计算 warp 后的 3D 点
      T dt_early = T(delta_time_early_);
      T dt_early_sq = T(0.5 * delta_time_early_ * delta_time_early_);
      points_early_T(0) = T(point_(0)) + delta_points_T(0) * dt_early;
      points_early_T(1) = T(point_(1)) + delta_points_T(1) * dt_early;
      points_early_T(2) = T(point_(2)) + delta_points_T(2) * dt_early;

      T dt_later = T(delta_time_later_);
      T dt_later_sq = T(0.5 * delta_time_later_ * delta_time_later_);
      points_later_T(0) = T(point_(0)) + delta_points_T(0) * dt_later;
      points_later_T(1) = T(point_(1)) + delta_points_T(1) * dt_later;
      points_later_T(2) = T(point_(2)) + delta_points_T(2) * dt_later;

      // 投影到 2D
      points_2D_early_T(0) = points_early_T(0) / points_early_T(2) * T(fx_) + T(cx_);
      points_2D_early_T(1) = points_early_T(1) / points_early_T(2) * T(fy_) + T(cy_);

      points_2D_later_T(0) = points_later_T(0) / points_later_T(2) * T(fx_) + T(cx_);
      points_2D_later_T(1) = points_later_T(1) / points_later_T(2) * T(fy_) + T(cy_);

      // 插值获取残差
      T early_loss = T(0), later_loss = T(0);
      interpolator_early_ptr_->Evaluate(points_2D_early_T(1) + T(padding_), points_2D_early_T(0) + T(padding_), &early_loss);
      interpolator_later_ptr_->Evaluate(points_2D_later_T(1) + T(padding_), points_2D_later_T(0) + T(padding_), &later_loss);

      residual[0] = early_loss + later_loss;

      return true;
    }

    Vec3 point_;
    int padding_;
    double delta_time_early_, delta_time_later_, fx_, fy_, cx_, cy_;
    ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> *interpolator_early_ptr_;
    ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> *interpolator_later_ptr_;
  };
  struct PEMETranslation
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PEMETranslation(
        const Vec3 &point,
        const int padding,
        double delta_time_early,
        double delta_time_later,
        double fx, double fy, double cx, double cy,
        ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> *interpolator_early_ptr,
        ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> *interpolator_later_ptr)
        : point_(point),
          padding_(padding),
          delta_time_early_(delta_time_early),
          delta_time_later_(delta_time_later),
          fx_(fx), fy_(fy), cx_(cx), cy_(cy),
          interpolator_early_ptr_(interpolator_early_ptr),
          interpolator_later_ptr_(interpolator_later_ptr)
    {
    }

    template <typename T>
    bool operator()(const T *const ag, T *residual) const
    {
      Eigen::Matrix<T, 3, 1> delta_points_T;
      Eigen::Matrix<T, 3, 1> points_early_T, points_later_T;
      Eigen::Matrix<T, 2, 1> points_2D_early_T, points_2D_later_T;

      // Taylor 二阶展开: points * skew_matrix * skew_matrix
      delta_points_T(0) = ag[0];
      delta_points_T(1) = ag[1];
      delta_points_T(2) = ag[2];

      // 计算 warp 后的 3D 点
      T dt_early = T(delta_time_early_);
      points_early_T = T(point_) + delta_points_T * dt_early;

      T dt_later = T(delta_time_later_);
      points_later_T = T(point_) + delta_points_T * dt_later;

      // 投影到 2D
      points_2D_early_T(0) = points_early_T(0) / points_early_T(2) * T(fx_) + T(cx_);
      points_2D_early_T(1) = points_early_T(1) / points_early_T(2) * T(fy_) + T(cy_);

      points_2D_later_T(0) = points_later_T(0) / points_later_T(2) * T(fx_) + T(cx_);
      points_2D_later_T(1) = points_later_T(1) / points_later_T(2) * T(fy_) + T(cy_);

      // 插值获取残差
      T early_loss = T(0), later_loss = T(0);
      interpolator_early_ptr_->Evaluate(points_2D_early_T(1) + T(padding_), points_2D_early_T(0) + T(padding_), &early_loss);
      interpolator_later_ptr_->Evaluate(points_2D_later_T(1) + T(padding_), points_2D_later_T(0) + T(padding_), &later_loss);

      residual[0] = early_loss + later_loss;

      return true;
    }

    Vec3 point_;
    int padding_;
    double delta_time_early_, delta_time_later_, fx_, fy_, cx_, cy_;
    ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> *interpolator_early_ptr_;
    ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> *interpolator_later_ptr_;
  };

  PEMEEstimator::PEMEEstimator(double fx, double fy, double cx, double cy,
                               int width, int height, const Config &config)
      : width_(width), height_(height), count_(1), fx_(fx), fy_(fy), cx_(cx), cy_(cy)
  {
    // 初始化估计状态
    est_angleAxis_.setZero();
  }

  void PEMEEstimator::projectEventsTo3D(const std::vector<Vec2> &xy,
                                        std::vector<Vec3> &coord_3d,
                                        const std::vector<double> &depth)
  {
    int n = xy.size();
    coord_3d.resize(n);

    for (int i = 0; i < n; ++i)
    {
      double d = depth.empty() ? 1.0 : depth[i];
      double x = (xy[i](0) - cx_) / fx_ * d;
      double y = (xy[i](1) - cy_) / fy_ * d;
      coord_3d[i] = Vec3(x, y, d);
    }
  }

  void PEMEEstimator::getSampledIndices(const cv::Mat &iwe, std::vector<int> &indices, int count,
                                        double start_ratio, double end_ratio)
  {
    cv::RNG rng(static_cast<unsigned>(std::time(nullptr)));
    for (int i = 0; i < count;)
    {
      int sample_idx = rng.uniform(int(eventBundle_.size * start_ratio), int(eventBundle_.size * end_ratio));

      // check valid 8 neighbor hood existed
      int sampled_x = eventBundle_.coord[sample_idx].x() + padding_;
      int sampled_y = eventBundle_.coord[sample_idx].y() + padding_;
      if (sampled_x > 0 && sampled_x < width_ + 2 * padding_ - 1 &&
          sampled_y > 0 && sampled_y < height_ + 2 * padding_ - 1)
      {
      int count = 0;
      for (int j = -1; j < 2; j++)
        for (int k = -1; k < 2; k++)
        {
          count += int(iwe.at<unsigned short>(sampled_y + j, sampled_x + k) > 0);
        }

      if (count > params_.denoise_num)
      {
        indices.push_back(sample_idx);
        i++;
      }
    }
    }
  }
  cv::Mat PEMEEstimator::getWarpedEventImage(const Vec3 &angularVelocity,
                                             EventBundle &warped_bundle,
                                             bool use_later_time)
  {
    warped_bundle.coord.clear();
    warped_bundle.coord_3d.clear();
    warped_bundle.isInner.clear();
    // 创建三通道16位无符号整型图像
    cv::Mat image = cv::Mat::zeros(height_ + 2 * padding_, width_ + 2 * padding_, CV_16U);
    int n = eventBundle_.size;

    warped_bundle.coord.resize(n);
    warped_bundle.coord_3d.resize(n);
    warped_bundle.isInner.resize(n);
    warped_bundle.size = n;

    double ref_time = use_later_time ? eventBundle_.time_delta.back() : 0.0;
    Vec3 point_first, p3d, rotation_vec;
    for (int i = 0; i < n; ++i)
    {
      double dt = eventBundle_.time_delta[i] - ref_time;
      rotation_vec = angularVelocity * dt;
      p3d = eventBundle_.coord_3d[i];
      point_first(0) = -rotation_vec(2) * p3d(1) + rotation_vec(1) * p3d(2);
      point_first(1) = rotation_vec(2) * p3d(0) - rotation_vec(0) * p3d(2);
      point_first(2) = -rotation_vec(1) * p3d(0) + rotation_vec(0) * p3d(1);
      
      p3d += point_first;
        
      // 投影到 2D
      double x = p3d(0) / p3d(2) * fx_ + cx_;
      double y = p3d(1) / p3d(2) * fy_ + cy_;

      warped_bundle.coord[i] = Vec2(x, y);
      warped_bundle.coord_3d[i] = p3d;

      // 检查是否在图像内
      if (x >= -padding_ && x <= width_ + padding_ - 1 &&
          y >= -padding_ && y <= height_ + padding_ - 1)
      {
        warped_bundle.isInner[i] = true;
        image.at<unsigned short>(y + padding_, x + padding_) += 1;
      }
      else
      {
        warped_bundle.isInner[i] = false;
      }
    }

    return image;
  }

  double PEMEEstimator::optimizeMotion()
  {
    double angleAxis[3] = {est_angleAxis_(0), est_angleAxis_(1), est_angleAxis_(2)};
    double best_loss = std::numeric_limits<double>::max();
    best_angleAxis_ = est_angleAxis_;
    // 获取 warp 后的事件图像
    cv::Mat iwe = getWarpedEventImage(Vec3(0, 0, 0), event_warpped_Bundle_);
    for (int iter = 1; iter <= params_.iter_num; ++iter)
    {
      Vec3 eg_angleAxis(angleAxis[0], angleAxis[1], angleAxis[2]);

      // 计算时间表面范围
      double timesurface_range = params_.ts_start +
                                 iter / static_cast<double>(params_.iter_num) * (params_.ts_end - params_.ts_start);

      // 初始化时间表面
      cv::Mat cv_earlier_timesurface = cv::Mat(height_ + 2 * padding_, width_ + 2 * padding_, CV_64F);
      cv::Mat cv_later_timesurface = cv::Mat(height_ + 2 * padding_, width_ + 2 * padding_, CV_64F);

      double early_default_value = eventBundle_.time_delta[static_cast<int>(eventBundle_.size * timesurface_range)];
      cv_earlier_timesurface.setTo(early_default_value * params_.default_value_factor);

      double later_default_value = eventBundle_.time_delta.back() -
                                  eventBundle_.time_delta[static_cast<int>(eventBundle_.size * (1 - timesurface_range))];
      cv_later_timesurface.setTo(later_default_value * params_.default_value_factor);

      // 获取 warp 后的事件图像
      getWarpedEventImage(eg_angleAxis, event_warpped_Bundle_);

      // 构建 earlier 时间表面
      for (int i = event_warpped_Bundle_.size * timesurface_range; i >= 0; --i)
      {
        int sampled_x = std::round(event_warpped_Bundle_.coord[i](0));
        int sampled_y = std::round(event_warpped_Bundle_.coord[i](1));

        if (!event_warpped_Bundle_.isInner[i])
          continue;

        cv_earlier_timesurface.at<double>(sampled_y + padding_, sampled_x + padding_) = eventBundle_.time_delta[i];
      }

      // 获取 later 时间表面
      getWarpedEventImage(eg_angleAxis, event_warpped_Bundle_, true);

      for (int i = event_warpped_Bundle_.size * (1 - timesurface_range);i < event_warpped_Bundle_.size; ++i)
      {
        int sampled_x = std::round(event_warpped_Bundle_.coord[i](0));
        int sampled_y = std::round(event_warpped_Bundle_.coord[i](1));

        if (!event_warpped_Bundle_.isInner[i])
          continue;

        cv_later_timesurface.at<double>(sampled_y + padding_, sampled_x + padding_) = eventBundle_.time_delta.back() - eventBundle_.time_delta[i];
      }

      // 高斯模糊
      cv::GaussianBlur(cv_earlier_timesurface, cv_earlier_timesurface,
                       cv::Size(params_.gaussian_size, params_.gaussian_size),
                       params_.gaussian_sigma);
      cv::GaussianBlur(cv_later_timesurface, cv_later_timesurface,
                       cv::Size(params_.gaussian_size, params_.gaussian_size),
                       params_.gaussian_sigma);

      ceres::Grid2D<double, 1> grid_early(cv_earlier_timesurface.ptr<double>(), 0, height_ + 2 * padding_, 0, width_ + 2 * padding_);
      ceres::Grid2D<double, 1> grid_later(cv_later_timesurface.ptr<double>(), 0, height_ + 2 * padding_, 0, width_ + 2 * padding_);
      auto interpolator_early_ptr = std::make_unique<ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>>(grid_early);
      auto interpolator_later_ptr = std::make_unique<ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>>(grid_later);

      // 采样
      std::vector<int> vec_sampled_idx;
      int samples_count = std::min(params_.sample_count, eventBundle_.size);
      getSampledIndices(iwe, vec_sampled_idx, samples_count, 0.0, 1.0);

      // 构建 Ceres 问题
      ceres::Problem problem;
      // 将vec_sampled_idx打印至文件流中
      for (int sample_idx : vec_sampled_idx)
      {
        double early_time = eventBundle_.time_delta[sample_idx];
        double later_time = eventBundle_.time_delta[sample_idx] - eventBundle_.time_delta.back();

        ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<PEMERotation, 1, 3>(
            new spatiotemporal_vo::PEMERotation(eventBundle_.coord_3d[sample_idx],
                                                            padding_,
                                                            early_time, later_time,
                                                            fx_, fy_, cx_, cy_,
                                                            interpolator_early_ptr.get(),
                                                            interpolator_later_ptr.get()));

        problem.AddResidualBlock(cost_function, nullptr, &angleAxis[0]);
      }

      // Ceres 求解器配置
      ceres::Solver::Options options;
      options.minimizer_progress_to_stdout = false;
      options.logging_type = ceres::SILENT;
      options.num_threads = params_.ceres_iter_thread;
      options.linear_solver_type = ceres::DENSE_QR;
      options.max_num_iterations = params_.ceres_iter_num;

      ceres::Solver::Summary summary;

      // 第一次迭代时评估初始值
      if (iter == 1)
      {
        double cost;
        std::vector<double> residual_vec_old, residual_vec_zero;

        // 测试旧速度
        angleAxis[0] = est_angleAxis_(0);
        angleAxis[1] = est_angleAxis_(1);
        angleAxis[2] = est_angleAxis_(2);
        problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residual_vec_old, nullptr, nullptr);
        double residual_sum_old = std::accumulate(residual_vec_old.begin(), residual_vec_old.end(), 0.0);

        // 测试零速度
        angleAxis[0] = angleAxis[1] = angleAxis[2] = 0;
        problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residual_vec_zero, nullptr, nullptr);
        double residual_sum_zero = std::accumulate(residual_vec_zero.begin(), residual_vec_zero.end(), 0.0);

        if (residual_sum_old < residual_sum_zero)
        {
          angleAxis[0] = est_angleAxis_(0);
          angleAxis[1] = est_angleAxis_(1);
          angleAxis[2] = est_angleAxis_(2);
        }
      }

      // 求解
      ceres::Solve(options, &problem, &summary);
      std::vector<double> residual_vec;
      problem.Evaluate(ceres::Problem::EvaluateOptions(), &best_loss, &residual_vec, nullptr, nullptr);
    }
    // 更新估计值
    est_angleAxis_ = Vec3(angleAxis[0], angleAxis[1], angleAxis[2]);
    return best_loss;
  }

  RegistrationOutput<3> PEMEEstimator::run(const EventPacket& events)
  {
    RegistrationOutput<3> result;

    if (events.empty())
    {
      result.params.setZero();
      result.stats.loss = 0;
      result.stats.num_iterations = 0;
      return result;
    }

    // 初始化事件数据
    eventBundle_.coord = events.xy;
    eventBundle_.polarity = events.polarity;
    eventBundle_.coord_3d = events.coord_3d;
    eventBundle_.time_delta = events.t;
    eventBundle_.size = events.size();
    eventBundle_.first_tstamp = events.t.front();
    eventBundle_.last_tstamp = events.t.back();

    std::cout << '[' << count_ << "] " << "PEME: Processing "
              << eventBundle_.size << " events, duration: "
              << (eventBundle_.last_tstamp - eventBundle_.first_tstamp) << "s" << std::endl;

    // 运行优化
    result.stats.time_taken = omp_get_wtime();
    result.stats.loss = optimizeMotion();

    // 填充结果
    result.params = est_angleAxis_;
    result.stats.num_iterations = count_;
    result.stats.time_taken = omp_get_wtime() - result.stats.time_taken;

    std::cout << '[' << count_ << "] " << "PEME: Estimated angular velocity: "
              << est_angleAxis_.transpose()
              << ", best loss: " << result.stats.loss << ", Optimization took " << result.stats.time_taken << " seconds." << std::endl;
    count_++;
    return result;
  }

} // namespace spatiotemporal_vo