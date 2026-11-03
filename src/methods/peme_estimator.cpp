#include "spatiotemporal_vo/common.hpp"
#include "methods/peme_estimator.hpp"
#include <algorithm>
#include <numeric>
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
      // 计算 warp 后的 3D 点
      T dt_early = T(delta_time_early_);
      points_early_T(0) = point_(0) + ag[0] * dt_early;
      points_early_T(1) = point_(1) + ag[1] * dt_early;
      points_early_T(2) = point_(2) + ag[2] * dt_early;

      T dt_later = T(delta_time_later_);
      points_later_T(0) = point_(0) + ag[0] * dt_later;
      points_later_T(1) = point_(1) + ag[1] * dt_later;
      points_later_T(2) = point_(2) + ag[2] * dt_later;

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
  struct PEMESixDof
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PEMESixDof(
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
      // 计算 warp 后的 3D 点
      T dt_early = T(delta_time_early_);
      points_early_T(0) = point_(0) + (ag[0] - ag[5] * T(point_(1)) + ag[4] * T(point_(2))) * dt_early;
      points_early_T(1) = point_(1) + (ag[1] + ag[5] * T(point_(0)) - ag[3] * T(point_(2))) * dt_early;
      points_early_T(2) = point_(2) + (ag[2] - ag[4] * T(point_(0)) + ag[3] * T(point_(1))) * dt_early;

      T dt_later = T(delta_time_later_);
      points_later_T(0) = point_(0) + (ag[0] - ag[5] * T(point_(1)) + ag[4] * T(point_(2))) * dt_later;
      points_later_T(1) = point_(1) + (ag[1] + ag[5] * T(point_(0)) - ag[3] * T(point_(2))) * dt_later;
      points_later_T(2) = point_(2) + (ag[2] - ag[4] * T(point_(0)) + ag[3] * T(point_(1))) * dt_later;

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
    est_vec3_.setZero();
    est_vec6_.setZero();

    use_dynamic_batch_ = config.data.adaptive_split;
  }

  void PEMEEstimator::getSampledIndices(const cv::Mat &iwe, std::vector<int> &indices, int count,
                                        double start_ratio, double end_ratio)
  {
    indices.clear();
    if (count <= 0 || eventBundle_.empty())
      return;

    const int n = eventBundle_.size();
    const int begin = std::clamp(static_cast<int>(n * start_ratio), 0, n);
    const int end = std::clamp(static_cast<int>(n * end_ratio), begin, n);
    if (begin >= end)
      return;

    // Rejection sampling can loop forever when no event has enough occupied
    // neighbours in the 3x3 denoising window. This is common for the
    // 2000-event dynamic-batch packets. Build the candidate pools once and
    // fall back to all in-image events when the denoising condition is too
    // strict for the current packet.
    std::vector<int> denoised_indices;
    std::vector<int> fallback_indices;
    denoised_indices.reserve(end - begin);
    fallback_indices.reserve(end - begin);

    for (int sample_idx = begin; sample_idx < end; ++sample_idx)
    {
      int sampled_x = eventBundle_.xy[sample_idx].x() + padding_;
      int sampled_y = eventBundle_.xy[sample_idx].y() + padding_;
      if (sampled_x <= 0 || sampled_x >= width_ + 2 * padding_ - 1 ||
          sampled_y <= 0 || sampled_y >= height_ + 2 * padding_ - 1)
        continue;

      fallback_indices.push_back(sample_idx);

      int occupied_neighbors = 0;
      for (int j = -1; j <= 1; ++j)
      {
        for (int k = -1; k <= 1; ++k)
        {
          occupied_neighbors +=
              int(iwe.at<unsigned short>(sampled_y + j, sampled_x + k) > 0);
        }
      }

      if (occupied_neighbors > params_.denoise_num)
        denoised_indices.push_back(sample_idx);
    }

    const std::vector<int> *candidate_indices = &denoised_indices;
    if (candidate_indices->empty())
    {
      candidate_indices = &fallback_indices;
      std::cout << "[PEME] Warning: no samples passed the denoising "
                   "neighbour test; falling back to "
                << candidate_indices->size() << " in-image events."
                << std::endl;
    }

    if (candidate_indices->empty())
      return;

    cv::RNG rng(static_cast<uint64>(cv::getTickCount()));
    indices.reserve(count);
    // Sampling with replacement preserves the requested sample count even
    // when the valid pool is smaller than count.
    for (int i = 0; i < count; ++i)
    {
      const int candidate =
          rng.uniform(0, static_cast<int>(candidate_indices->size()));
      indices.push_back((*candidate_indices)[candidate]);
    }
  }

  double PEMEEstimator::optimizeRotation()
  {
    double angleAxis[3] = {est_vec3_(0), est_vec3_(1), est_vec3_(2)};
    double best_loss = std::numeric_limits<double>::max();
    // 获取 warp 后的事件图像
    std::vector<bool> isInner(eventBundle_.size());
    cv::Mat iwe = getWarpedEventImage(Vec3::Zero(), Vec3::Zero(), event_warpped_Bundle_, isInner);
    for (int iter = 1; iter <= params_.iter_num; ++iter)
    {
      Vec3 eg_angleAxis(angleAxis[0], angleAxis[1], angleAxis[2]);

      // 计算时间表面范围
      double timesurface_range = params_.ts_start +
                                 iter / static_cast<double>(params_.iter_num) * (params_.ts_end - params_.ts_start);

      // 初始化时间表面
      cv::Mat cv_earlier_timesurface = cv::Mat(height_ + 2 * padding_, width_ + 2 * padding_, CV_64F);
      cv::Mat cv_later_timesurface = cv::Mat(height_ + 2 * padding_, width_ + 2 * padding_, CV_64F);

      double early_default_value = eventBundle_.t[static_cast<int>(eventBundle_.size() * timesurface_range)];
      cv_earlier_timesurface.setTo(early_default_value * params_.default_value_factor);

      double later_default_value = eventBundle_.t.back() -
                                   eventBundle_.t[static_cast<int>(eventBundle_.size() * (1 - timesurface_range))];
      cv_later_timesurface.setTo(later_default_value * params_.default_value_factor);

      // 获取 warp 后的事件图像
      getWarpedEventImage(Vec3::Zero(), eg_angleAxis, event_warpped_Bundle_, isInner);

      // 构建 earlier 时间表面
      for (int i = event_warpped_Bundle_.size() * timesurface_range; i >= 0; --i)
      {
        int sampled_x = std::round(event_warpped_Bundle_.xy[i](0));
        int sampled_y = std::round(event_warpped_Bundle_.xy[i](1));

        if (!isInner[i])
          continue;

        cv_earlier_timesurface.at<double>(sampled_y + padding_, sampled_x + padding_) = eventBundle_.t[i];
      }

      // 获取 later 时间表面
      getWarpedEventImage(Vec3::Zero(), eg_angleAxis, event_warpped_Bundle_, isInner, true);

      for (int i = event_warpped_Bundle_.size() * (1 - timesurface_range); i < event_warpped_Bundle_.size(); ++i)
      {
        int sampled_x = std::round(event_warpped_Bundle_.xy[i](0));
        int sampled_y = std::round(event_warpped_Bundle_.xy[i](1));

        if (!isInner[i])
          continue;

        cv_later_timesurface.at<double>(sampled_y + padding_, sampled_x + padding_) = eventBundle_.t.back() - eventBundle_.t[i];
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
      int samples_count = std::min(params_.sample_count, eventBundle_.size());
      getSampledIndices(iwe, vec_sampled_idx, samples_count, 0.0, 1.0);

      // 构建 Ceres 问题
      ceres::Problem problem;
      // 将vec_sampled_idx打印至文件流中
      for (int sample_idx : vec_sampled_idx)
      {
        double early_time = eventBundle_.t[sample_idx];
        double later_time = eventBundle_.t[sample_idx] - eventBundle_.t.back();

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
        angleAxis[0] = est_vec3_(0);
        angleAxis[1] = est_vec3_(1);
        angleAxis[2] = est_vec3_(2);
        problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residual_vec_old, nullptr, nullptr);
        double residual_sum_old = std::accumulate(residual_vec_old.begin(), residual_vec_old.end(), 0.0);

        // 测试零速度
        angleAxis[0] = angleAxis[1] = angleAxis[2] = 0;
        problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residual_vec_zero, nullptr, nullptr);
        double residual_sum_zero = std::accumulate(residual_vec_zero.begin(), residual_vec_zero.end(), 0.0);

        if (residual_sum_old < residual_sum_zero)
        {
          angleAxis[0] = est_vec3_(0);
          angleAxis[1] = est_vec3_(1);
          angleAxis[2] = est_vec3_(2);
        }
      }

      // 求解
      ceres::Solve(options, &problem, &summary);
      std::vector<double> residual_vec;
      problem.Evaluate(ceres::Problem::EvaluateOptions(), &best_loss, &residual_vec, nullptr, nullptr);
    }
    // 更新估计值
    est_vec3_ = Vec3(angleAxis[0], angleAxis[1], angleAxis[2]);
    return best_loss;
  }

  double PEMEEstimator::optimizeTranslation()
  {
    double transAxis[3] = {est_vec3_(0), est_vec3_(1), est_vec3_(2)};
    double best_loss = std::numeric_limits<double>::max();
    // 获取 warp 后的事件图像
    std::vector<bool> isInner(eventBundle_.size());
    cv::Mat iwe = getWarpedEventImage(Vec3::Zero(), Vec3::Zero(), event_warpped_Bundle_, isInner);

    for (int iter = 1; iter <= params_.iter_num; ++iter)
    {
      Vec3 eg_transAxis(transAxis[0], transAxis[1], transAxis[2]);

      // 计算时间表面范围
      double timesurface_range = params_.ts_start +
                                 iter / static_cast<double>(params_.iter_num) * (params_.ts_end - params_.ts_start);

      // 初始化时间表面
      cv::Mat cv_earlier_timesurface = cv::Mat(height_ + 2 * padding_, width_ + 2 * padding_, CV_64F);
      cv::Mat cv_later_timesurface = cv::Mat(height_ + 2 * padding_, width_ + 2 * padding_, CV_64F);

      double early_default_value = eventBundle_.t[static_cast<int>(eventBundle_.size() * timesurface_range)];
      cv_earlier_timesurface.setTo(early_default_value * params_.default_value_factor);

      double later_default_value = eventBundle_.t.back() -
                                   eventBundle_.t[static_cast<int>(eventBundle_.size() * (1 - timesurface_range))];
      cv_later_timesurface.setTo(later_default_value * params_.default_value_factor);

      // 获取 warp 后的事件图像
      getWarpedEventImage(eg_transAxis, Vec3::Zero(), event_warpped_Bundle_, isInner);

      // 构建 earlier 时间表面
      for (int i = event_warpped_Bundle_.size() * timesurface_range; i >= 0; --i)
      {
        int sampled_x = std::round(event_warpped_Bundle_.xy[i](0));
        int sampled_y = std::round(event_warpped_Bundle_.xy[i](1));

        if (!isInner[i])
          continue;

        cv_earlier_timesurface.at<double>(sampled_y + padding_, sampled_x + padding_) = eventBundle_.t[i];
      }

      // 获取 later 时间表面
      getWarpedEventImage(eg_transAxis, Vec3::Zero(), event_warpped_Bundle_, isInner, true);

      for (int i = event_warpped_Bundle_.size() * (1 - timesurface_range); i < event_warpped_Bundle_.size(); ++i)
      {
        int sampled_x = std::round(event_warpped_Bundle_.xy[i](0));
        int sampled_y = std::round(event_warpped_Bundle_.xy[i](1));

        if (!isInner[i])
          continue;

        cv_later_timesurface.at<double>(sampled_y + padding_, sampled_x + padding_) = eventBundle_.t.back() - eventBundle_.t[i];
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
      int samples_count = std::min(params_.sample_count, eventBundle_.size());
      getSampledIndices(iwe, vec_sampled_idx, samples_count, 0.0, 1.0);

      // 构建 Ceres 问题
      ceres::Problem problem;
      // 将vec_sampled_idx打印至文件流中
      for (int sample_idx : vec_sampled_idx)
      {
        double early_time = eventBundle_.t[sample_idx];
        double later_time = eventBundle_.t[sample_idx] - eventBundle_.t.back();

        ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<PEMETranslation, 1, 3>(
            new spatiotemporal_vo::PEMETranslation(eventBundle_.coord_3d[sample_idx],
                                                   padding_,
                                                   early_time, later_time,
                                                   fx_, fy_, cx_, cy_,
                                                   interpolator_early_ptr.get(),
                                                   interpolator_later_ptr.get()));

        problem.AddResidualBlock(cost_function, nullptr, &transAxis[0]);
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
        transAxis[0] = est_vec3_(0);
        transAxis[1] = est_vec3_(1);
        transAxis[2] = est_vec3_(2);
        problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residual_vec_old, nullptr, nullptr);
        double residual_sum_old = std::accumulate(residual_vec_old.begin(), residual_vec_old.end(), 0.0);

        // 测试零速度
        transAxis[0] = transAxis[1] = transAxis[2] = 0;
        problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residual_vec_zero, nullptr, nullptr);
        double residual_sum_zero = std::accumulate(residual_vec_zero.begin(), residual_vec_zero.end(), 0.0);

        if (residual_sum_old < residual_sum_zero)
        {
          transAxis[0] = est_vec3_(0);
          transAxis[1] = est_vec3_(1);
          transAxis[2] = est_vec3_(2);
        }
      }

      // 求解
      ceres::Solve(options, &problem, &summary);
      std::vector<double> residual_vec;
      problem.Evaluate(ceres::Problem::EvaluateOptions(), &best_loss, &residual_vec, nullptr, nullptr);
    }
    // 更新估计值
    est_vec3_ = Vec3(transAxis[0], transAxis[1], transAxis[2]);
    return best_loss;
  }

  double PEMEEstimator::optimizeSixDof()
  {
    double trans_rot_vec[6] = {est_vec6_(0), est_vec6_(1), est_vec6_(2), est_vec6_(3), est_vec6_(4), est_vec6_(5)};
    double best_loss = std::numeric_limits<double>::max();
    // 获取 warp 后的事件图像
    std::vector<bool> isInner(eventBundle_.size());
    cv::Mat iwe = getWarpedEventImage(Vec3::Zero(), Vec3::Zero(), event_warpped_Bundle_, isInner);
    for (int iter = 1; iter <= params_.iter_num; ++iter)
    {
      std::cout << "[PEME] 6DoF optimization iteration " << iter << "/"
                << params_.iter_num << std::endl;
      Vec6 eg_vec6;
      eg_vec6 << trans_rot_vec[0], trans_rot_vec[1], trans_rot_vec[2], trans_rot_vec[3], trans_rot_vec[4], trans_rot_vec[5];

      // 计算时间表面范围
      double timesurface_range = params_.ts_start +
                                 iter / static_cast<double>(params_.iter_num) * (params_.ts_end - params_.ts_start);

      // 初始化时间表面
      cv::Mat cv_earlier_timesurface = cv::Mat(height_ + 2 * padding_, width_ + 2 * padding_, CV_64F);
      cv::Mat cv_later_timesurface = cv::Mat(height_ + 2 * padding_, width_ + 2 * padding_, CV_64F);

      double early_default_value = eventBundle_.t[static_cast<int>(eventBundle_.size() * timesurface_range)];
      cv_earlier_timesurface.setTo(early_default_value * params_.default_value_factor);

      double later_default_value = eventBundle_.t.back() -
                                   eventBundle_.t[static_cast<int>(eventBundle_.size() * (1 - timesurface_range))];
      cv_later_timesurface.setTo(later_default_value * params_.default_value_factor);

      // 获取 warp 后的事件图像
      getWarpedEventImage(eg_vec6.head<3>(), eg_vec6.tail<3>(), event_warpped_Bundle_, isInner);

      // 构建 earlier 时间表面
      for (int i = event_warpped_Bundle_.size() * timesurface_range; i >= 0; --i)
      {
        int sampled_x = std::round(event_warpped_Bundle_.xy[i](0));
        int sampled_y = std::round(event_warpped_Bundle_.xy[i](1));

        if (!isInner[i])
          continue;

        cv_earlier_timesurface.at<double>(sampled_y + padding_, sampled_x + padding_) = eventBundle_.t[i];
      }

      // 获取 later 时间表面
      getWarpedEventImage(eg_vec6.head<3>(), eg_vec6.tail<3>(), event_warpped_Bundle_, isInner, true);

      for (int i = event_warpped_Bundle_.size() * (1 - timesurface_range); i < event_warpped_Bundle_.size(); ++i)
      {
        int sampled_x = std::round(event_warpped_Bundle_.xy[i](0));
        int sampled_y = std::round(event_warpped_Bundle_.xy[i](1));

        if (!isInner[i])
          continue;

        cv_later_timesurface.at<double>(sampled_y + padding_, sampled_x + padding_) = eventBundle_.t.back() - eventBundle_.t[i];
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
      int samples_count = std::min(params_.sample_count, eventBundle_.size());
      getSampledIndices(iwe, vec_sampled_idx, samples_count, 0.0, 1.0);

      if (vec_sampled_idx.empty())
      {
        std::cerr << "[PEME] No valid samples available; stopping 6DoF "
                     "optimization for this packet."
                  << std::endl;
        break;
      }

      // 构建 Ceres 问题
      ceres::Problem problem;
      // 将vec_sampled_idx打印至文件流中
      for (int sample_idx : vec_sampled_idx)
      {
        double early_time = eventBundle_.t[sample_idx];
        double later_time = eventBundle_.t[sample_idx] - eventBundle_.t.back();

        ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<PEMESixDof, 1, 6>(
            new spatiotemporal_vo::PEMESixDof(eventBundle_.coord_3d[sample_idx],
                                              padding_,
                                              early_time, later_time,
                                              fx_, fy_, cx_, cy_,
                                              interpolator_early_ptr.get(),
                                              interpolator_later_ptr.get()));

        problem.AddResidualBlock(cost_function, nullptr, &trans_rot_vec[0]);
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
        eg_vec6[0] = est_vec6_(0);
        eg_vec6[1] = est_vec6_(1);
        eg_vec6[2] = est_vec6_(2);
        eg_vec6[3] = est_vec6_(3);
        eg_vec6[4] = est_vec6_(4);
        eg_vec6[5] = est_vec6_(5);
        problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residual_vec_old, nullptr, nullptr);
        double residual_sum_old = std::accumulate(residual_vec_old.begin(), residual_vec_old.end(), 0.0);

        // 测试零速度
        eg_vec6[0] = eg_vec6[1] = eg_vec6[2] = 0;
        eg_vec6[3] = eg_vec6[4] = eg_vec6[5] = 0;
        problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residual_vec_zero, nullptr, nullptr);
        double residual_sum_zero = std::accumulate(residual_vec_zero.begin(), residual_vec_zero.end(), 0.0);

        if (residual_sum_old < residual_sum_zero)
        {
          eg_vec6[0] = est_vec6_(0);
          eg_vec6[1] = est_vec6_(1);
          eg_vec6[2] = est_vec6_(2);
          eg_vec6[3] = est_vec6_(3);
          eg_vec6[4] = est_vec6_(4);
          eg_vec6[5] = est_vec6_(5);
        }
      }

      // 求解
      ceres::Solve(options, &problem, &summary);
      std::vector<double> residual_vec;
      problem.Evaluate(ceres::Problem::EvaluateOptions(), &best_loss, &residual_vec, nullptr, nullptr);
    }
    // 更新估计值
    est_vec6_ << trans_rot_vec[0], trans_rot_vec[1], trans_rot_vec[2], trans_rot_vec[3], trans_rot_vec[4], trans_rot_vec[5];
    return best_loss;
  }

  RegistrationOutput<3> PEMEEstimator::runRotation(const EventPacket &events)
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
    eventBundle_ = events;

    std::cout << '[' << count_ << "] " << "PEME: Processing "
              << eventBundle_.size() << " events, duration: "
              << (eventBundle_.t_end - eventBundle_.t_start) << "s" << std::endl;

    // 运行优化
    result.stats.time_taken = omp_get_wtime();
    result.stats.loss = optimizeRotation();

    // 填充结果
    result.params = est_vec3_;
    result.stats.num_iterations = count_;
    result.stats.time_taken = omp_get_wtime() - result.stats.time_taken;

    std::cout << '[' << count_ << "] " << "PEME: Estimated angular velocity: "
              << est_vec3_.transpose()
              << ", best loss: " << result.stats.loss << ", Optimization took " << result.stats.time_taken << " seconds." << std::endl;
    count_++;
    return result;
  }

  RegistrationOutput<3> PEMEEstimator::runTranslation(const EventPacket &events)
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
    eventBundle_ = events;

    std::cout << '[' << count_ << "] " << "PEME: Processing "
              << eventBundle_.size() << " events, duration: "
              << (eventBundle_.t_end - eventBundle_.t_start) << "s" << std::endl;

    // 运行优化
    result.stats.time_taken = omp_get_wtime();
    result.stats.loss = optimizeTranslation();

    // 填充结果
    result.params = est_vec3_;
    result.stats.num_iterations = count_;
    result.stats.time_taken = omp_get_wtime() - result.stats.time_taken;

    std::cout << '[' << count_ << "] " << "PEME: Estimated translation velocity: "
              << est_vec3_.transpose()
              << ", best loss: " << result.stats.loss << ", Optimization took " << result.stats.time_taken << " seconds." << std::endl;
    count_++;
    return result;
  }

  RegistrationOutput<6> PEMEEstimator::runSixDof(const EventPacket &events)
  {
    RegistrationOutput<6> result;

    if (events.empty())
    {
      result.params.setZero();
      result.stats.loss = 0;
      result.stats.num_iterations = 0;
      return result;
    }

    // 初始化事件数据
    eventBundle_ = events;

    std::cout << '[' << count_ << "] " << "PEME: Processing "
              << eventBundle_.size() << " events, duration: "
              << (eventBundle_.t_end - eventBundle_.t_start) << "s" << std::endl;

    // 运行优化
    result.stats.time_taken = omp_get_wtime();
    result.stats.loss = optimizeSixDof();

    // 填充结果
    result.params = est_vec6_;
    result.stats.num_iterations = count_;
    result.stats.time_taken = omp_get_wtime() - result.stats.time_taken;

    std::cout << '[' << count_ << "] " << "PEME: Estimated translation and angular velocity: "
              << est_vec6_.transpose()
              << ", best loss: " << result.stats.loss << ", Optimization took " << result.stats.time_taken << " seconds." << std::endl;
    count_++;
    return result;
  }
} // namespace spatiotemporal_vo
