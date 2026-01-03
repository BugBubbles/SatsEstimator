#include "spatiotemporal_vo/common.hpp"
#include "methods/pemep_estimator.hpp"
#include <sophus/so3.hpp>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <iostream>
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

namespace spatiotemporal_vo
{
  /**
   * @brief Ceres 残差函数
   */
  struct PEMEPlus
  {
    PEMEPlus(
        const Vec3 &point,
        double delta_time_early,
        double delta_time_later,
        const Eigen::Matrix3d &K,
        ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>> *interpolator_early_ptr,
        ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>> *interpolator_later_ptr,
        ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>> *iwe_interpolator_early_ptr,
        ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>> *iwe_interpolator_later_ptr,
        const Vec3 &last_est_var)
        : point_(point),
          delta_time_early_(delta_time_early),
          delta_time_later_(delta_time_later),
          intrinsic_(K),
          interpolator_early_ptr_(interpolator_early_ptr),
          interpolator_later_ptr_(interpolator_later_ptr),
          iwe_interpolator_early_ptr_(iwe_interpolator_early_ptr),
          iwe_interpolator_later_ptr_(iwe_interpolator_later_ptr),
          last_est_var_(last_est_var)
    {
    }

    template <typename T>
    bool operator()(const T *const ag, T *residual) const
    {
      Eigen::Matrix<T, 3, 1> delta_points_T, delta_second_points_T;
      Eigen::Matrix<T, 3, 1> points_early_T, points_later_T;
      Eigen::Matrix<T, 2, 1> points_2D_early_T, points_2D_later_T;

      // Taylor 二阶展开: points * skew_matrix * skew_matrix
      delta_points_T(0) = -ag[2] * T(point_(1)) + ag[1] * T(point_(2));
      delta_points_T(1) = ag[2] * T(point_(0)) - ag[0] * T(point_(2));
      delta_points_T(2) = -ag[1] * T(point_(0)) + ag[0] * T(point_(1));

      delta_second_points_T(0) = -ag[2] * delta_points_T(1) + ag[1] * delta_points_T(2);
      delta_second_points_T(1) = ag[2] * delta_points_T(0) - ag[0] * delta_points_T(2);
      delta_second_points_T(2) = -ag[1] * delta_points_T(0) + ag[0] * delta_points_T(1);

      // 计算 warp 后的 3D 点
      T dt_early = T(delta_time_early_);
      T dt_early_sq = T(0.5 * delta_time_early_ * delta_time_early_);
      points_early_T(0) = T(point_(0)) + delta_points_T(0) * dt_early + delta_second_points_T(0) * dt_early_sq;
      points_early_T(1) = T(point_(1)) + delta_points_T(1) * dt_early + delta_second_points_T(1) * dt_early_sq;
      points_early_T(2) = T(point_(2)) + delta_points_T(2) * dt_early + delta_second_points_T(2) * dt_early_sq;

      T dt_later = T(delta_time_later_);
      T dt_later_sq = T(0.5 * delta_time_later_ * delta_time_later_);
      points_later_T(0) = T(point_(0)) + delta_points_T(0) * dt_later + delta_second_points_T(0) * dt_later_sq;
      points_later_T(1) = T(point_(1)) + delta_points_T(1) * dt_later + delta_second_points_T(1) * dt_later_sq;
      points_later_T(2) = T(point_(2)) + delta_points_T(2) * dt_later + delta_second_points_T(2) * dt_later_sq;

      // 投影到 2D
      points_2D_early_T(0) = points_early_T(0) / points_early_T(2) * T(intrinsic_(0, 0)) + T(intrinsic_(0, 2));
      points_2D_early_T(1) = points_early_T(1) / points_early_T(2) * T(intrinsic_(1, 1)) + T(intrinsic_(1, 2));

      points_2D_later_T(0) = points_later_T(0) / points_later_T(2) * T(intrinsic_(0, 0)) + T(intrinsic_(0, 2));
      points_2D_later_T(1) = points_later_T(1) / points_later_T(2) * T(intrinsic_(1, 1)) + T(intrinsic_(1, 2));

      // 插值获取残差
      T early_loss, later_loss, iwe_early, iwe_later;
      iwe_interpolator_early_ptr_->Evaluate(points_2D_early_T(1), points_2D_early_T(0), &iwe_early);
      iwe_interpolator_later_ptr_->Evaluate(points_2D_later_T(1), points_2D_later_T(0), &iwe_later);
      interpolator_early_ptr_->Evaluate(points_2D_early_T(1), points_2D_early_T(0), &early_loss);
      interpolator_later_ptr_->Evaluate(points_2D_later_T(1), points_2D_later_T(0), &later_loss);

      residual[0] = exp(-iwe_early) * (early_loss + later_loss);

      return true;
    }

    Vec3 point_;
    double delta_time_early_, delta_time_later_;
    Eigen::Matrix3d intrinsic_;
    ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>> *interpolator_early_ptr_;
    ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>> *interpolator_later_ptr_;
    ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>> *iwe_interpolator_early_ptr_;
    ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>> *iwe_interpolator_later_ptr_;
    Vec3 last_est_var_;
  };

  PEMEPlusEstimator::PEMEPlusEstimator(double fx, double fy, double cx, double cy,
                                       int width, int height, const Config &config)
      : width_(width), height_(height), count_(1)
  {
    // 初始化相机内参矩阵
    K_ << fx, 0, cx,
        0, fy, cy,
        0, 0, 1;

    // 初始化估计状态
    est_angleAxis_.setZero();
    last_est_var_.setZero();

    // 从配置中加载参数
    params_.ts_start = 0.8;
    params_.ts_end = 0.8;
    params_.sample_count = 1000;
    params_.iter_num = 10;
    params_.ceres_iter_num = 10;
    params_.ceres_iter_thread = 8;
    params_.gaussian_size = 5;
    params_.gaussian_sigma = 1.0f;
    params_.default_value_factor = 1.0f;
  }

  void PEMEPlusEstimator::projectEventsTo3D(const std::vector<Vec2> &xy,
                                            std::vector<Vec3> &coord_3d,
                                            const std::vector<double> &depth)
  {
    int n = xy.size();
    coord_3d.resize(n);

    double fx = K_(0, 0), fy = K_(1, 1);
    double cx = K_(0, 2), cy = K_(1, 2);

    for (int i = 0; i < n; ++i)
    {
      double x = (xy[i](0) - cx) / fx;
      double y = (xy[i](1) - cy) / fy;
      coord_3d[i] = Vec3(x, y, depth.empty() ? 1.0 : depth[i]);
      coord_3d[i].normalize();
    }
  }

  void PEMEPlusEstimator::getSampledIndices(std::vector<int> &indices, int count,
                                            double start_ratio, double end_ratio)
  {
    int start_idx = static_cast<int>(eventBundle_.size * start_ratio);
    int end_idx = static_cast<int>(eventBundle_.size * end_ratio);
    int range = end_idx - start_idx;

    if (range <= 0 || count <= 0)
    {
      indices.clear();
      return;
    }

    count = std::min(count, range);
    indices.resize(count);

    // 随机采样
    std::vector<int> all_indices(range);
    std::iota(all_indices.begin(), all_indices.end(), start_idx);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(all_indices.begin(), all_indices.end(), gen);

    std::copy(all_indices.begin(), all_indices.begin() + count, indices.begin());
    std::sort(indices.begin(), indices.end());
  }

  cv::Mat PEMEPlusEstimator::getWarpedEventImage(const Vec3 &angularVelocity,
                                                 EventBundle &warped_bundle,
                                                 bool use_later_time)
  {
    // 创建单通道32位浮点图像
    cv::Mat image = cv::Mat::zeros(height_, width_, CV_32FC1);
    int n = eventBundle_.size;

    warped_bundle.coord.resize(n);
    warped_bundle.coord_3d.resize(n);
    warped_bundle.isInner.resize(n);
    warped_bundle.size = n;

    double ref_time = use_later_time ? eventBundle_.time_delta.back() : 0.0;

    for (int i = 0; i < n; ++i)
    {
      double dt = eventBundle_.time_delta[i] - ref_time;
      Vec3 rotation_vec = angularVelocity * dt;
      Sophus::SO3d R = Sophus::SO3d::exp(rotation_vec);

      Vec3 p3d = R.matrix() * event_undis_Bundle_.coord_3d[i];
      warped_bundle.coord_3d[i] = p3d;

      // 投影到 2D
      double x = p3d(0) / p3d(2) * K_(0, 0) + K_(0, 2);
      double y = p3d(1) / p3d(2) * K_(1, 1) + K_(1, 2);

      warped_bundle.coord[i] = Vec2(x, y);

      // 检查是否在图像内
      int xi = static_cast<int>(std::round(x));
      int yi = static_cast<int>(std::round(y));
      warped_bundle.isInner[i] = (xi >= 0 && xi < width_ && yi >= 0 && yi < height_);
      float dx = x - xi;
      float dy = y - yi;

      if (warped_bundle.isInner[i])
      {
        // 获取指向该像素的指针
        uint16_t *left_top = image.ptr<uint16_t>(yi, xi);
        uint16_t *right_top = (xi + 1 < width_) ? image.ptr<uint16_t>(yi, xi + 1) : nullptr;
        uint16_t *left_bottom = (yi + 1 < height_) ? image.ptr<uint16_t>(yi + 1, xi) : nullptr;
        uint16_t *right_bottom = (xi + 1 < width_ && yi + 1 < height_) ? image.ptr<uint16_t>(yi + 1, xi + 1) : nullptr;
        // 双线性插值分配事件值
        left_top[0] += static_cast<uint16_t>((1 - dx) * (1 - dy));
        if (right_top)
          right_top[0] += static_cast<uint16_t>(dx * (1 - dy));
        if (left_bottom)
          left_bottom[0] += static_cast<uint16_t>((1 - dx) * dy);
        if (right_bottom)
          right_bottom[0] += static_cast<uint16_t>(dx * dy);
      }
    }

    return image;
  }

  double PEMEPlusEstimator::optimizeMotion()
  {
    double angleAxis[3] = {est_angleAxis_(0), est_angleAxis_(1), est_angleAxis_(2)};
    double best_loss = std::numeric_limits<double>::max();
    best_angleAxis_ = est_angleAxis_;

    for (int iter = 1; iter <= params_.iter_num; ++iter)
    {
      Vec3 eg_angleAxis(angleAxis[0], angleAxis[1], angleAxis[2]);

      // 计算时间表面范围
      double timesurface_range = params_.ts_start +
                                 iter / static_cast<double>(params_.iter_num) * (params_.ts_end - params_.ts_start);

      // 初始化时间表面
      cv::Mat cv_earlier_timesurface = cv::Mat(height_, width_, CV_32FC1);
      cv::Mat cv_later_timesurface = cv::Mat(height_, width_, CV_32FC1);

      float early_default_value = eventBundle_.time_delta[static_cast<int>(eventBundle_.size * timesurface_range)];
      cv_earlier_timesurface.setTo(early_default_value * params_.default_value_factor);

      float later_default_value = eventBundle_.time_delta.back() -
                                  eventBundle_.time_delta[static_cast<int>(eventBundle_.size * (1 - timesurface_range))];
      cv_later_timesurface.setTo(later_default_value * params_.default_value_factor);

      // 获取 warp 后的事件图像
      cv::Mat iwe_early = getWarpedEventImage(eg_angleAxis, event_warpped_Bundle_);

      // 构建 earlier 时间表面
      for (int i = static_cast<int>(event_warpped_Bundle_.size * timesurface_range); i >= 0; --i)
      {
        int sampled_x = static_cast<int>(std::round(event_warpped_Bundle_.coord[i](0)));
        int sampled_y = static_cast<int>(std::round(event_warpped_Bundle_.coord[i](1)));

        if (!event_warpped_Bundle_.isInner[i])
          continue;

        cv_earlier_timesurface.at<float>(sampled_y, sampled_x) = eventBundle_.time_delta[i];
      }

      // 获取 later 时间表面
      cv::Mat iwe_later = getWarpedEventImage(eg_angleAxis, event_warpped_Bundle_, true);

      for (int i = static_cast<int>(event_warpped_Bundle_.size * (1 - timesurface_range));
           i < event_warpped_Bundle_.size; ++i)
      {
        int sampled_x = static_cast<int>(std::round(event_warpped_Bundle_.coord[i](0)));
        int sampled_y = static_cast<int>(std::round(event_warpped_Bundle_.coord[i](1)));

        if (!event_warpped_Bundle_.isInner[i])
          continue;

        cv_later_timesurface.at<float>(sampled_y, sampled_x) =
            eventBundle_.time_delta.back() - eventBundle_.time_delta[i];
      }

      // 高斯模糊
      cv::Mat cv_earlier_timesurface_blur, cv_later_timesurface_blur;
      cv::GaussianBlur(cv_earlier_timesurface, cv_earlier_timesurface_blur,
                       cv::Size(params_.gaussian_size, params_.gaussian_size),
                       params_.gaussian_sigma);
      cv::GaussianBlur(cv_later_timesurface, cv_later_timesurface_blur,
                       cv::Size(params_.gaussian_size, params_.gaussian_size),
                       params_.gaussian_sigma);
      cv::GaussianBlur(iwe_early, iwe_early,
                       cv::Size(params_.gaussian_size, params_.gaussian_size),
                       params_.gaussian_sigma);
      cv::GaussianBlur(iwe_later, iwe_later,
                       cv::Size(params_.gaussian_size, params_.gaussian_size),
                       params_.gaussian_sigma);

      // 转换为 Ceres 插值器
      std::vector<float> line_grid_early(
          reinterpret_cast<float *>(cv_earlier_timesurface_blur.data),
          reinterpret_cast<float *>(cv_earlier_timesurface_blur.data) + height_ * width_);
      std::vector<float> line_grid_later(
          reinterpret_cast<float *>(cv_later_timesurface_blur.data),
          reinterpret_cast<float *>(cv_later_timesurface_blur.data) + height_ * width_);
      std::vector<float> line_iwe_early(
          reinterpret_cast<float *>(iwe_early.data),
          reinterpret_cast<float *>(iwe_early.data) + height_ * width_);
      std::vector<float> line_iwe_later(
          reinterpret_cast<float *>(iwe_later.data),
          reinterpret_cast<float *>(iwe_later.data) + height_ * width_);

      ceres::Grid2D<float, 1> grid_early(line_grid_early.data(), 0, height_, 0, width_);
      ceres::Grid2D<float, 1> grid_later(line_grid_later.data(), 0, height_, 0, width_);
      ceres::Grid2D<float, 1> iwe_grid_early(line_iwe_early.data(), 0, height_, 0, width_);
      ceres::Grid2D<float, 1> iwe_grid_later(line_iwe_later.data(), 0, height_, 0, width_);
      auto interpolator_early_ptr = std::make_unique<ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>>(grid_early);
      auto interpolator_later_ptr = std::make_unique<ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>>(grid_later);
      auto iwe_interpolator_early_ptr =
          std::make_unique<ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>>(iwe_grid_early);
      auto iwe_interpolator_later_ptr =
          std::make_unique<ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>>(iwe_grid_later);

      // 采样
      std::vector<int> vec_sampled_idx;
      int samples_count = std::min(params_.sample_count, eventBundle_.size);
      getSampledIndices(vec_sampled_idx, samples_count, 0.0, 1.0);

      // 构建 Ceres 问题
      ceres::Problem problem;

      for (int sample_idx : vec_sampled_idx)
      {
        double early_time = eventBundle_.time_delta[sample_idx];
        double later_time = eventBundle_.time_delta[sample_idx] - eventBundle_.time_delta.back();

        ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<PEMEPlus, 1, 3>(
            new PEMEPlus(event_undis_Bundle_.coord_3d[sample_idx],
                         early_time, later_time, K_,
                         interpolator_early_ptr.get(),
                         interpolator_later_ptr.get(),
                         iwe_interpolator_early_ptr.get(),
                         iwe_interpolator_later_ptr.get(),
                         last_est_var_));

        problem.AddResidualBlock(cost_function, nullptr, angleAxis);
      }

      // Ceres 求解器配置
      ceres::Solver::Options options;
      options.minimizer_progress_to_stdout = false;
      options.num_threads = params_.ceres_iter_thread;
      options.linear_solver_type = ceres::SPARSE_SCHUR;
      options.max_num_iterations = params_.ceres_iter_num;

      ceres::Solver::Summary summary;

      // 第一次迭代时评估初始值
      if (iter == 1)
      {
        double cost_old = 0, cost_zero = 0;
        std::vector<double> residual_vec;

        // 测试旧速度
        angleAxis[0] = est_angleAxis_(0);
        angleAxis[1] = est_angleAxis_(1);
        angleAxis[2] = est_angleAxis_(2);
        problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost_old, &residual_vec, nullptr, nullptr);

        // 测试零速度
        angleAxis[0] = angleAxis[1] = angleAxis[2] = 0;
        problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost_zero, &residual_vec, nullptr, nullptr);

        if (cost_old < cost_zero)
        {
          angleAxis[0] = est_angleAxis_(0);
          angleAxis[1] = est_angleAxis_(1);
          angleAxis[2] = est_angleAxis_(2);
        }
      }

      // 求解
      ceres::Solve(options, &problem, &summary);

      // 计算当前损失
      double current_loss = 0.0;
      std::vector<double> residual_vec;
      problem.Evaluate(ceres::Problem::EvaluateOptions(), &current_loss, &residual_vec, nullptr, nullptr);

      if (current_loss < best_loss)
      {
        best_loss = current_loss;
        best_angleAxis_ = Vec3(angleAxis[0], angleAxis[1], angleAxis[2]);
      }

      // 可选：输出每次迭代损失
      std::cout << "[PEME] Iter " << iter << " loss: " << current_loss << std::endl;
    }

    // 更新估计值
    est_angleAxis_ = Vec3(angleAxis[0], angleAxis[1], angleAxis[2]);
    last_est_var_ = est_angleAxis_;
    return best_loss;
  }

  RegistrationOutput PEMEPlusEstimator::run(const std::vector<Vec2> &xy,
                                            const std::vector<double> &timestamps,
                                            const std::vector<int> &polarity,
                                            const std::vector<double> &depth)
  {
    RegistrationOutput result;
    result.timespan = {timestamps.front(), timestamps.back()};

    if (xy.empty())
    {
      result.twist.setZero();
      result.stats.loss = 0;
      result.stats.num_iterations = 0;
      return result;
    }

    // 初始化事件数据
    eventBundle_.size = xy.size();
    eventBundle_.coord = xy;
    eventBundle_.polarity = polarity;
    eventBundle_.first_tstamp = timestamps.front();
    eventBundle_.last_tstamp = timestamps.back();

    // 计算相对时间
    eventBundle_.time_delta.resize(eventBundle_.size);
    for (int i = 0; i < eventBundle_.size; ++i)
    {
      eventBundle_.time_delta[i] = timestamps[i] - timestamps.front();
    }

    // 投影到 3D
    event_undis_Bundle_.size = eventBundle_.size;
    projectEventsTo3D(xy, event_undis_Bundle_.coord_3d, depth);

    std::cout << '[' << count_ << "] " << "PEME: Processing "
              << eventBundle_.size << " events, duration: "
              << (eventBundle_.last_tstamp - eventBundle_.first_tstamp) << "s" << std::endl;

    // 运行优化
    result.stats.loss = optimizeMotion();

    // 填充结果
    result.twist = est_angleAxis_;
    result.stats.num_iterations = count_;

    std::cout << '[' << count_ << "] " << "PEME: Estimated angular velocity: "
              << est_angleAxis_.transpose()
              << ", best loss: " << result.stats.loss << std::endl;
    count_++;
    return result;
  }

} // namespace spatiotemporal_vo