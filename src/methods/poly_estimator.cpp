#include "spatiotemporal_vo/common.hpp"
#include "methods/poly_estimator.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <iostream>

namespace spatiotemporal_vo
{

  class PolyOptimizer
  {
  private:
    // 你的数据成员（和原CMaxFunction一致）
    std::vector<Vec3> points_;
    int width_, height_;
    Eigen::Matrix3d intrinsic_;
    Eigen::Matrix<double, Eigen::Dynamic, 13, Eigen::RowMajor> iwe_data_;
    EigenMat warped_image_, pixel_grad_, pixel_hess_;
    std::vector<double> delta_time_;
    std::vector<int> polarity_;
    int n_points_;
    int padding_ = 100;
    double final_loss_ = 0.0;
    Vec3 est_angleAxis_;

  public:
    void getEstimatedParams(Vec3 &est_angleAxis) const { est_angleAxis = est_angleAxis_; }
    double getFinalLoss() const { return final_loss_; }
    PolyOptimizer(const std::vector<Vec3> &points,
                  const std::vector<double> &delta_time,
                  const std::vector<int> &polarity,
                  int width, int height,
                  const Eigen::Matrix3d &K,
                  const Vec3 &initial_angleAxis)
        : points_(points), delta_time_(delta_time), polarity_(polarity),
          width_(width), height_(height), intrinsic_(K), est_angleAxis_(initial_angleAxis)
    {
      n_points_ = points_.size();
      warped_image_.resize(height_ + 2 * padding_, width_ + 2 * padding_);
      pixel_grad_.resize(height_ + 2 * padding_, width_ + 2 * padding_);
      pixel_hess_.resize(height_ + 2 * padding_, width_ + 2 * padding_);
      iwe_data_.resize((height_ + 2 * padding_) * (width_ + 2 * padding_), 13); // 13 channel for 1 loss + 3 grad + 9 hess
    }
    /**
     * @brief 计算CMax函数对像素的一阶梯度
     */
    void iwe_gradient(const EigenMat &iwe, double &loss, EigenMat &grad, EigenMat &hess) const
    {
      // CMax函数
      loss = -(iwe.array() - iwe.mean()).square().mean();
      grad = -2 * (iwe.array() - iwe.mean()).matrix();
      hess = -2 * Eigen::MatrixXd::Ones(iwe.rows(), iwe.cols());
      // Ours
      // loss = -((0.9416 * iwe.array() + (iwe.array() + 0.1).log()).mean());
      // grad = 0.9416 + 1 / (iwe.array() + 0.1);
      // hess = -1 / (iwe.array() + 0.1).square();
    }
    /**
     * @brief 计算运动参数对翘曲事件点的梯度
     * @param parameters 运动参数（角轴表示法）
     * @param xs, ys 翘曲后的事件点（2D坐标），注意，这里的坐标是减去u0 v0后的坐标
     * @param grad_warp 输出的梯度矩阵，大小为2x3
     */
    void warp_gradient(const Vec3 &parameters, const Vec3 &point, double dt, Eigen::Matrix<double, 2, 3> &grad_warp, Eigen::Matrix3d &hess_warp) const
    {
      double X_Z = point(0) / point(2);
      double Y_Z = point(1) / point(2);
      grad_warp << intrinsic_(0, 0) * X_Z * Y_Z, -intrinsic_(0, 0) * (1 + X_Z * X_Z), intrinsic_(0, 0) * Y_Z,
          intrinsic_(1, 1) * (1 + Y_Z * Y_Z), -intrinsic_(1, 1) * X_Z * Y_Z, -intrinsic_(1, 1) * X_Z;
      grad_warp *= dt;
      hess_warp = grad_warp.row(0).transpose() * grad_warp.row(1) + grad_warp.row(1).transpose() * grad_warp.row(0);
    }
    void compute_gradient(const Vec3 &parameters, double &loss, Vec3 &gradient, Eigen::Matrix3d &hessian, bool use_fast)
    {
      const int pad_width = width_ + 2 * padding_;
      const int pad_height = height_ + 2 * padding_;
      const int channels = 13;
      // 使用 vector 保证内存连续，初始化为 0
      iwe_data_.setZero();
      Eigen::Matrix<double, 3, 1> delta_points_T, delta_second_points_T;
      Eigen::Matrix<double, 3, 1> points_T;

      double warped_x, warped_y, event_x, event_y;
      Eigen::Matrix<double, 2, 3> grad_warp;
      Eigen::Matrix3d hess_warp;

      // 提取内参，避免在循环中重复访问成员变量
      const double fx = intrinsic_(0, 0);
      const double cx = intrinsic_(0, 2);
      const double fy = intrinsic_(1, 1);
      const double cy = intrinsic_(1, 2);
      for (int i = 0; i < n_points_; ++i)
      {
        auto point_ = points_[i];
        // Taylor 二阶展开: points * skew_matrix * skew_matrix
        delta_points_T(0) = -parameters[2] * point_(1) + parameters[1] * point_(2);
        delta_points_T(1) = parameters[2] * point_(0) - parameters[0] * point_(2);
        delta_points_T(2) = -parameters[1] * point_(0) + parameters[0] * point_(1);

        // delta_second_points_T(0) = -parameters[2] * delta_points_T(1) + parameters[1] * delta_points_T(2);
        // delta_second_points_T(1) = parameters[2] * delta_points_T(0) - parameters[0] * delta_points_T(2);
        // delta_second_points_T(2) = -parameters[1] * delta_points_T(0) + parameters[0] * delta_points_T(1);
        // 计算 warp 后的 3D 点
        double dt = delta_time_[i];
        // double dt_sq = 0.5 * dt * dt;
        points_T(0) = point_(0) + delta_points_T(0) * dt;
        points_T(1) = point_(1) + delta_points_T(1) * dt;
        points_T(2) = point_(2) + delta_points_T(2) * dt;

        // 投影到 2D
        warped_x = points_T(0) / points_T(2) * fx + cx + padding_;
        warped_y = points_T(1) / points_T(2) * fy + cy + padding_;
        // 使用双线性插值将当前事件翘曲至图像
        int x = static_cast<int>(floor(warped_x));
        int y = static_cast<int>(floor(warped_y));
        double dx = warped_x - x;
        double dy = warped_y - y;

        if (x >= 0 && x < pad_width - 1 && y >= 0 && y < pad_height - 1)
        {
          double dx = warped_x - x;
          double dy = warped_y - y;
          double dx_inv = 1.0 - dx;
          double dy_inv = 1.0 - dy;
          // 计算梯度
          warp_gradient(parameters, point_, dt, grad_warp, hess_warp);
          // 注意：如果 grad_warp 是 RowMajor，data布局不同。
          // 假设 grad_warp 是默认的 Matrix<double, 2, 3> (ColMajor):
          // data: [r0c0, r1c0, r0c1, r1c1, r0c2, r1c2]
          // 手动获取值比用 Eigen 接口更快
          double g00 = grad_warp(0, 0), g01 = grad_warp(0, 1), g02 = grad_warp(0, 2);
          double g10 = grad_warp(1, 0), g11 = grad_warp(1, 1), g12 = grad_warp(1, 2);

          // 获取 Hessian 数据指针
          const double *h_ptr = hess_warp.data(); // 9个数据
                                                  // Lambda: 更新单个像素的 13 个通道
          auto update_pixel = [&](int offset, double w, double c1, double c2, double h_sign)
          {
            // 直接获取该像素在 Eigen 矩阵中的行指针
            // iwe_data_ 是 RowMajor，所以 row(offset) 的数据是连续的
            double *ptr = iwe_data_.row(offset).data();

            // [0]: Image
            ptr[0] += w;
            // [1-3]: Gradient
            ptr[1] += c1 * g00 + c2 * g10;
            ptr[2] += c1 * g01 + c2 * g11;
            ptr[3] += c1 * g02 + c2 * g12;
            // [4-12]: Hessian
            if (h_sign > 0)
            {
              for (int k = 0; k < 9; ++k)
                ptr[4 + k] += h_ptr[k];
            }
            else
            {
              for (int k = 0; k < 9; ++k)
                ptr[4 + k] -= h_ptr[k];
            }
          };
          int pixel_idx = y * pad_width + x;
          update_pixel(pixel_idx, dx_inv * dy_inv, (dy - 1.0), (dx - 1.0), 1.0);   // TL
          update_pixel(pixel_idx + 1, dx * dy_inv, (dy_inv), (-dx), -1.0);         // TR
          update_pixel(pixel_idx + pad_width, dx_inv * dy, (-dy), (dx_inv), -1.0); // BL
          update_pixel(pixel_idx + pad_width + 1, dx * dy, (dy), (dx), 1.0);       // BR
        }
      }
      cv::Mat cv_view(pad_height, pad_width, CV_MAKETYPE(CV_64F, 13), iwe_data_.data());

      // OpenCV 会并行地对这 13 个通道分别进行高斯模糊
      cv::GaussianBlur(cv_view, cv_view, cv::Size(5, 5), 1.0);

      // 计算最终的梯度与海森矩阵
      iwe_gradient(iwe_data_.col(0), loss, pixel_grad_, pixel_hess_);
      // 计算总梯度，各个像素梯度的加权和
      // gradient 等于将 pixel_grad 与 iwe_grad_ 第一个通道按像素位置对应相乘后求和
      for (int i = 0; i < 3; ++i)
      {
        gradient(i) = (pixel_grad_.array() * iwe_data_.col(1 + i).array()).sum();
      }
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; ++j)
        {
          hessian(i, j) = (pixel_grad_.array() * iwe_data_.col(4 + i * 3 + j).array() + pixel_hess_.array() * iwe_data_.col(1 + i).array() * iwe_data_.col(1 + j).array()).sum();
        }
      }
    }
    void MainOptimize()
    {
      // 使用牛顿法进行优化
      const int max_iterations = 40;
      const double tolerance = 1e-5;
      Eigen::Matrix3d hessian;
      Vec3 gradient;
      Eigen::Matrix3d D;
      double best_loss = std::numeric_limits<double>::max();
      double loss = 0.0, prev_loss = best_loss;
      Vec3 best_params = est_angleAxis_;
      for (int iter = 0; iter < max_iterations; ++iter)
      {
        // compute_loss(est_angleAxis_, loss);
        compute_gradient(est_angleAxis_, loss, gradient, hessian, true);
        // compute_gradient(est_angleAxis_, loss, gradient, hessian, true);

        // 添加阻尼项以确保Hessian矩阵是正定的
        Eigen::Matrix3d damped_hessian = hessian;

        // 求解线性系统
        Eigen::Vector3d step = damped_hessian.ldlt().solve(-gradient);
        est_angleAxis_ += step;
        // 检查收敛性
        if (step.norm() < tolerance || std::abs(prev_loss - loss) < tolerance)
        {
          break;
        }
        // 更新参数
        if (best_loss > loss)
        {
          best_loss = loss;
          best_params = est_angleAxis_;
        }
        prev_loss = loss;
#ifdef DEBUG
        std::cout << "Iteration " << iter << ": Loss = " << loss << ", Step Norm = " << step.norm() << std::endl;
        std::cout << "Estimated Parameters: " << est_angleAxis_.transpose() << std::endl;
#endif
      }
      // 计算最终损失
      est_angleAxis_ = best_params;
      final_loss_ = best_loss;
      // compute_loss(est_angleAxis_, final_loss_);
    }
  };
  PolyEstimator::PolyEstimator(double fx, double fy, double cx, double cy,
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
    params_.iter_num = 1;
    params_.ceres_iter_num = 250;
    params_.ceres_iter_thread = 8;
    params_.gaussian_size = 5;
    params_.gaussian_sigma = 1.0f;
    params_.default_value_factor = 1.0f;
  }

  void PolyEstimator::projectEventsTo3D(const std::vector<Vec2> &xy,
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
      // coord_3d[i].normalize();
    }
  }

  RegistrationOutput PolyEstimator::run(const std::vector<Vec2> &xy,
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

    std::cout << '[' << count_ << "] " << "Poly: Processing "
              << eventBundle_.size << " events, duration: "
              << (eventBundle_.last_tstamp - eventBundle_.first_tstamp) << "s" << std::endl;
    PolyOptimizer optimizer(
        event_undis_Bundle_.coord_3d,
        eventBundle_.time_delta,
        eventBundle_.polarity,
        width_, height_,
        K_, est_angleAxis_);

    // 运行优化
    optimizer.MainOptimize();

    // 更新结果
    optimizer.getEstimatedParams(est_angleAxis_);
#ifdef DEBUG
    std::cout << "Final estimated angular velocity: " << est_angleAxis_.transpose() << std::endl;
#endif
    // 运行优化
    result.stats.loss = optimizer.getFinalLoss();

    // 填充结果
    result.twist = est_angleAxis_;
    result.stats.num_iterations = count_;

    std::cout << '[' << count_ << "] " << "CMax: Estimated angular velocity: "
              << est_angleAxis_.transpose()
              << ", best loss: " << result.stats.loss << std::endl;
    count_++;
    return result;
  }

} // namespace spatiotemporal_vo