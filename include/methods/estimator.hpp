#pragma once

#include "spatiotemporal_vo/common.hpp"
#include "spatiotemporal_vo/config.hpp"
#include <vector>
#include <utility>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace spatiotemporal_vo
{

  /**
   * @brief
   */
  template <int N>
  class MotionEstimator
  {
  public:
    /**
     * @brief 构造函数
     * @param fx, fy, cx, cy 相机内参
     * @param width, height 图像尺寸
     */
    MotionEstimator(double fx, double fy, double cx, double cy,
                    int width, int height) : width_(width), height_(height), count_(1), fx_(fx), fy_(fy), cx_(cx), cy_(cy)
    {
      // 初始化梯度和海森矩阵图像容器
#ifdef SEPARATE_POLARITY
      warped_image_pos_.resize(height_ + 2 * padding_, width_ + 2 * padding_);
      warped_image_neg_.resize(height_ + 2 * padding_, width_ + 2 * padding_);
      pixel_grad_pos_.resize(height_ + 2 * padding_, width_ + 2 * padding_);
      pixel_grad_neg_.resize(height_ + 2 * padding_, width_ + 2 * padding_);
      pixel_hess_pos_.resize(height_ + 2 * padding_, width_ + 2 * padding_);
      pixel_hess_neg_.resize(height_ + 2 * padding_, width_ + 2 * padding_);
      iwe_grad_pos_.resize(N);
      iwe_grad_neg_.resize(N);
#ifdef ANALYTIC_HESSIAN
      iwe_hess_pos_.resize(N * N);
      iwe_hess_neg_.resize(N * N);
#endif
      for (int i = 0; i < N; ++i)
      {
        iwe_grad_pos_[i].resize(height_ + 2 * padding_, width_ + 2 * padding_);
        iwe_grad_neg_[i].resize(height_ + 2 * padding_, width_ + 2 * padding_);
      }
#ifdef ANALYTIC_HESSIAN
      for (int i = 0; i < N * N; ++i)
      {
        iwe_hess_pos_[i].resize(height_ + 2 * padding_, width_ + 2 * padding_);
        iwe_hess_neg_[i].resize(height_ + 2 * padding_, width_ + 2 * padding_);
      }
#endif
#else
      warped_image_.resize(height_ + 2 * padding_, width_ + 2 * padding_);
      pixel_grad_.resize(height_ + 2 * padding_, width_ + 2 * padding_);
      pixel_hess_.resize(height_ + 2 * padding_, width_ + 2 * padding_);
      iwe_grad_.resize(N);
#ifdef ANALYTIC_HESSIAN
      iwe_hess_.resize(N * N);
#endif
      for (int i = 0; i < N; ++i)
      {
        iwe_grad_[i].resize(height_ + 2 * padding_, width_ + 2 * padding_);
      }
#ifdef ANALYTIC_HESSIAN
      for (int i = 0; i < N * N; ++i)
      {
        iwe_hess_[i].resize(height_ + 2 * padding_, width_ + 2 * padding_);
      }
#endif
#endif
    };

    /**
     * @brief 虚析构函数
     */
    virtual ~MotionEstimator() = default;

    /**
     * @brief 运行 PEME 估计
     * @param packet 事件包
     * @param config 配置对象
     * @return 估计结果
     */
    virtual RegistrationOutput<N> run(const Config &config, const EventPacket &packet)
    {
      points_ = &packet.coord_3d;
      delta_time_ = &packet.t;
      polarity_ = &packet.polarity;
      n_points_ = points_->size();

      RegistrationOutput<N> result;

      if (n_points_ < 10)
      {
        result.params.setZero();
        result.stats.loss = 0;
        result.stats.time_taken = 0.0;
        result.stats.num_iterations = 0;
        return result;
      }

      std::cout << '[' << count_ << "] " << "<<" << getMethodName() << ">>" << ": Processing "
                << n_points_ << " events, duration: "
                << delta_time_->back() - delta_time_->front() << "s" << std::endl;
      result.stats.time_taken = omp_get_wtime();

      // 运行优化
      MainOptimize(config);

      // 更新结果
#ifdef DEBUG
      std::cout << "[DEBUG] Final estimated parammeters: " << est_params_.transpose() << std::endl;
#endif
      // 运行优化
      result.stats.loss = final_loss_;
      result.stats.time_taken = omp_get_wtime() - result.stats.time_taken;
      std::cout << '[' << count_ << "] " << "<<" << getMethodName() << ">>" << ": Optimization took " << result.stats.time_taken << " seconds." << std::endl;

      // 填充结果
      result.params = est_params_;
      result.stats.num_iterations = count_;

      std::cout << '[' << count_ << "] " << "<<" << getMethodName() << ">>" << ": Estimated parammeters: "
                << est_params_.transpose()
                << ", best loss: " << result.stats.loss << std::endl;
      count_++;
      return result;
    };
    /**
     * @brief 获取当前估计的角速度
     */
    Eigen::Matrix<double, N, 1> getEstimatedParameters() const { return est_params_; }

    /**
     * @brief 获取最终损失值
     */
    double getFinalLoss() const { return final_loss_; }

  protected:
    // 相机内参矩阵
    double fx_, fy_, cx_, cy_;
    int width_, height_, padding_ = 100;
    const std::vector<double> *delta_time_;
    const std::vector<int> *polarity_;
#ifdef ANALYTIC_HESSIAN
#ifdef SEPARATE_POLARITY
    std::vector<EigenMat> iwe_hess_pos_, iwe_hess_neg_, iwe_grad_pos_, iwe_grad_neg_;
#else
    std::vector<EigenMat> iwe_hess_, iwe_grad_;
#endif
#else
#ifdef SEPARATE_POLARITY
    std::vector<EigenMat> iwe_grad_pos_, iwe_grad_neg_;
#else
    std::vector<EigenMat> iwe_grad_;
#endif
#endif
    int n_points_;
    double final_loss_ = 0.0;
#ifdef SEPARATE_POLARITY
    EigenMat warped_image_pos_, warped_image_neg_;
    EigenMat pixel_grad_pos_, pixel_grad_neg_;
    EigenMat pixel_hess_pos_, pixel_hess_neg_;
#else
    EigenMat warped_image_, pixel_grad_, pixel_hess_;
#endif
    Eigen::Matrix<double, N, 1> est_params_; // 当前估计的角速度
    // 估计状态
    int count_;

    const std::vector<Vec3> *points_;
    virtual std::string getMethodName() const { return "MotionEstimator"; }
    /**
     * @brief 计算CMax函数对像素的一阶梯度
     * @param iwe 输入的翘曲事件图像
     * @param loss 输出的损失值
     * @param grad 输出的梯度图像
     * @param hess 输出的海森图像
     */
    virtual void iwe_gradient(const EigenMat &iwe, double &loss, EigenMat &grad, EigenMat &hess) const = 0;
    /**
     * @brief 计算运动参数对翘曲事件点的梯度
     * @param parameters 运动参数（角轴表示法）
     * @param xs, ys 翘曲后的事件点（2D坐标），注意，这里的坐标是减去u0 v0后的坐标
     * @param grad_warp 输出的梯度矩阵，大小为2x3
     */
    virtual void warp_gradient(const Eigen::Matrix<double, N, 1> &parameters, const Vec3 &point, double dt, Eigen::Matrix<double, 2, N> &grad_warp, Eigen::Matrix<double, N, N> &hess_warp) const = 0;

    /**
     * @brief 核心优化函数
     * @param config 配置对象
     * @return 损失函数值
     */
    virtual void MainOptimize(const Config &config)
    {
      // 使用牛顿法进行优化
      const int max_iterations = config.solver.max_iters;
      const double tolerance = config.solver.convergence_thresh;
      const double decay = config.solver.decay;
      double lr = config.solver.lr;
      Eigen::Matrix<double, N, N> hessian;
      Eigen::Matrix<double, N, 1> gradient;
      double best_loss = std::numeric_limits<double>::max();
      double loss, lambda = 1.0, prev_loss = best_loss;
      Eigen::Matrix<double, N, 1> best_params = est_params_, step;
      for (int iter = 0; iter < max_iterations; ++iter)
      {
        compute_gradient(est_params_, loss, gradient, hessian);
#ifdef DEBUG
        std::cout << "[DEBUG] loss = " << loss << std::endl;
        std::cout << "[DEBUG] gradient = " << gradient.transpose() << std::endl;
        std::cout << "[DEBUG] hessian :" << hessian << std::endl;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> eigensolver(hessian);
        std::cout << "[DEBUG] Hessian eigenvalues: " << eigensolver.eigenvalues().transpose() << std::endl;
#endif
        // 添加阻尼项以确保Hessian矩阵是正定的
        // hessian.diagonal().array() += (lambda * hessian.diagonal().mean());

        // 求解线性系统
        step = hessian.ldlt().solve(-gradient) * lr;
        est_params_ += step;
        // 检查收敛性
        if (step.norm() < tolerance || std::abs(prev_loss - loss) < tolerance)
        {
          break;
        }
        // 更新参数
        if (best_loss > loss)
        {
          best_loss = loss;
          best_params = est_params_;
        }
        lr *= decay; // 减小学习率
        prev_loss = loss;
#ifdef DEBUG
        std::cout << "[DEBUG] Iteration " << iter << ": Loss = " << loss << ", Step Norm = " << step.norm() << std::endl;
        std::cout << "[DEBUG] Estimated Parameters: " << est_params_.transpose() << std::endl;
#endif
      }
      // 计算最终损失
      est_params_ = best_params;
      final_loss_ = best_loss;
    };
    /**
     * @brief 计算损失函数对运动参数的梯度和海森矩阵
     * @param parameters 运动参数（角轴表示法）
     * @param loss 输出的损失函数值
     * @param gradient 输出的梯度
     * @param hessian 输出的海森矩阵
     */
    virtual void compute_gradient(const Eigen::Matrix<double, N, 1> &parameters, double &loss, Eigen::Matrix<double, N, 1> &gradient, Eigen::Matrix<double, N, N> &hessian) = 0;

    /**
     * @brief 将向量值分配到对应的矩阵位置
     * @param vec 输入向量
     * @param mat_vec 矩阵向量
     */
    inline void assign_vec(const Eigen::Matrix<double, N, 1> &vec, std::vector<EigenMat> &mat_vec, int x, int y) const
    {
      for (int i = 0; i < N; ++i)
      {
        mat_vec[i](y, x) += vec(i);
      }
    }
    /**
     * @brief 将矩阵值分配到对应的矩阵位置
     * @param mat 输入矩阵
     * @param mat_vec 矩阵向量
     */
    inline void assign_mat(const Eigen::Matrix<double, N, N> &mat, std::vector<EigenMat> &mat_vec, int x, int y) const
    {
      for (int i = 0; i < N; ++i)
      {
        for (int j = 0; j < N; ++j)
        {
          mat_vec[i * 3 + j](y, x) += mat(i, j);
        }
      }
    };
  };
} // namespace spatiotemporal_vo