#pragma once

#include "spatiotemporal_vo/common.hpp"
#include "spatiotemporal_vo/config.hpp"
#include <vector>
#include <utility>
#include <Eigen/Core>

namespace spatiotemporal_vo
{

  /**
   * @brief
   */
  class MotionEstimator
  {
  public:

    /**
     * @brief 构造函数
     * @param fx, fy, cx, cy 相机内参
     * @param width, height 图像尺寸
     */
    MotionEstimator(double fx, double fy, double cx, double cy,
                      int width, int height);

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
    virtual RegistrationOutput run(const Config &config, const EventPacket &packet);

    /**
     * @brief 获取当前估计的角速度
     */
    Vec3 getEstimatedAngular() const { return est_angleAxis_; }

    /**
     * @brief 获取当前估计的平移速度
     */
    Vec3 getEstimatedTranslation() const { return est_transAxis_; }

    /**
     * @brief 获取最终损失值
     */
    double getFinalLoss() const { return final_loss_; }
    
  protected:
      // 相机内参矩阵
    double fx_, fy_, cx_, cy_;
    int width_, height_, padding_ = 100;
    const std::vector<double>* delta_time_;
    const std::vector<int>* polarity_;
    std::vector<EigenMat> iwe_hess_, iwe_grad_;
    int n_points_;
    double final_loss_ = 0.0;
    EigenMat warped_image_, pixel_grad_, pixel_hess_;
    Vec3 est_angleAxis_, est_transAxis_; // 当前估计的角速度
    // 估计状态
    int count_;

    const std::vector<Vec3>* points_;
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
    virtual void warp_gradient(const Vec3 &parameters, const Vec3 &point, double dt, Eigen::Matrix<double, 2, 3> &grad_warp, Mat3 &hess_warp) const = 0;

    /**
     * @brief 核心优化函数
     * @param config 配置对象
     * @return 损失函数值
     */
    virtual void MainOptimize(const Config &config);

    /**
     * @brief 计算损失函数对运动参数的梯度和海森矩阵
     * @param parameters 运动参数（角轴表示法）
     * @param loss 输出的损失函数值
     * @param gradient 输出的梯度
     * @param hessian 输出的海森矩阵
     */
    virtual void compute_gradient(const Vec3 &parameters, double &loss, Vec3 &gradient, Mat3 &hessian) = 0;

    /**
     * @brief 将向量值分配到对应的矩阵位置
     * @param vec 输入向量
     * @param mat_vec 矩阵向量
     */
    inline void assign_vec(const Vec3 &vec, std::vector<EigenMat> &mat_vec, int x, int y) const
    {
      for (int i = 0; i < 3; ++i)
      {
        mat_vec[i](y, x) += vec(i);
      }
    }
    /**
     * @brief 将矩阵值分配到对应的矩阵位置
     * @param mat 输入矩阵
     * @param mat_vec 矩阵向量
     */
    inline void assign_mat(const Mat3 &mat, std::vector<EigenMat> &mat_vec, int x, int y) const
    {
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; ++j)
        {
          mat_vec[i * 3 + j](y, x) += mat(i, j);
        }
      }
    };
  };
} // namespace spatiotemporal_vo