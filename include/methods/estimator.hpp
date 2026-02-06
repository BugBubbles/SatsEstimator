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
     * @param packet 事件包
     */
    MotionEstimator(double fx, double fy, double cx, double cy,
                      int width, int height, const EventPacket &packet);

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
    RegistrationOutput run(const Config &config);


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
    EigenMat warped_image_, pixel_grad_, pixel_hess_;
    Vec3 est_angleAxis_, est_transAxis_; // 当前估计的角速度
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
    virtual void warp_gradient(const Vec3 &parameters, const Vec3 &point, double dt, Eigen::Matrix<double, 2, 3> &grad_warp, Eigen::Matrix3d &hess_warp) const = 0;

  private:
    int width_, height_, padding_ = 100;
    // 估计状态
    int count_;
    std::vector<Vec3> points_;
    std::vector<EigenMat> iwe_grad_;
    std::vector<EigenMat> iwe_hess_;
    std::vector<double> delta_time_;
    std::vector<int> polarity_;
    int n_points_;
    double final_loss_ = 0.0;
    /**
     * @brief 获取采样索引
     */
    void getSampledIndices(const cv::Mat &iwe,
                           std::vector<int> &indices, int count,
                           double start_ratio, double end_ratio);

    /**
     * @brief 获取 warp 后的事件图像
     * @param angularVelocity 角速度
     * @param warped_events 输出的翘曲事件包
     * @param use_later_time 是否使用事件的后时刻进行翘曲
     * @return 翘曲后的事件图像
     */
    cv::Mat getWarpedEventImage(const Vec3 &angularVelocity,
                                EventPacket &warped_events,
                                bool use_later_time = false);

    /**
     * @brief 核心优化函数
     * @param config 配置对象
     * @return 损失函数值
     */
    void MainOptimize(const Config &config);


    /**
     * @brief 将向量值分配到对应的矩阵位置
     * @param vec 输入向量
     * @param mat_vec 矩阵向量
     */
    void assign_vec(const Vec3 &vec, std::vector<EigenMat> &mat_vec, int x, int y) const;

    /**
     * @brief 将矩阵值分配到对应的矩阵位置
     * @param mat 输入矩阵
     * @param mat_vec 矩阵向量
     */
    void assign_mat(const Eigen::Matrix3d &mat, std::vector<EigenMat> &mat_vec, int x, int y) const;

    /**
     * @brief 计算损失函数对运动参数的梯度和海森矩阵
     * @param parameters 运动参数（角轴表示法）
     * @param loss 输出的损失函数值
     * @param gradient 输出的梯度
     * @param hessian 输出的海森矩阵
     */
    void compute_gradient(const Vec3 &parameters, double &loss, Vec3 &gradient, Eigen::Matrix3d &hessian);
  };
} // namespace spatiotemporal_vo