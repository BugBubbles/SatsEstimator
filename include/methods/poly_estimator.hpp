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
  class PolyEstimator
  {
  public:

    /**
     * @brief 构造函数
     * @param fx, fy, cx, cy 相机内参
     * @param width, height 图像尺寸
     * @param config 配置对象
     */
    PolyEstimator(double fx, double fy, double cx, double cy,
                      int width, int height, const Config &config);

    ~PolyEstimator() = default;

    /**
     * @brief 运行 PEME 估计
     * @param packet 事件包
     * @param config 配置对象
     * @return 估计结果
     */
    RegistrationOutput run(const EventPacket &packet, const Config &config);


    /**
     * @brief 获取当前估计的角速度
     */
    Vec3 getEstimatedAngularVelocity() const { return est_angleAxis_; }

  private:
    // 相机内参矩阵
    Eigen::Matrix3d K_;
    int width_, height_, padding_ = 100;

    // 估计状态
    Vec3 est_angleAxis_; // 当前估计的角速度
    Vec3 last_est_var_;  // 上一次估计值
    Vec3 best_angleAxis_;
    int count_;

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
    double optimizeMotion(const Config &config);
  };
} // namespace spatiotemporal_vo