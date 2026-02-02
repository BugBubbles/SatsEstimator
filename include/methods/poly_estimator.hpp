#pragma once

#include "spatiotemporal_vo/common.hpp"
#include "spatiotemporal_vo/config.hpp"
#include <vector>
#include <utility>
#include <Eigen/Core>

namespace spatiotemporal_vo
{

  /**
   * @brief PEME (Patch-based Event Motion Estimation) 估计器
   * 基于时间表面的对比度最大化方法
   */
  class PolyEstimator
  {
  public:
    // 配置参数结构体
    struct PEMEParams
    {
      double ts_start = 0.2;     // 时间表面起始比例
      double ts_end = 0.8;       // 时间表面结束比例
      int sample_count = 1000;   // 采样点数量
      int iter_num = 50;         // 外层迭代次数
      int ceres_iter_num = 10;   // Ceres 内部迭代次数
      int ceres_iter_thread = 4; // Ceres 线程数
      int denoise_num = 3;
      float tau = 0.05f;               // 去噪时间常数
      int gaussian_size = 5;             // 高斯模糊核大小
      float gaussian_sigma = 1.0f;       // 高斯模糊 sigma
      float default_value_factor = 1.5f; // 默认值因子
    };

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
     * @param xy 事件坐标 (2xN)
     * @param timestamps 事件时间戳
     * @param polarity 事件极性
     * @return 估计结果
     */
    RegistrationOutput run(const std::vector<Vec2> &xy,
                           const std::vector<double> &timestamps,
                           const std::vector<int> &polarity,
                           const std::vector<double> &depth);

    /**
     * @brief 设置参数
     */
    void setParams(const PEMEParams &params) { params_ = params; }

    /**
     * @brief 获取当前估计的角速度
     */
    Vec3 getEstimatedAngularVelocity() const { return est_angleAxis_; }

  private:
    // 相机内参矩阵
    Eigen::Matrix3d K_;
    int width_, height_, padding_ = 100;

    // 参数
    PEMEParams params_;

    // 估计状态
    Vec3 est_angleAxis_; // 当前估计的角速度
    Vec3 last_est_var_;  // 上一次估计值
    Vec3 best_angleAxis_;
    int count_;

    // 事件数据缓存
    struct EventBundle
    {
      std::vector<Vec2> coord;        // 2D 坐标
      std::vector<Vec3> coord_3d;     // 3D 归一化坐标
      std::vector<double> time_delta; // 相对时间
      std::vector<int> polarity;      // 极性
      std::vector<bool> isInner;      // 是否在图像内
      int size;
      double first_tstamp;
      double last_tstamp;
    };

    EventBundle eventBundle_;
    EventBundle event_undis_Bundle_;
    EventBundle event_warpped_Bundle_;

    /**
     * @brief 将事件投影到归一化平面
     */
    void projectEventsTo3D(const std::vector<Vec2> &xy,
                           std::vector<Vec3> &coord_3d,
                           const std::vector<double> &depth);

    /**
     * @brief 获取采样索引
     */
    void getSampledIndices(const cv::Mat &iwe,
                           std::vector<int> &indices, int count,
                           double start_ratio, double end_ratio);

    /**
     * @brief 获取 warp 后的事件图像
     */
    cv::Mat getWarpedEventImage(const Vec3 &angularVelocity,
                                EventBundle &warped_bundle,
                                bool use_later_time = false);

    /**
     * @brief 核心优化函数
     * @return 损失函数值
     */
    double optimizeMotion();
  };
} // namespace spatiotemporal_vo