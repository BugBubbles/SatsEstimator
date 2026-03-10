#pragma once

#include "spatiotemporal_vo/common.hpp"
#include "spatiotemporal_vo/config.hpp"
#include <vector>
#include <Eigen/Core>

namespace spatiotemporal_vo
{

  /**
   * @brief PEME (Patch-based Event Motion Estimation) 估计器
   * 基于时间表面的对比度最大化方法
   */
  class PEMEEstimator
  {
    using Vec6 = Eigen::Matrix<double, 6, 1>;

  public:
    // 配置参数结构体
    struct PEMEParams
    {
      double ts_start = 0.8;             // 时间表面起始比例
      double ts_end = 0.8;               // 时间表面结束比例
      int sample_count = 30000;          // 采样点数量
      int unit_count = 5000;             // 事件单元数量
      int iter_num = 10;                 // 外层迭代次数
      int denoise_num = 4;               // 去噪邻域阈值
      int ceres_iter_num = 10;           // Ceres 内部迭代次数
      int ceres_iter_thread = 8;         // Ceres 线程数
      int gaussian_size = 5;             // 高斯模糊核大小
      double gaussian_sigma = 0.5;       // 高斯模糊 sigma
      double default_value_factor = 1.0; // 默认值因子
      double overlap_init = 0.5;
    };

    /**
     * @brief 构造函数
     * @param fx, fy, cx, cy 相机内参
     * @param width, height 图像尺寸
     * @param config 配置对象
     */
    PEMEEstimator(double fx, double fy, double cx, double cy,
                  int width, int height, const Config &config);

    ~PEMEEstimator() = default;

    /**
     * @brief 运行 PEME 估计
     * @param events 输入事件数据包
     * @return 估计结果
     */
    RegistrationOutput<3> runRotation(const EventPacket &events);

    RegistrationOutput<6> runSixDof(const EventPacket &events);

    RegistrationOutput<3> runTranslation(const EventPacket &events);

    /**
     * @brief 计算重叠度
     */
    double overlapRatio(const Vec3 &transVelocity,
                        const Vec3 &angularVelocity,
                        const EventPacket &warped_bundle,
                        const EventPacket &next_warped_bundle);

  private:
    // 相机内参矩阵
    double fx_, fy_, cx_, cy_;
    int width_, height_, padding_ = 100;

    // 参数
    PEMEParams params_;

    // 估计状态
    Vec3 est_vec3_;
    Vec6 est_vec6_;
    int count_;
    bool use_dynamic_batch_ = false;

    EventPacket eventBundle_, event_warpped_Bundle_;

    /**
     * @brief 获取采样索引
     */
    void getSampledIndices(const cv::Mat &iwe,
                           std::vector<int> &indices, int count,
                           double start_ratio, double end_ratio);

    /**
     * @brief 获取 warp 后的事件图像
     */
    cv::Mat getWarpedEventImage(const Vec3 &transVelocity,
                                const Vec3 &angularVelocity,
                                EventPacket &warped_bundle,
                                std::vector<bool> &isInner,
                                bool use_later_time = false);

    /**
     * @brief 核心优化函数
     * @return 损失函数值
     */
    double optimizeRotation();

    double optimizeTranslation();

    double optimizeSixDof();
  };
} // namespace spatiotemporal_vo