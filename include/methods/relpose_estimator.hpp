#pragma once

#include "spatiotemporal_vo/common.hpp"
#include "spatiotemporal_vo/config.hpp"
#include <Eigen/Core>
#include <vector>

namespace spatiotemporal_vo {

/**
 * @brief Relpose (Patch-based Event Motion Estimation) 估计器
 * 基于时间表面的对比度最大化方法
 */
class RelposeEstimator {
  using Vec6 = Eigen::Matrix<double, 6, 1>;
  struct Line {
    double x1, y1, x2, y2;
  };
  struct Params {
    double tol_fun = 1e-6;   // 函数值收敛阈值
    double tol_x = 1e-6;     // 变量收敛阈值
    int max_iter = 100;      // 最大迭代次数
    int max_fun_evals = 500; // 最大函数评估次数
    double lr = 1e-3;        // Adam 步长
    double beta1 = 0.9;      // Adam beta1
    double beta2 = 0.999;    // Adam beta2
    double epsilon = 1e-8;   // Adam epsilon
    int epoch_size = 10;     // 平滑窗口大小
  };

public:
  /**
   * @brief 构造函数
   * @param fx, fy, cx, cy 相机内参
   * @param width, height 图像尺寸
   * @param config 配置对象
   */
  RelposeEstimator(double fx, double fy, double cx, double cy, int width,
                   int height, const Config &config);

  ~RelposeEstimator() = default;

  /**
   * @brief 运行 Relpose 估计
   * @param events 输入事件数据包
   * @return 估计结果
   */
  RegistrationOutput<3> runRotation(const EventPacket &events);

  RegistrationOutput<6> runSixDof(const EventPacket &events);

  RegistrationOutput<3> runTranslation(const EventPacket &events);

private:
  // 相机内参矩阵
  double fx_, fy_, cx_, cy_;
  int width_, height_, padding_ = 100;

  // 估计状态
  Vec3 est_vec3_;
  Vec6 est_vec6_;
  int count_;
  std::vector<Vec2> normal_flows_;
  std::vector<std::vector<int>> clusters_;
  std::vector<Line> lines_;
  Params params_;

  // RelposeEstimator 私有成员中增加：
  EventPacket last_events_; // 保存最近一次 runRotation 的原始事件
  /**
   * @brief 累积事件图像并执行直线检测
   * @param events 输入事件数据包
   * @param lines 输出检测到的直线列表，每条直线由二维起点与终点坐标表示
   */
  void lineDetection(const EventPacket &events);

  /**
   * @brief 事件聚类，利用给定的事件流、直线段，将事件点分配给对应的直线段
   * @param events 输入事件数据包
   * @param lines 输入直线段列表，每条直线由二维起点与终点坐标表示
   * @param line_threshold
   * 事件点到直线的距离阈值，超过该距离的事件点将被视为噪声，默认为5像素
   * @param clusters 输出事件聚类结果，每个聚类为一个事件点列表
   */
  void eventClustering(const EventPacket &events, double line_thresholds);
  /**
   * @brief 法向流估计，沿亮度梯度方向的速度分量
   * @param events 输入事件数据包
   * @param lines 输入直线段列表，每条直线由二维起点与
   */
  void normalFlowEstimation(const EventPacket &events);

  /**
   * @brief 核心优化函数
   * @return 损失函数值
   */
  double optimizeRotation();

  double optimizeTranslation();

  double optimizeSixDof();
};
} // namespace spatiotemporal_vo