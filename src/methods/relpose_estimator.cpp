#include "methods/relpose_estimator.hpp"
#include "relpose/optimizer_cop_flow.h" // adam_cop_flow
#include "spatiotemporal_vo/common.hpp"
#include "spatiotemporal_vo/config.hpp"
#include <Eigen/SVD>
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ctime>
#include <vector>

namespace spatiotemporal_vo {
RelposeEstimator::RelposeEstimator(double fx, double fy, double cx, double cy,
                                   int width, int height, const Config &config)
    : width_(width), height_(height), count_(1), fx_(fx), fy_(fy), cx_(cx),
      cy_(cy) {
  // 初始化估计状态
  est_vec3_.setZero();
  est_vec6_.setZero();
  count_ = 1;
  lines_.clear();
  clusters_.clear();
  normal_flows_.clear();
}

void RelposeEstimator::lineDetection(const EventPacket &events) {
  lines_.clear();
  if (events.empty())
    return;
  int width = width_ + 2 * padding_;
  int height = height_ + 2 * padding_;
  cv::Mat accum(height, width, CV_32FC1, cv::Scalar(0));

  for (const auto &pt : events.xy) {
    int col = static_cast<int>(pt.x() + padding_);
    int row = static_cast<int>(pt.y() + padding_);
    if (col >= 0 && col < width && row >= 0 && row < height)
      accum.at<float>(row, col) += 1.0f;
  }

  // 归一化到0-255
  cv::Mat accum_norm;
  cv::normalize(accum, accum_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);

  // HoughLinesP
  std::vector<cv::Vec4i> linesP;
  cv::HoughLinesP(accum_norm, linesP, 1, CV_PI / 180, 50, 50, 10); // 参数可调

  for (size_t i = 0; i < linesP.size(); ++i) {
    Line l;
    l.x1 = linesP[i][0] - padding_;
    l.y1 = linesP[i][1] - padding_;
    l.x2 = linesP[i][2] - padding_;
    l.y2 = linesP[i][3] - padding_;
    lines_.push_back(l);
  }
}
void RelposeEstimator::eventClustering(const EventPacket &events,
                                       double line_threshold) {
  clusters_.clear();
  if (events.empty() || lines_.empty())
    return;

  const size_t num_lines = lines_.size();
  const size_t num_events = events.size();
  clusters_.resize(num_lines);

  for (size_t i = 0; i < num_lines; ++i) {
    const Line &line = lines_[i];

    double dx = line.x2 - line.x1;
    double dy = line.y2 - line.y1;
    double a = dy;  // a = y2 - y1
    double b = -dx; // b = x1 - x2
    double c = line.x2 * line.y1 - line.x1 * line.y2;
    double denom = std::sqrt(a * a + b * b);

    // 忽略退化线段（端点重合）
    if (denom < 1e-12)
      continue;

    // 预计算线段长度平方，用于投影参数
    double seg_len_sq = dx * dx + dy * dy;

    std::vector<int> &cluster = clusters_[i];

    for (size_t j = 0; j < num_events; ++j) {
      double x = events.xy[j].x();
      double y = events.xy[j].y();

      // 垂直距离
      double dist = std::abs(a * x + b * y + c) / denom;
      if (dist > line_threshold)
        continue;

      // 端点限制：投影参数 t 必须在 [0, 1] 之间
      // t = ((x-x1)*dx + (y-y1)*dy) / seg_len_sq
      double t = ((x - line.x1) * dx + (y - line.y1) * dy) / seg_len_sq;
      if (t < 0.0 || t > 1.0)
        continue;

      // 通过双重检查，加入聚类
      cluster.push_back(static_cast<int>(j));
    }
  }
}

void RelposeEstimator::normalFlowEstimation(const EventPacket &events) {
  normal_flows_.clear();
  const int N = events.size();
  if (N == 0)
    return;
  normal_flows_.resize(N, Vec2(0, 0));

  const int L = 3;         // 空间窗口半径 (像素)
  const double DT = 0.001; // 时间窗口半径 (秒)，假设 t 为秒
  const double th1 = 1e-5; // 收敛阈值
  const double th2 = 0.05; // 外点距离阈值

  for (int i = 0; i < N; ++i) {
    double x_i = events.xy[i].x();
    double y_i = events.xy[i].y();
    double t_i = events.t[i];

    // 1. 收集邻域事件
    std::vector<int> neighbor_indices;
    neighbor_indices.reserve(N);
    for (int j = 0; j < N; ++j) {
      if (std::abs(events.xy[j].x() - x_i) <= L &&
          std::abs(events.xy[j].y() - y_i) <= L &&
          std::abs(events.t[j] - t_i) <= DT) {
        neighbor_indices.push_back(j);
      }
    }
    int M = neighbor_indices.size();
    if (M < 4) {
      normal_flows_[i] = Vec2(0, 0);
      continue;
    }

    // 构建矩阵 A (M x 4)
    Eigen::MatrixXd A(M, 4);
    for (int k = 0; k < M; ++k) {
      int idx = neighbor_indices[k];
      A(k, 0) = events.xy[idx].x();
      A(k, 1) = events.xy[idx].y();
      A(k, 2) = events.t[idx];
      A(k, 3) = 1.0;
    }

    // 初始 SVD 拟合
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d xi = svd.matrixV().col(3); // 最小奇异值对应的向量

    // 迭代外点去除
    Eigen::Vector4d xi0 = xi;
    const int MAX_ITER = 10; // 防止无限循环
    for (int iter = 0; iter < MAX_ITER; ++iter) {
      // 计算内点
      std::vector<int> inliers;
      inliers.reserve(M);
      for (int k = 0; k < M; ++k) {
        double r = std::abs(xi0(0) * A(k, 0) + xi0(1) * A(k, 1) +
                            xi0(2) * A(k, 2) + xi0(3));
        if (r <= th2) {
          inliers.push_back(k);
        }
      }
      if (inliers.size() < 4)
        break; // 内点太少

      // 重构矩阵
      Eigen::MatrixXd A_in(inliers.size(), 4);
      for (size_t k = 0; k < inliers.size(); ++k) {
        int row = inliers[k];
        A_in(k, 0) = A(row, 0);
        A_in(k, 1) = A(row, 1);
        A_in(k, 2) = A(row, 2);
        A_in(k, 3) = A(row, 3);
      }

      // 重新拟合
      Eigen::JacobiSVD<Eigen::MatrixXd> svd_in(A_in, Eigen::ComputeFullV);
      Eigen::Vector4d xi_new = svd_in.matrixV().col(3);

      // 保证方向一致，避免符号翻转导致收敛错误
      if (xi_new.dot(xi0) < 0) {
        xi_new = -xi_new;
      }
      double eps = (xi_new - xi0).norm();
      xi0 = xi_new;
      if (eps <= th1)
        break;
    }

    double a = xi0(0);
    double b = xi0(1);
    double c = xi0(2);
    // 避免 c 接近于 0
    if (std::abs(c) < 1e-12) {
      normal_flows_[i] = Vec2(0, 0);
      continue;
    }
    double vx = -a / c;
    double vy = -b / c;
    normal_flows_[i] = Vec2(vx, vy);
  }
}

RegistrationOutput<3> RelposeEstimator::runRotation(const EventPacket &events) {
  RegistrationOutput<3> result;

  if (events.empty()) {
    result.params.setZero();
    result.stats.loss = 0;
    result.stats.num_iterations = 0;
    return result;
  }

  std::cout << '[' << count_ << "] " << "Relpose: Processing " << events.size()
            << " events, duration: " << (events.t_end - events.t_start) << "s"
            << std::endl;
  last_events_ = events;

  // 运行优化
  result.stats.time_taken = omp_get_wtime();

  // 提取直线
  lineDetection(events);
  std::cout << "Detected " << lines_.size() << " lines_." << std::endl;
  // 聚类事件
  eventClustering(events, 5.0);
  // 估计法线流
  normalFlowEstimation(events);

  // 执行优化
  result.stats.loss = optimizeRotation();

  // 填充结果
  result.params = est_vec3_;
  result.stats.num_iterations = count_;
  result.stats.time_taken = omp_get_wtime() - result.stats.time_taken;

  std::cout << '[' << count_ << "] "
            << "Relpose: Estimated angular velocity: " << est_vec3_.transpose()
            << ", best loss: " << result.stats.loss << ", Optimization took "
            << result.stats.time_taken << " seconds." << std::endl;
  count_++;
  return result;
}

double RelposeEstimator::optimizeRotation() {
  // 无聚类数据或原始事件时直接返回
  if (clusters_.empty() || last_events_.empty()) {
    return 0.0;
  }

  // ---------- 步骤1：根据聚类索引构建 vector<vector<Event>> ----------
  std::vector<std::vector<Event>> event_groups;
  event_groups.reserve(clusters_.size());

  for (size_t i = 0; i < clusters_.size(); ++i) {
    const auto &indices = clusters_[i];
    if (indices.empty())
      continue; // 跳过没有事件的直线

    std::vector<Event> group;
    group.reserve(indices.size());
    for (int idx : indices) {
      Event ev;
      ev.t = last_events_.t[idx];
      ev.img = last_events_.xy[idx];

      // 计算 bearing_vec：像素反投影方向（单位向量）
      double x = last_events_.xy[idx].x();
      double y = last_events_.xy[idx].y();
      double ux = (x - cx_) / fx_;
      double uy = (y - cy_) / fy_;
      Eigen::Vector3d bearing(ux, uy, 1.0);
      bearing.normalize();
      ev.bearing_vec = bearing;

      // 计算 plane_nml：由法线流构造时空平面法线
      // 法线流 (vx, vy) = -A/C, -B/C，平面法线正比于 (-vx, -vy, 1)
      Vec2 nf = normal_flows_[idx];
      if (nf.x() == 0.0 && nf.y() == 0.0) {
        // 法线流为零时用默认值，避免退化（对应纯垂直于图像平面的平面）
        ev.plane_nml = Eigen::Vector3d(0, 0, 1);
      } else {
        Eigen::Vector3d pn(-nf.x(), -nf.y(), 1.0);
        pn.normalize();
        ev.plane_nml = pn;
      }
      group.push_back(ev);
    }
    event_groups.push_back(group);
  }

  if (event_groups.empty()) {
    return 0.0;
  }

  // ---------- 步骤2：调用 Adam 优化器 ----------
  Vector3d x0 = est_vec3_; // 使用上一次估计作为初值（首次为零）

  double loss =
      adam_cop_flow(event_groups, x0,
                    301, // 方法：精确解析梯度（obj_grad_cop_flow）
                    1.0, // scale_lambda（可根据需要调整）
                    1e-6, 1.0, params_.tol_fun, params_.tol_x, params_.max_iter,
                    params_.max_fun_evals, params_.lr, params_.beta1,
                    params_.beta2, params_.epsilon, params_.epoch_size);

  // 更新估计结果
  est_vec3_ = x0;
  return loss;
}
} // namespace spatiotemporal_vo