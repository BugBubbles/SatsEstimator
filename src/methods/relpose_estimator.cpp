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
  cv::GaussianBlur(accum_norm, accum_norm, cv::Size(3, 3), 0);
  // 计算Canny边缘图
  cv::Mat edges;
  cv::Canny(accum_norm, edges, 50, 150, 3); // 参数可调

  // HoughLinesP
  std::vector<cv::Vec4i> linesP;
  cv::HoughLinesP(edges, linesP, 1, CV_PI / 60, 30, 30, 5); // 参数可调

  for (size_t i = 0; i < linesP.size(); ++i) {
    Line l;
    l.x1 = linesP[i][0] - padding_;
    l.y1 = linesP[i][1] - padding_;
    l.x2 = linesP[i][2] - padding_;
    l.y2 = linesP[i][3] - padding_;
    lines_.push_back(l);
  }

  // #ifdef DEBUG
  //   // 可视化检测结果
  //   cv::Mat color_accum;
  //   cv::cvtColor(edges, color_accum, cv::COLOR_GRAY2BGR);
  //   for (size_t i = 0; i < linesP.size(); ++i) {
  //     cv::line(color_accum, cv::Point(linesP[i][0], linesP[i][1]),
  //              cv::Point(linesP[i][2], linesP[i][3]), cv::Scalar(0, 0, 255),
  //              2);
  //   }
  //   cv::imwrite("HoughLinesP.png", color_accum);
  // #endif
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
      clusters_[i].push_back(static_cast<int>(j));
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
  std::cout << "Detected " << lines_.size() << " lines." << std::endl;
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

// double RelposeEstimator::optimizeRotation() {
//   if (clusters_.empty() || last_events_.empty()) {
//     return 0.0;
//   }

//   std::vector<std::vector<Event>> event_groups;
//   event_groups.reserve(clusters_.size());

//   for (size_t i = 0; i < clusters_.size(); ++i) {
//     const auto &indices = clusters_[i];
//     if (indices.size() < 8)
//       continue; // 事件数太少，忽略该直线

//     std::vector<Event> group;
//     group.reserve(indices.size());

//     for (int idx : indices) {
//       // ---- bearing 向量（步骤 1）----
//       double x = last_events_.xy[idx].x();
//       double y = last_events_.xy[idx].y();
//       double ux = (x - cx_) / fx_;
//       double uy = (y - cy_) / fy_;
//       Vec3 bearing(ux, uy, 1.0);
//       bearing.normalize();

//       // ---- 平面法向量计算（步骤 2+3）----
//       Vec2 nf = normal_flows_[idx];
//       if (nf.squaredNorm() < 1e-12) // 法向流太小，无法计算有效法向量
//         continue;

//       Vec3 h(-nf.y(), nf.x(), 0.0); // 线方向（步骤 2）
//       Vec3 n = bearing.cross(h);    // 叉积（步骤 3）
//       double n_norm = n.norm();
//       if (n_norm < 1e-12) // 叉积退化
//         continue;
//       n.normalize();

//       // ---- 构造 Event（不含旋转补偿，角速度未知）----
//       Event ev;
//       ev.t = last_events_.t[idx];
//       ev.img = last_events_.xy[idx];
//       ev.bearing_vec = bearing; // 保留轴承向量（可能另有用处）
//       ev.plane_nml = n;         // 相机坐标系下的平面法线
//       group.push_back(ev);
//     }

//     if (!group.empty())
//       event_groups.push_back(group);
//   }

//   if (event_groups.empty())
//     return 0.0;

//   // 数值保护：防止角速度初值异常
//   if (std::isnan(est_vec3_.norm()) || std::isinf(est_vec3_.norm()))
//     est_vec3_.setZero();

//   double loss = adam_cop_flow(
//       event_groups, est_vec3_, 301, 1.0, 1e-6, 1.0, params_.tol_fun,
//       params_.tol_x, params_.max_iter, params_.max_fun_evals, params_.lr,
//       params_.beta1, params_.beta2, params_.epsilon, params_.epoch_size);

//   return loss;
// }
double RelposeEstimator::optimizeRotation() {
  // 0. 构建事件组（每条直线一个组），与之前相同
  if (clusters_.empty() || last_events_.empty())
    return 0.0;

  std::vector<std::vector<Event>> event_groups;
  event_groups.reserve(clusters_.size());

  for (size_t i = 0; i < clusters_.size(); ++i) {
    const auto &indices = clusters_[i];
    if (indices.empty())
      continue;

    std::vector<Event> group;
    group.reserve(indices.size());
    for (int idx : indices) {
      // bearing 向量
      double x = last_events_.xy[idx].x();
      double y = last_events_.xy[idx].y();
      double ux = (x - cx_) / fx_;
      double uy = (y - cy_) / fy_;
      Eigen::Vector3d bearing(ux, uy, 1.0);
      bearing.normalize();

      // 平面法向量（从法向流计算）
      Vec2 nf = normal_flows_[idx];
      if (nf.squaredNorm() < 1e-12)
        continue;
      Eigen::Vector3d h(-nf.y(), nf.x(), 0.0);
      Eigen::Vector3d n = bearing.cross(h);
      double n_norm = n.norm();
      if (n_norm < 1e-12)
        continue;
      n.normalize();

      Event ev;
      ev.t = last_events_.t[idx];
      ev.img = last_events_.xy[idx];
      ev.bearing_vec = bearing;
      ev.plane_nml = n;
      group.push_back(ev);
    }
    if (!group.empty())
      event_groups.push_back(group);
  }

  if (event_groups.empty())
    return 0.0;

  // 1. GC‑RANSAC 参数
  const int max_iter = 1000;
  const double angle_thresh_deg = 0.2; // 角度阈值（度）
  const double sin_thresh =
      std::sin(angle_thresh_deg * M_PI / 180.0); // ≈0.00349
  const int sample_per_line = 8;                 // 每条线最少采样事件数
  const int inner_refine_iters = 20;             // 局部精化迭代次数
  const double confidence = 0.99;

  // 将所有事件展开为全局索引（用于内点标记）
  const int total_events = [&]() {
    int sum = 0;
    for (auto &g : event_groups)
      sum += g.size();
    return sum;
  }();
  std::vector<int> group_offset(event_groups.size(), 0);
  int offset = 0;
  for (size_t i = 0; i < event_groups.size(); ++i) {
    group_offset[i] = offset;
    offset += event_groups[i].size();
  }

  // 随机数设置
  std::srand(static_cast<unsigned>(std::time(nullptr)));

  Vector3d best_omega = est_vec3_;
  int best_score = 0;
  std::vector<bool> best_inliers(total_events, false);

  int iterations = 0;
  double w = 0.1; // 初始内点比例估计
  double prob = 0.0;

  // ---------- 主迭代 ----------
  while (iterations < max_iter && prob < confidence) {
    // 步骤1：空间一致性采样（每条线随机选 sample_per_line 个事件）
    std::vector<std::vector<Event>> sampled_groups(event_groups.size());
    int total_sampled = 0;
    for (size_t i = 0; i < event_groups.size(); ++i) {
      const auto &group = event_groups[i];
      if (group.empty())
        continue;
      int k = std::min(sample_per_line, static_cast<int>(group.size()));
      std::vector<int> pool(group.size());
      std::iota(pool.begin(), pool.end(), 0);
      std::random_shuffle(pool.begin(), pool.end());
      for (int j = 0; j < k; ++j) {
        sampled_groups[i].push_back(group[pool[j]]);
        total_sampled++;
      }
    }
    if (total_sampled < 4)
      continue; // 最小集不足

    // 步骤2：用采样集估计角速度（Adam 优化）
    Vector3d omega_cand =
        best_omega.norm() > 1e-12 ? best_omega : Vector3d::Zero();
    double loss = adam_cop_flow(
        sampled_groups, omega_cand, 301, 1.0, 1e-6, 1.0, params_.tol_fun,
        params_.tol_x, params_.max_iter, params_.max_fun_evals, params_.lr,
        params_.beta1, params_.beta2, params_.epsilon, params_.epoch_size);
    if (std::isnan(omega_cand.norm()))
      continue;

    // 步骤3：计算角度残差（全部事件）
    std::vector<bool> inliers(total_events, false);
    int inlier_count = 0;
    for (size_t g = 0; g < event_groups.size(); ++g) {
      const auto &group = event_groups[g];
      for (size_t e = 0; e < group.size(); ++e) {
        const Event &ev = group[e];
        // 计算旋转矩阵 R_j
        double t = ev.t;
        double theta = t * omega_cand.norm();
        Matrix3d R;
        if (theta > THRESHOLD) {
          Matrix3d omega_cross;
          omega_cross << 0, -omega_cand(2) * t, omega_cand(1) * t,
              omega_cand(2) * t, 0, -omega_cand(0) * t, -omega_cand(1) * t,
              omega_cand(0) * t, 0;
          double a = std::sin(theta) / theta;
          double b = (1 - std::cos(theta)) / (theta * theta);
          R = Matrix3d::Identity() + a * omega_cross +
              b * omega_cross * omega_cross;
        } else {
          Matrix3d omega_cross;
          omega_cross << 0, -omega_cand(2) * t, omega_cand(1) * t,
              omega_cand(2) * t, 0, -omega_cand(0) * t, -omega_cand(1) * t,
              omega_cand(0) * t, 0;
          R = Matrix3d::Identity() + omega_cross;
        }
        Vector3d f_rot = R * ev.bearing_vec;
        double dot_abs = std::abs(ev.plane_nml.dot(f_rot));
        if (dot_abs <= sin_thresh) {
          int global_idx = group_offset[g] + e;
          inliers[global_idx] = true;
          inlier_count++;
        }
      }
    }

    // 步骤4,5：更新最佳模型
    if (inlier_count > best_score) {
      best_score = inlier_count;
      best_omega = omega_cand;
      best_inliers = inliers;
    }

    // 步骤6：更新概率
    w = std::max(0.05, static_cast<double>(best_score) / total_events);
    double n = sample_per_line * event_groups.size();
    prob =
        (w >= 1.0) ? 1.0 : 1.0 - std::pow(1.0 - std::pow(w, n), iterations + 1);
    iterations++;
  }

  // ---------- 局部精化 ----------
  if (best_score > 0) {
    Vector3d refined_omega = best_omega;
    int refined_score = best_score;
    auto refined_inliers = best_inliers;
    for (int iter = 0; iter < inner_refine_iters; ++iter) {
      // 从当前内点集中随机抽取一部分
      std::vector<int> inlier_global_ids;
      for (int i = 0; i < total_events; ++i)
        if (refined_inliers[i])
          inlier_global_ids.push_back(i);
      if (inlier_global_ids.size() < 8)
        break;
      std::random_shuffle(inlier_global_ids.begin(), inlier_global_ids.end());
      int take = std::min(static_cast<int>(inlier_global_ids.size()), 100);

      // 构建采样组（保持线结构，但只使用选中内点）
      std::vector<std::vector<Event>> refine_groups(event_groups.size());
      for (int id : inlier_global_ids) {
        // 找到所属的线和局部索引
        for (size_t g = 0; g < event_groups.size(); ++g) {
          if (id >= group_offset[g] &&
              id < group_offset[g] + (int)event_groups[g].size()) {
            int local_idx = id - group_offset[g];
            refine_groups[g].push_back(event_groups[g][local_idx]);
            break;
          }
        }
      }
      // 移除空组
      refine_groups.erase(
          std::remove_if(refine_groups.begin(), refine_groups.end(),
                         [](const auto &v) { return v.empty(); }),
          refine_groups.end());

      Vector3d cand_omega = refined_omega;
      double loss = adam_cop_flow(
          refine_groups, cand_omega, 301, 1.0, 1e-6, 1.0, params_.tol_fun,
          params_.tol_x, params_.max_iter, params_.max_fun_evals, params_.lr,
          params_.beta1, params_.beta2, params_.epsilon, params_.epoch_size);
      if (std::isnan(cand_omega.norm()))
        continue;

      // 评估新模型
      std::vector<bool> new_inliers(total_events, false);
      int new_count = 0;
      // ...（与主循环中相同的残差计算）
      for (size_t g = 0; g < event_groups.size(); ++g) {
        const auto &group = event_groups[g];
        for (size_t e = 0; e < group.size(); ++e) {
          const Event &ev = group[e];
          double t = ev.t;
          double theta = t * cand_omega.norm();
          Matrix3d R;
          if (theta > THRESHOLD) {
            Matrix3d omega_cross;
            omega_cross << 0, -cand_omega(2) * t, cand_omega(1) * t,
                cand_omega(2) * t, 0, -cand_omega(0) * t, -cand_omega(1) * t,
                cand_omega(0) * t, 0;
            double a = std::sin(theta) / theta;
            double b = (1 - std::cos(theta)) / (theta * theta);
            R = Matrix3d::Identity() + a * omega_cross +
                b * omega_cross * omega_cross;
          } else {
            Matrix3d omega_cross;
            omega_cross << 0, -cand_omega(2) * t, cand_omega(1) * t,
                cand_omega(2) * t, 0, -cand_omega(0) * t, -cand_omega(1) * t,
                cand_omega(0) * t, 0;
            R = Matrix3d::Identity() + omega_cross;
          }
          Vector3d f_rot = R * ev.bearing_vec;
          if (std::abs(ev.plane_nml.dot(f_rot)) <= sin_thresh) {
            int global_idx = group_offset[g] + e;
            new_inliers[global_idx] = true;
            new_count++;
          }
        }
      }
      if (new_count > refined_score) {
        refined_score = new_count;
        refined_omega = cand_omega;
        refined_inliers = new_inliers;
      }
    }
    best_omega = refined_omega;
    best_score = refined_score;
    best_inliers = refined_inliers;
  }

  // ---------- 最终估计（使用所有最佳内点）----------
  if (best_score > 0) {
    std::vector<std::vector<Event>> final_groups(event_groups.size());
    for (size_t g = 0; g < event_groups.size(); ++g) {
      for (size_t e = 0; e < event_groups[g].size(); ++e) {
        int idx = group_offset[g] + e;
        if (best_inliers[idx])
          final_groups[g].push_back(event_groups[g][e]);
      }
    }
    // 移除空组
    final_groups.erase(std::remove_if(final_groups.begin(), final_groups.end(),
                                      [](const auto &v) { return v.empty(); }),
                       final_groups.end());
    if (!final_groups.empty()) {
      Vector3d final_omega = best_omega;
      double final_loss = adam_cop_flow(
          final_groups, final_omega, 301, 1.0, 1e-6, 1.0, params_.tol_fun,
          params_.tol_x, params_.max_iter, params_.max_fun_evals, params_.lr,
          params_.beta1, params_.beta2, params_.epsilon, params_.epoch_size);
      if (!std::isnan(final_omega.norm())) {
        est_vec3_ = final_omega;
        return final_loss;
      }
    }
  }

  // 回退：GC‑RANSAC 失败时，用全部事件优化
  est_vec3_.setZero();
  return adam_cop_flow(event_groups, est_vec3_, 301, 1.0, 1e-6, 1.0,
                       params_.tol_fun, params_.tol_x, params_.max_iter,
                       params_.max_fun_evals, params_.lr, params_.beta1,
                       params_.beta2, params_.epsilon, params_.epoch_size);
}
} // namespace spatiotemporal_vo