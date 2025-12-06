#include "spatiotemporal_vo/registration.hpp"
#include "spatiotemporal_vo/common.hpp"
#include <ceres/ceres.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <fstream>

namespace spatiotemporal_vo
{
  // ============================================================================
  // 辅助函数：SVD 求解最优旋转（参考 Go-ICP 的 jly_icp3d.hpp）
  // ============================================================================
  static Eigen::Matrix3d computeOptimalRotation(
      const std::vector<Vec3> &src,            // 源点云（已去质心）
      const std::vector<Vec3> &tgt,            // 目标点云（已去质心）
      const std::vector<int> &correspondences) // 对应关系
  {
    // 计算协方差矩阵 H = sum(src_i * tgt_i^T)
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < correspondences.size(); ++i)
    {
      if (correspondences[i] >= 0)
      {
        H += src[i] * tgt[correspondences[i]].transpose();
      }
    }

    // SVD 分解
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // 计算旋转矩阵 R = V * U^T
    Eigen::Matrix3d R = V * U.transpose();

    // 确保 R 是一个有效的旋转矩阵（行列式为 1）
    if (R.determinant() < 0)
    {
      V.col(2) *= -1;
      R = V * U.transpose();
    }

    return R;
  }

  // ============================================================================
  // ICP 核心算法（参考 Go-ICP 的 jly_icp3d.hpp 中的 Run 函数）
  // ============================================================================
  double SpatioTemporalRegistration::runICP(
      const std::vector<Vec3> &data_points,      // 源点云（待配准）
      const std::vector<Vec3> &model_points,     // 目标点云（模型）
      std::vector<TemporalNeighborhood> &neighborhoods, // 目标点云的 KDTree
      Eigen::Matrix3d &R_result,                 // 输出：最优旋转
      int max_iterations,                        // 最大迭代次数
      double trim_fraction)                      // 截断比例
  {
    int num_data = static_cast<int>(data_points.size());
    int num_model = static_cast<int>(model_points.size());

    if (num_data == 0 || num_model == 0)
    {
      R_result = Eigen::Matrix3d::Identity();
      return std::numeric_limits<double>::max();
    }

    // 计算要保留的点数（Trimmed ICP）
    int num_trimmed = static_cast<int>(trim_fraction * num_data);
    if (num_trimmed < 10)
      num_trimmed = std::min(10, num_data);

    // 初始化
    Eigen::Matrix3d R_current = R_result;
    double prev_error = std::numeric_limits<double>::max();
    double current_error = 0.0;
    double best_error = 100000000;
    Eigen::Matrix3d best_R = R_result;

    std::vector<double> distances(num_data);
    std::vector<int> correspondences(num_data);
    std::vector<int> sorted_indices(num_data);

    for (int iter = 0; iter < max_iterations; ++iter)
    {
      // ========== Step 1: 变换源点云并寻找最近邻 ==========
      std::vector<int> matches;
      for (int i = 0; i < num_data; ++i)
      {
        if (neighborhoods[i].points_3d.empty())
          continue;
        Vec3 transformed_point = R_current * data_points[i];

        pcl::PointXYZ search_point;
        search_point.x = static_cast<float>(transformed_point.x());
        search_point.y = static_cast<float>(transformed_point.y());
        search_point.z = static_cast<float>(transformed_point.z());

        std::vector<int> pointIdxNKNSearch(1);
        std::vector<float> pointNKNSquaredDistance(1);

        if (neighborhoods[i].kdtree.nearestKSearch(search_point, 1,
                                                   pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
          correspondences[i] = neighborhoods[i].original_indices[pointIdxNKNSearch[0]];
          distances[i] = pointNKNSquaredDistance[0];
        }
        else
        {
          correspondences[i] = -1;
          distances[i] = std::numeric_limits<double>::max();
        }

        sorted_indices[i] = i;
      }

      // ========== Step 2: Trimmed ICP - 按距离排序并截断 ==========
      std::sort(sorted_indices.begin(), sorted_indices.end(),
                [&distances](int a, int b)
                { return distances[a] < distances[b]; });

      // 计算截断后的误差
      current_error = 0.0;
      for (int i = 0; i < num_trimmed; ++i)
      {
        current_error += distances[sorted_indices[i]];
      }
      current_error /= num_trimmed;

      // ========== Step 3: 检查收敛 ==========
      if (std::abs(prev_error - current_error) < 1e-12)
      {
        break;
      }
      if (best_error > current_error){
        best_error = current_error;
        best_R = R_current;
      }
      prev_error = current_error;

      // ========== Step 4: 使用 SVD 计算最优旋转 ==========
      // 准备去质心的点对
      std::vector<Vec3> src_centered;
      std::vector<Vec3> tgt_centered;
      std::vector<int> valid_correspondences;

      Vec3 src_centroid = Vec3::Zero();
      Vec3 tgt_centroid = Vec3::Zero();
      int valid_count = 0;

      // 只使用截断后的点计算质心
      for (int i = 0; i < num_trimmed; ++i)
      {
        int idx = sorted_indices[i];
        if (correspondences[idx] >= 0)
        {
          Vec3 src_pt = R_current * data_points[idx];
          Vec3 tgt_pt = model_points[correspondences[idx]];
          src_centroid += src_pt;
          tgt_centroid += tgt_pt;
          valid_count++;
        }
      }

      if (valid_count < 3)
      {
        break; // 点太少，无法计算旋转
      }

      src_centroid /= valid_count;
      tgt_centroid /= valid_count;

      // 去质心
      src_centered.reserve(valid_count);
      tgt_centered.reserve(valid_count);
      valid_correspondences.reserve(valid_count);

      int corr_idx = 0;
      for (int i = 0; i < num_trimmed; ++i)
      {
        int idx = sorted_indices[i];
        if (correspondences[idx] >= 0)
        {
          Vec3 src_pt = R_current * data_points[idx] - src_centroid;
          Vec3 tgt_pt = model_points[correspondences[idx]] - tgt_centroid;
          src_centered.push_back(src_pt);
          tgt_centered.push_back(tgt_pt);
          valid_correspondences.push_back(corr_idx);
          corr_idx++;
        }
      }

      // 计算增量旋转
      Eigen::Matrix3d delta_R = computeOptimalRotation(src_centered, tgt_centered, valid_correspondences);

      // 更新累积旋转
      R_current = delta_R * R_current;

      // 确保旋转矩阵的正交性（防止数值漂移）
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_current, Eigen::ComputeFullU | Eigen::ComputeFullV);
      R_current = svd.matrixU() * svd.matrixV().transpose();
      // Vec3 delta_R;
      // optimizeWithCeres(src_centered, tgt_centered, valid_correspondences, delta_R);

      // R_current = SO3::exp(delta_R).matrix() * R_current;
    }

    R_result = best_R;
    return current_error;
  }

  // ============================================================================
  // 构造函数
  // ============================================================================
  SpatioTemporalRegistration::SpatioTemporalRegistration(
      double fx, double fy, double cx, double cy, const Config &cfg)
      : fx_(fx), fy_(fy), cx_(cx), cy_(cy),
        epsilon_T_(cfg.solver.epsilon_T),
        max_iters_(cfg.solver.max_iters),
        lr_(cfg.solver.lr),
        trim_ratio_(cfg.solver.trim_ratio),
        convergence_thresh_(cfg.solver.convergence_thresh)
  {
  }

  Vec2 SpatioTemporalRegistration::projectPoint(const Vec3 &p)
  {
    if (std::abs(p(2)) < 1e-8)
    {
      return Vec2(cx_, cy_);
    }
    double u = fx_ * p(0) / p(2) + cx_;
    double v = fy_ * p(1) / p(2) + cy_;
    return Vec2(u, v);
  }

  Vec3 SpatioTemporalRegistration::backprojectPoint(const Vec2 &uv, double d)
  {
    double x = (uv(0) - cx_) / fx_ * d;
    double y = (uv(1) - cy_) / fy_ * d;
    Vec3 v(x, y, d);
    return v / v.norm();
  }

  std::vector<Vec3> SpatioTemporalRegistration::backprojectPoints(
      const std::vector<Vec2> &uvs,
      const std::vector<double> &depths)
  {
    std::vector<Vec3> points;
    points.reserve(uvs.size());
    for (size_t i = 0; i < uvs.size(); ++i)
    {
      points.push_back(backprojectPoint(uvs[i], depths[i]));
    }
    return points;
  }

  std::vector<SpatioTemporalRegistration::TemporalNeighborhood>
  SpatioTemporalRegistration::buildTemporalNeighborhoods(
      const std::vector<double> &ts_a,
      const std::vector<double> &ts_b,
      const std::vector<Vec3> &pts_cam_b,
      double dt)
  {
    std::vector<TemporalNeighborhood> neighborhoods;
    neighborhoods.reserve(ts_a.size());

    double time_window = epsilon_T_ * dt;

    for (size_t i = 0; i < ts_a.size(); ++i)
    {
      double target_time = ts_a[i] + 0.5 * dt;

      TemporalNeighborhood neighborhood;

      for (size_t j = 0; j < ts_b.size(); ++j)
      {
        if (std::abs(ts_b[j] - target_time) < time_window)
        {
          neighborhood.original_indices.push_back(static_cast<int>(j));
          neighborhood.points_3d.push_back(pts_cam_b[j]);
        }
      }

      if (!neighborhood.points_3d.empty())
      {
        // 将 std::vector<Eigen::Vector3d> 转换为 pcl::PointCloud<pcl::PointXYZ>
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto &point : neighborhood.points_3d)
        {
          pcl::PointXYZ pcl_point;
          pcl_point.x = static_cast<float>(point.x());
          pcl_point.y = static_cast<float>(point.y());
          pcl_point.z = static_cast<float>(point.z());
          pcl_cloud->points.push_back(pcl_point);
        }
        pcl_cloud->width = pcl_cloud->points.size();
        pcl_cloud->height = 1;
        pcl_cloud->is_dense = true;

        // 设置 KDTree 的输入点云
        neighborhood.kdtree.setInputCloud(pcl_cloud);
      }

      neighborhoods.push_back(std::move(neighborhood));
    }

    return neighborhoods;
  }
  // 主运行函数（使用 Go-ICP 风格的 ICP）
  // ============================================================================
  RegistrationOutput SpatioTemporalRegistration::run(
      const std::vector<Vec2> &events_xy,
      const std::vector<double> &events_t,
      const std::vector<int> &polarity,
      const std::vector<double> &depth_in)
  {
    RegistrationOutput output;

    if (events_t.empty())
    {
      return output;
    }

    // 计算时间范围
    double t_min = events_t.front();
    double t_max = events_t.back();
    double dt = t_max - t_min;
    double mid_time = 0.5 * (t_max + t_min);

#ifdef DEBUG
    std::cout << "[DEBUG] Time span is " << t_min << " " << t_max << std::endl;
#endif
    output.timespan = {t_min, t_max};

    if (dt < 1e-9)
    {
      std::cerr << "[Registration] Time range too small" << std::endl;
      return output;
    }

    // 1. 按时间分割数据
    std::vector<int> idx_a, idx_b;
    for (size_t i = 0; i < events_t.size(); ++i)
    {
      if (events_t[i] <= mid_time)
      {
        idx_a.push_back(static_cast<int>(i));
      }
      else
      {
        idx_b.push_back(static_cast<int>(i));
      }
    }

    if (idx_a.empty() || idx_b.empty())
    {
      std::cerr << "[Registration] Not enough events in both halves" << std::endl;
      return output;
    }

    // 2. 提取子集
    std::vector<Vec2> Ia, Ib;
    std::vector<double> ts_a, ts_b;
    Ia.reserve(idx_a.size());
    Ib.reserve(idx_b.size());
    ts_a.reserve(idx_a.size());
    ts_b.reserve(idx_b.size());

    for (int i : idx_a)
    {
      Ia.push_back(events_xy[i]);
      ts_a.push_back(events_t[i]);
    }
    for (int i : idx_b)
    {
      Ib.push_back(events_xy[i]);
      ts_b.push_back(events_t[i]);
    }

#ifdef DEBUG
    std::cout << "[DEBUG] The shape of Ia is " << Ia.size() << std::endl;
    std::cout << "[DEBUG] The shape of Ib is " << Ib.size() << std::endl;
#endif

    // 3. 准备深度
    std::vector<double> depth_a(idx_a.size(), 1.0);
    std::vector<double> depth_b(idx_b.size(), 1.0);

    if (!depth_in.empty())
    {
      for (size_t i = 0; i < idx_a.size(); ++i)
      {
        if (idx_a[i] < static_cast<int>(depth_in.size()))
        {
          depth_a[i] = depth_in[idx_a[i]];
        }
      }
      for (size_t i = 0; i < idx_b.size(); ++i)
      {
        if (idx_b[i] < static_cast<int>(depth_in.size()))
        {
          depth_b[i] = depth_in[idx_b[i]];
        }
      }
    }

    // 4. 反投影到 3D（归一化到单位球面）
    std::vector<Vec3> pts_cam_a = backprojectPoints(Ia, depth_a);
    std::vector<Vec3> pts_cam_b = backprojectPoints(Ib, depth_b);

    // 5. 构建目标点云的全局 KDTree
    auto kdtree = buildTemporalNeighborhoods(ts_a, ts_b, pts_cam_b, dt);

#ifdef DEBUG
    std::cout << "[DEBUG] The neighborhoods is finished." << std::endl;
#endif

    // ========================================================================
    // 6. Go-ICP 风格的 ICP 优化
    // ========================================================================
    Eigen::Matrix3d R_best = Eigen::Matrix3d::Identity();
    double best_loss = std::numeric_limits<double>::max();

    // 外层循环：多次重启以避免局部最优
    int num_restarts = 2; // 可以增加重启次数以提高鲁棒性

    for (int restart = 0; restart < num_restarts; ++restart)
    {
      Eigen::Matrix3d R_current = Eigen::Matrix3d::Identity();

      // 如果是重启，添加随机扰动（可选）
      if (restart > 0)
      {
        // 添加小的随机旋转作为初始值
        Vec3 random_axis = Vec3::Random().normalized();
        double random_angle = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 0.2; // [-0.1, 0.1] rad
        R_current = Eigen::AngleAxisd(random_angle, random_axis).toRotationMatrix();
      }

      // 运行 Trimmed ICP
      double loss = runICP(pts_cam_a, pts_cam_b, kdtree, R_current,
                           max_iters_, trim_ratio_);

      if (loss < best_loss)
      {
        best_loss = loss;
        R_best = R_current;
      }
    }

    // 7. 将旋转矩阵转换为旋转向量
    SO3 so3_result(R_best);

    // 8. 输出结果
    output.twist = -so3_result.log() / dt * 2.0;
    output.stats.loss = best_loss;
    output.stats.num_iterations = count_;

    std::cout << "[" << count_ << "] "
              << "Final loss: " << best_loss
              << " Optimized xi: " << output.twist.transpose() << std::endl;

    count_++;
    return output;
  }

  // ============================================================================
  // 保留 Ceres 优化函数（可选，用于对比或精细化）
  // ============================================================================
  struct RegistrationResidual
  {
    RegistrationResidual(const Eigen::Vector3d &p_a, const Eigen::Vector3d &p_b)
        : p_a_normalized_(p_a.normalized()), p_b_normalized_(p_b.normalized()) {}

    template <typename T>
    bool operator()(const T *const xi, T *residual) const
    {
      Eigen::Matrix<T, 3, 1> xi_vec(xi[0], xi[1], xi[2]);
      Sophus::SO3<T> R = Sophus::SO3<T>::exp(xi_vec);

      Eigen::Matrix<T, 3, 1> p_a_w = R * p_a_normalized_.cast<T>();
      Eigen::Matrix<T, 3, 1> res = p_a_w - p_b_normalized_.cast<T>();

      residual[0] = res[0];
      residual[1] = res[1];
      residual[2] = res[2];

      return true;
    }

    const Eigen::Vector3d p_a_normalized_;
    const Eigen::Vector3d p_b_normalized_;
  };

  void SpatioTemporalRegistration::optimizeWithCeres(
      const std::vector<Eigen::Vector3d> &pts_cam_a,
      const std::vector<Eigen::Vector3d> &pts_cam_b,
      const std::vector<int> &correspondences,
      Eigen::Vector3d &xi)
  {
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.5);

    for (size_t i = 0; i < correspondences.size(); ++i)
    {
      ceres::CostFunction *cost_function =
          new ceres::AutoDiffCostFunction<RegistrationResidual, 3, 3>(
              new RegistrationResidual(pts_cam_a[i], pts_cam_b[correspondences[i]]));

      problem.AddResidualBlock(cost_function, loss_function, xi.data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 2;
    options.num_threads = 8;
    options.minimizer_progress_to_stdout = false;
    options.function_tolerance = 1e-6;
    options.gradient_tolerance = 1e-10;
    options.parameter_tolerance = 1e-8;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    double final_loss;

    problem.Evaluate(ceres::Problem::EvaluateOptions(), &final_loss, nullptr, nullptr, nullptr);

#ifdef DEBUG
    std::cout << "[" << count_ << "] "
              << "Ceres loss: " << final_loss
              << " xi: " << xi.transpose() << std::endl;
#endif
  }

} // namespace spatiotemporal_vo
