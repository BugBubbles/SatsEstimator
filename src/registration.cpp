#include "spatiotemporal_vo/registration.hpp"
#include "spatiotemporal_vo/common.hpp"
#include <ceres/ceres.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>

namespace spatiotemporal_vo
{
  template <typename T>
  Eigen::Matrix<T, 3, 3> skewSymmetric(const Eigen::Matrix<T, 3, 1> &vec)
  {
    Eigen::Matrix<T, 3, 3> S;
    S << T(0), -vec.z(), vec.y(),
        vec.z(), T(0), -vec.x(),
        -vec.y(), vec.x(), T(0);
    return S;
  }
  SpatioTemporalRegistration::SpatioTemporalRegistration(
      double fx, double fy, double cx, double cy, const Config &cfg)
      : fx_(fx), fy_(fy), cx_(cx), cy_(cy), epsilon_T_(cfg.solver.epsilon_T), max_iters_(cfg.solver.max_iters), lr_(cfg.solver.lr), trim_ratio_(cfg.solver.trim_ratio), convergence_thresh_(cfg.solver.convergence_thresh)
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

    // 4. 反投影到 3D
    std::vector<Vec3> pts_cam_a = backprojectPoints(Ia, depth_a);
    std::vector<Vec3> pts_cam_b = backprojectPoints(Ib, depth_b);

    // 5. 构建时空邻域 KDTree
    auto neighborhoods = buildTemporalNeighborhoods(ts_a, ts_b, pts_cam_b, dt);

#ifdef DEBUG
    std::cout << "[DEBUG] The neighborhoods is finished." << std::endl;
#endif
    // 6. 优化循环
    Vec3 xi = Vec3::Zero();
    // double best_loss = std::numeric_limits<double>::max();
    Vec3 last_xi, best_xi, delta_xi;
    last_xi = xi;
    best_xi = xi;
    double final_loss = 0.0, best_loss = 100000000;
    // int total_iterations = 0;
    int iter = 0;
    for (; iter < max_iters_; ++iter)
    {
      SO3 R = SO3::exp(xi);

      // 寻找匹配点
      std::vector<std::pair<int, int>> matches;

      for (size_t i = 0; i < pts_cam_a.size(); ++i)
      {
        Vec3 p_w_a = R * pts_cam_a[i];
        if (neighborhoods[i].points_3d.empty())
          continue;

        pcl::PointXYZ search_point;
        search_point.x = static_cast<float>(p_w_a.x());
        search_point.y = static_cast<float>(p_w_a.y());
        search_point.z = static_cast<float>(p_w_a.z());

        std::vector<int> pointIdxNKNSearch(1);
        std::vector<float> pointNKNSquaredDistance(1);

        if (neighborhoods[i].kdtree.nearestKSearch(search_point, 1,
                                                   pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
          matches.emplace_back(static_cast<int>(i), pointIdxNKNSearch[0]);
        }
      }

      if (matches.empty())
      {
        std::cerr << "[Registration] No valid matches found at iteration " << iter << std::endl;
        break;
      }
      // 根据匹配点对重新排序
      std::vector<Vec3> sorted_pts_cam_a;
      std::vector<Vec3> sorted_pts_cam_b;

      sorted_pts_cam_a.reserve(matches.size());
      sorted_pts_cam_b.reserve(matches.size());
      // 对残差排序，仅取前trim_ratio_ * matches.size() 个
      int trim_num = static_cast<int>(trim_ratio_ * matches.size());
      // 创建一个包含残差和匹配点对的向量
      std::vector<std::tuple<double, Vec3, Vec3>> residual_matches;
      residual_matches.reserve(matches.size());
      
      // std::ofstream events_file("./" + std::to_string(iter) + "temp.txt");
      for (size_t i = 0; i < matches.size(); ++i)
      {
        int idx_a = matches[i].first;
        int idx_b = matches[i].second;

        // 计算残差
        Vec3 p_a_w = R * pts_cam_a[idx_a];
        Vec3 residual = p_a_w - pts_cam_b[idx_b];
        double residual_norm = residual.norm();

        // 存储残差和匹配点对
          residual_matches.emplace_back(residual_norm, pts_cam_a[idx_a], pts_cam_b[idx_b]);

      }

      //按残差值升序排序
      std::sort(residual_matches.begin(), residual_matches.end(),
                [](const std::tuple<double, Vec3, Vec3> &a, const std::tuple<double, Vec3, Vec3> &b)
                {
                  return std::get<0>(a) < std::get<0>(b);
                });
      // // 截断到 trim_num 个匹配点对
      if (residual_matches.size() > trim_num)
        residual_matches.resize(trim_num);

      // 更新排序后的点
      sorted_pts_cam_a.clear();
      sorted_pts_cam_b.clear();
      for (const auto &match : residual_matches)
      {
        sorted_pts_cam_a.push_back(std::get<1>(match));
        sorted_pts_cam_b.push_back(std::get<2>(match));
          //         #ifdef DEBUG
          // events_file << std::get<0>(match) << " " << std::get<1>(match).transpose() << " " << std::get<2>(match).transpose() << std::endl;
          // #endif
      }
      // 内层 Gauss-Newton 优化
      optimizeWithCeres(sorted_pts_cam_a, sorted_pts_cam_b, xi, final_loss);
      delta_xi = xi - last_xi;
      last_xi = xi;
      // 检查收敛
      // if (delta_xi.norm() < convergence_thresh_)
      // {
      //   break;
      // }
      if (best_loss > final_loss)
      {
        best_loss = final_loss;
        best_xi = xi;
      }
    }

    output.twist = best_xi / dt * 2;
    output.stats.loss = best_loss;
    output.stats.num_iterations = count_;
    std::cout << "[" << count_ << "] " << "Final loss: " << best_loss << " Optimized xi: " << best_xi.transpose() << std::endl;
    count_++;
    return output;
  }

  struct RegistrationResidual
  {
    RegistrationResidual(const Eigen::Vector3d &p_a, const Eigen::Vector3d &p_b)
        : p_a_(p_a), p_b_(p_b) {}

    template <typename T>
    bool operator()(const T *const xi, T *residual) const
    {
      Eigen::Matrix<T, 3, 1> xi_vec(xi[0], xi[1], xi[2]);
      // 使用 Sophus::SO3 进行旋转
      Sophus::SO3<T> R = Sophus::SO3<T>::exp(xi_vec);
      // Sophus::SO3<T> R = Sophus::SO3<T>::exp(skewSymmetric(xi_vec));

      // 计算残差
      Eigen::Matrix<T, 3, 1> p_a_w = R * p_a_.cast<T>();
      Eigen::Matrix<T, 3, 1> res = p_a_w - p_b_.cast<T>();

      residual[0] = res[0];
      residual[1] = res[1];
      residual[2] = res[2];

      return true;
    }

    const Eigen::Vector3d p_a_;
    const Eigen::Vector3d p_b_;
  };

  void SpatioTemporalRegistration::optimizeWithCeres(
      const std::vector<Eigen::Vector3d> &pts_cam_a,
      const std::vector<Eigen::Vector3d> &pts_cam_b,
      Eigen::Vector3d &xi, double &final_loss)
  {
    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0); 

    // 添加残差块
    for (size_t i = 0; i < pts_cam_a.size(); ++i)
    {
      ceres::CostFunction *cost_function =
          new ceres::AutoDiffCostFunction<RegistrationResidual, 3, 3>(
              new RegistrationResidual(pts_cam_a[i], pts_cam_b[i]));

      problem.AddResidualBlock(cost_function, loss_function, xi.data());
    }

    // 配置求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 5;
    options.num_threads = 8;
    options.minimizer_progress_to_stdout = false;

    options.function_tolerance = 1e-6; // 函数值收敛阈值
    options.gradient_tolerance = 1e-10; // 梯度收敛阈值
    options.parameter_tolerance = 1e-8; // 参数收敛阈值
    options.max_solver_time_in_seconds = 60.0; // 最大求解时间
    // 求解
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 计算最终损失
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &final_loss, nullptr, nullptr, nullptr);
#ifdef DEBUG
    std::cout << "[" << count_ << "] " << "Final loss: " << final_loss << " Optimized xi: " << xi.transpose() << std::endl;
#endif
  }

} // namespace spatiotemporal_vo
