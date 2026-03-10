#include "methods/icp_estimator.hpp"
#include "spatiotemporal_vo/common.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <ceres/ceres.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace spatiotemporal_vo
{
  double SpatioTemporalRegistration::rotationICP(
      const std::vector<Vec3> &data_points,             // 源点云（待配准）
      const std::vector<Vec3> &model_points,            // 目标点云（模型）
      std::vector<TemporalNeighborhood> &neighborhoods, // 目标点云的 KDTree
      Mat4 &pose,                                       // 输出：最优位姿变换
      int max_iterations,                               // 最大迭代次数
      double trim_fraction)                             // 截断比例
  {
    int num_data = static_cast<int>(data_points.size());
    int num_model = static_cast<int>(model_points.size());

    if (num_data == 0 || num_model == 0)
    {
      pose = Mat4::Identity();
      return std::numeric_limits<double>::max();
    }

    // 计算要保留的点数（Trimmed ICP）
    int num_trimmed = static_cast<int>(trim_fraction * num_data);
    if (num_trimmed < 10)
      num_trimmed = std::min(10, num_data);

    // 初始化
    Mat3 R_current = pose.block<3, 3>(0, 0);
    double prev_error = std::numeric_limits<double>::max();
    double current_error = 0.0;
    double best_error = 100000000;
    Mat3 best_R = R_current;

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
      if (best_error > current_error)
      {
        best_error = current_error;
        best_R = R_current;
      }
      prev_error = current_error;

      // ========== Step 4: 使用 SVD 计算最优旋转 ==========
      // 准备去质心的点对
      std::vector<Vec3> source;
      std::vector<Vec3> target;
      std::vector<int> valid_correspondences;

      int valid_count = 0;

      // 只使用截断后的点计算质心
      for (int i = 0; i < num_trimmed; ++i)
      {
        int idx = sorted_indices[i];
        if (correspondences[idx] >= 0)
        {
          valid_count++;
        }
      }

      if (valid_count < 3)
      {
        break; // 点太少，无法计算旋转
      }

      // 去质心
      source.reserve(valid_count);
      target.reserve(valid_count);
      valid_correspondences.reserve(valid_count);

      int corr_idx = 0;
      for (int i = 0; i < num_trimmed; ++i)
      {
        int idx = sorted_indices[i];
        if (correspondences[idx] >= 0)
        {
          Vec3 src_pt = R_current * data_points[idx];
          Vec3 tgt_pt = model_points[correspondences[idx]];
          source.push_back(src_pt);
          target.push_back(tgt_pt);
          valid_correspondences.push_back(corr_idx);
          corr_idx++;
        }
      }

      // 计算增量旋转
      Mat3 H = Mat3::Zero();
      for (size_t i = 0; i < valid_correspondences.size(); ++i)
      {
        if (valid_correspondences[i] >= 0)
        {
          H += source[i] * target[valid_correspondences[i]].transpose();
        }
      }

      // SVD 分解
      Eigen::JacobiSVD<Mat3> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
      Mat3 U = svd.matrixU();
      Mat3 V = svd.matrixV();

      // 计算旋转矩阵 R = V * U^T
      Mat3 delta_R = V * U.transpose();

      // 确保 R 是一个有效的旋转矩阵（行列式为 1）
      if (delta_R.determinant() < 0)
      {
        V.col(2) *= -1;
        delta_R = V * U.transpose();
      }
      // 更新累积旋转
      R_current = delta_R * R_current;

      // 确保旋转矩阵的正交性（防止数值漂移）
      svd.compute(R_current, Eigen::ComputeFullU | Eigen::ComputeFullV);
      R_current = svd.matrixU() * svd.matrixV().transpose();
    }

    pose.block<3, 3>(0, 0) = best_R;
    return current_error;
  }

  double SpatioTemporalRegistration::translationICP(
      const std::vector<Vec3> &data_points,             // 源点云（待配准）
      const std::vector<Vec3> &model_points,            // 目标点云（模型）
      std::vector<TemporalNeighborhood> &neighborhoods, // 目标点云的 KDTree
      Mat4 &pose,                                       // 输出：最优位姿变换
      int max_iterations,                               // 最大迭代次数
      double trim_fraction)                             // 截断比例
  {
    int num_data = static_cast<int>(data_points.size());
    int num_model = static_cast<int>(model_points.size());

    if (num_data == 0 || num_model == 0)
    {
      pose = Mat4::Identity();
      return std::numeric_limits<double>::max();
    }

    // 计算要保留的点数（Trimmed ICP）
    int num_trimmed = static_cast<int>(trim_fraction * num_data);
    if (num_trimmed < 10)
      num_trimmed = std::min(10, num_data);

    // 初始化
    Vec3 curr_trans = pose.block<3, 1>(0, 3);
    double prev_error = std::numeric_limits<double>::max();
    double current_error = 0.0;
    double best_error = 100000000;
    Vec3 best_trans = curr_trans;

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
        Vec3 transformed_point = data_points[i] + curr_trans;

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
      if (best_error > current_error)
      {
        best_error = current_error;
        best_trans = curr_trans;
      }
      prev_error = current_error;

      // ========== Step 4: 使用 SVD 计算最优旋转 ==========
      // 准备去质心的点对
      std::vector<Vec3> source;
      std::vector<Vec3> target;
      std::vector<int> valid_correspondences;

      int valid_count = 0;

      // 只使用截断后的点计算质心
      for (int i = 0; i < num_trimmed; ++i)
      {
        int idx = sorted_indices[i];
        if (correspondences[idx] >= 0)
        {
          valid_count++;
        }
      }

      if (valid_count < 3)
      {
        break; // 点太少，无法计算旋转
      }

      // 去质心
      source.reserve(valid_count);
      target.reserve(valid_count);
      valid_correspondences.reserve(valid_count);

      int corr_idx = 0;
      for (int i = 0; i < num_trimmed; ++i)
      {
        int idx = sorted_indices[i];
        if (correspondences[idx] >= 0)
        {
          Vec3 src_pt = data_points[idx] + curr_trans;
          Vec3 tgt_pt = model_points[correspondences[idx]];
          source.push_back(src_pt);
          target.push_back(tgt_pt);
          valid_correspondences.push_back(corr_idx);
          corr_idx++;
        }
      }

      // 计算增量平移
      Vec3 delta_trans = Vec3::Zero();
      for (size_t i = 0; i < valid_correspondences.size(); ++i)
      {
        if (valid_correspondences[i] >= 0)
        {
          delta_trans += target[valid_correspondences[i]] - source[i];
        }
      }
      delta_trans /= static_cast<double>(valid_correspondences.size());

      // 更新累积旋转
      curr_trans = delta_trans;
    }
    pose.block<3, 1>(0, 3) = best_trans;
    return current_error;
  }

  double SpatioTemporalRegistration::sixDofICP(
      const std::vector<Vec3> &data_points,             // 源点云（待配准）
      const std::vector<Vec3> &model_points,            // 目标点云（模型）
      std::vector<TemporalNeighborhood> &neighborhoods, // 目标点云的 KDTree
      Mat4 &pose,                                       // 输出：最优位姿变换
      int max_iterations,                               // 最大迭代次数
      double trim_fraction)                             // 截断比例
  {
    int num_data = static_cast<int>(data_points.size());
    int num_model = static_cast<int>(model_points.size());

    if (num_data == 0 || num_model == 0)
    {
      pose = Mat4::Identity();
      return std::numeric_limits<double>::max();
    }

    // 计算要保留的点数（Trimmed ICP）
    int num_trimmed = static_cast<int>(trim_fraction * num_data);
    if (num_trimmed < 10)
      num_trimmed = std::min(10, num_data);

    // 初始化
    Mat3 curr_rot = pose.block<3, 3>(0, 0);
    Vec3 curr_trans = pose.block<3, 1>(0, 3);
    double prev_error = std::numeric_limits<double>::max();
    double current_error = 0.0;
    double best_error = 100000000;
    Vec3 best_trans = curr_trans;
    Mat3 best_rot = curr_rot;

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
        Vec3 transformed_point = data_points[i] + curr_trans;

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
      if (best_error > current_error)
      {
        best_error = current_error;
        best_trans = curr_trans;
        best_rot = curr_rot;
      }
      prev_error = current_error;

      // ========== Step 4: 使用 SVD 计算最优旋转 ==========
      // 准备去质心的点对
      std::vector<Vec3> source;
      std::vector<Vec3> target;
      std::vector<int> valid_correspondences;

      int valid_count = 0;

      // 只使用截断后的点计算质心
      for (int i = 0; i < num_trimmed; ++i)
      {
        int idx = sorted_indices[i];
        if (correspondences[idx] >= 0)
        {
          valid_count++;
        }
      }

      if (valid_count < 3)
      {
        break; // 点太少，无法计算旋转
      }

      // 去质心
      source.reserve(valid_count);
      target.reserve(valid_count);
      valid_correspondences.reserve(valid_count);

      int corr_idx = 0;
      for (int i = 0; i < num_trimmed; ++i)
      {
        int idx = sorted_indices[i];
        if (correspondences[idx] >= 0)
        {
          Vec3 src_pt = curr_rot * data_points[idx] + curr_trans;
          Vec3 tgt_pt = model_points[correspondences[idx]];
          source.push_back(src_pt);
          target.push_back(tgt_pt);
          valid_correspondences.push_back(corr_idx);
          corr_idx++;
        }
      }

      // 计算增量平移
      Mat3 delta_rot = Mat3::Identity();
      Vec3 delta_trans = Vec3::Zero();

      // 计算增量旋转
      Mat3 H = Mat3::Zero();
      for (size_t i = 0; i < valid_correspondences.size(); ++i)
      {
        if (valid_correspondences[i] >= 0)
        {
          H += source[i] * target[valid_correspondences[i]].transpose();
          delta_trans += target[valid_correspondences[i]] - (delta_rot * source[i]);
        }
      }
      // SVD 分解
      Eigen::JacobiSVD<Mat3> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
      Mat3 U = svd.matrixU();
      Mat3 V = svd.matrixV();
      // 计算旋转矩阵 R = V * U^T
      delta_rot = V * U.transpose();
      // 确保 R 是一个有效的旋转矩阵（行列式为 1）
      if (delta_rot.determinant() < 0)
      {
        V.col(2) *= -1;
        delta_rot = V * U.transpose();
      }
      // 计算增量平移
      delta_trans /= static_cast<double>(valid_correspondences.size());
      // 更新累积旋转
      curr_rot = delta_rot * curr_rot;
      curr_trans += delta_trans;
    }
    pose.block<3, 1>(0, 3) = best_trans;
    pose.block<3, 3>(0, 0) = best_rot;
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
        trim_ratio_(cfg.solver.trim_ratio),
        convergence_thresh_(cfg.solver.convergence_thresh)
  {
  }

  Vec2 SpatioTemporalRegistration::projectPoint(const Vec3 &p)
  {
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
} // namespace spatiotemporal_vo
