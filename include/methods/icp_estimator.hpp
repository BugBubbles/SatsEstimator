#pragma once

#include "spatiotemporal_vo/common.hpp"
#include "spatiotemporal_vo/config.hpp"
#include <vector>
#include <Eigen/Core>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>

namespace spatiotemporal_vo
{

  class SpatioTemporalRegistration
  {
    using Mat4 = Eigen::Matrix4d;

  public:
    SpatioTemporalRegistration(
        double fx, double fy, double cx, double cy, const Config &cfg);
    template <int N>
    RegistrationOutput<N> run(
        const std::vector<Vec2> &events_xy,
        const std::vector<double> &events_t,
        const std::vector<int> &polarity,
        const std::vector<double> &depth_in = {},
        const std::string &mode = "rotation")
    {
      assert(((N == 3 && mode == "rotation") || (N == 3 && mode == "translation") || (N == 6 && mode == "6dof")) && "Mode and output dimension mismatch");
      RegistrationOutput<N> output;

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
      std::vector<double> ts_a, ts_b, depth_a, depth_b;
      Ia.reserve(idx_a.size());
      Ib.reserve(idx_b.size());
      ts_a.reserve(idx_a.size());
      ts_b.reserve(idx_b.size());
      depth_a.reserve(idx_a.size());
      depth_b.reserve(idx_b.size());

      for (int i : idx_a)
      {
        Ia.push_back(events_xy[i]);
        ts_a.push_back(events_t[i]);
        if (!depth_in.empty())
        {
          depth_a.push_back(depth_in[i]);
        }
        else
        {
          depth_a.push_back(1.0); // 默认深度
        }
      }
      for (int i : idx_b)
      {
        Ib.push_back(events_xy[i]);
        ts_b.push_back(events_t[i]);
        if (!depth_in.empty())
        {
          depth_b.push_back(depth_in[i]);
        }
        else
        {
          depth_b.push_back(1.0); // 默认深度
        }
      }

#ifdef DEBUG
      std::cout << "[DEBUG] The shape of Ia is " << Ia.size() << std::endl;
      std::cout << "[DEBUG] The shape of Ib is " << Ib.size() << std::endl;
#endif

      // 4. 反投影到 3D（归一化到单位球面）
      std::vector<Vec3> pts_cam_a = backprojectPoints(Ia, depth_a);
      std::vector<Vec3> pts_cam_b = backprojectPoints(Ib, depth_b);

      // 5. 构建目标点云的全局 KDTree
      auto kdtree = buildTemporalNeighborhoods(ts_a, ts_b, pts_cam_b, dt);

#ifdef DEBUG
      std::cout << "[DEBUG] The neighborhoods is finished." << std::endl;
#endif

      Mat4 best_pose = Mat4::Identity();
      Mat4 curr_pose = Mat4::Identity();
      double best_loss = std::numeric_limits<double>::max();

      // 外层循环：多次重启以避免局部最优
      int num_restarts = 2; // 可以增加重启次数以提高鲁棒性
      output.stats.time_taken = omp_get_wtime();

      for (int restart = 0; restart < num_restarts; ++restart)
      {
        // 如果是重启，添加随机扰动（可选）
        if (restart > 0)
        {
          // 添加小的随机旋转作为初始值
          Vec3 random_axis = Vec3::Random().normalized();
          double random_angle = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 0.2; // [-0.1, 0.1] rad
          Mat3 R_current = Eigen::AngleAxisd(random_angle, random_axis).toRotationMatrix();
          // 零初始位移
          Vec3 curr_trans = Vec3::Zero();
          // 合成为初始变换
          curr_pose = Eigen::Matrix4d::Identity();
          curr_pose.block<3, 3>(0, 0) = R_current;
          curr_pose.block<3, 1>(0, 3) = curr_trans;
        }

        // 运行 Trimmed ICP
        double loss = 0.0;
        if constexpr (N == 3)
        {
          if (mode == "rotation")
          {
            loss = rotationICP(pts_cam_a, pts_cam_b, kdtree, curr_pose,
                               max_iters_, trim_ratio_);
          }
          else if (mode == "translation")
          {
            loss = translationICP(pts_cam_a, pts_cam_b, kdtree, curr_pose,
                                  max_iters_, trim_ratio_);
          }
        }
        else if constexpr (N == 6)
        {
          if (mode == "6dof")
          {
            loss = sixDofICP(pts_cam_a, pts_cam_b, kdtree, curr_pose,
                             max_iters_, trim_ratio_);
          }
        }

        if (loss < best_loss)
        {
          best_loss = loss;
          best_pose = curr_pose;
        }
      }
      output.stats.time_taken = omp_get_wtime() - output.stats.time_taken;

      // 从位姿矩阵中提取旋转和平移部分
      // 用Eigen的AngleAxis来提取旋转轴和旋转角度
      Mat3 R_final = best_pose.block<3, 3>(0, 0);
      Eigen::AngleAxisd angle_axis(R_final);
      Vec3 rot_result = angle_axis.angle() * angle_axis.axis();
      // 平移部分
      Vec3 trans_result = best_pose.block<3, 1>(0, 3);

      // 8. 输出结果
      if constexpr (N == 3)
      {
        if (mode == "rotation")
        {
          output.params = -rot_result / dt * 2.0; // 乘以 2 是因为我们只使用了半个时间窗口的数据
        }
        else if (mode == "translation")
        {
          output.params = -trans_result / dt * 2.0;
        }
      }
      else if (N == 6)
      {
        if (mode == "6dof")
        {
          output.params.template head<3>() = -rot_result / dt * 2.0;
          output.params.template tail<3>() = -trans_result / dt * 2.0;
        }
      }
      output.stats.loss = best_loss;
      output.stats.num_iterations = count_;

      std::cout << "[" << count_ << "] " << "<< STR >>"
                << "Final loss: " << best_loss
                << " Optimized xi: " << output.params.transpose() << std::endl;
      std::cout << '[' << count_ << "] " << "<< STR >>" << ": Optimization took " << output.stats.time_taken << " seconds." << std::endl;
      count_++;
      return output;
    }

    int count_ = 1;

  private:
    // 时空邻域结构
    struct TemporalNeighborhood
    {
      pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
      std::vector<int> original_indices;
      std::vector<Vec3> points_3d;
    };
    double rotationICP(
        const std::vector<Vec3> &data_points,
        const std::vector<Vec3> &model_points,
        std::vector<TemporalNeighborhood> &kdtree,
        Mat4 &pose,
        int max_iterations,
        double trim_fraction);
    double translationICP(
        const std::vector<Vec3> &data_points,
        const std::vector<Vec3> &model_points,
        std::vector<TemporalNeighborhood> &kdtree,
        Mat4 &pose,
        int max_iterations,
        double trim_fraction);
    double sixDofICP(
        const std::vector<Vec3> &data_points,
        const std::vector<Vec3> &model_points,
        std::vector<TemporalNeighborhood> &kdtree,
        Mat4 &pose,
        int max_iterations,
        double trim_fraction);

    Vec2 projectPoint(const Vec3 &p);
    Vec3 backprojectPoint(const Vec2 &uv, double d);
    std::vector<Vec3> backprojectPoints(
        const std::vector<Vec2> &uvs,
        const std::vector<double> &depths);
    std::vector<TemporalNeighborhood> buildTemporalNeighborhoods(
        const std::vector<double> &ts_a,
        const std::vector<double> &ts_b,
        const std::vector<Vec3> &pts_cam_b,
        double dt);

    // 相机内参
    double fx_, fy_, cx_, cy_;

    // 配置参数
    double epsilon_T_;
    int max_iters_;
    double trim_ratio_;
    double convergence_thresh_;
  };

} // namespace spatiotemporal_vo
