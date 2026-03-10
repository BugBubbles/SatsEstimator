#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>

namespace spatiotemporal_vo
{

  // 位姿类型定义 (Sophus)
  using Vec2 = Eigen::Vector2d;
  using Vec3 = Eigen::Vector3d;
  using Vec4 = Eigen::Vector4d;
  using Mat3 = Eigen::Matrix3d;
  using EigenMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

  struct RegistrationStats
  {
    double loss = 0.0;
    double time_taken = 0.0;
    int num_iterations = 0;
  };

  template<int N>
  struct RegistrationOutput
  {
    Eigen::Matrix<double, N, 1> params = Eigen::Matrix<double, N, 1>::Zero();
    RegistrationStats stats;
    std::pair<double, double> timespan = {0.0, 0.0};
  };
  
    /**
   * @brief 事件数据结构
   */
  struct EventPacket
  {
    std::vector<Vec2> xy; // 事件坐标
    std::vector<double> t;           // 时间戳
    std::vector<int> polarity;       // 极性 (1 或 -1)
    std::vector<double> depth;       // 深度值 (可选)
    std::vector<Vec3> coord_3d;     // 3D坐标
    double t_start, t_end;        // 时间范围

    int size() const { return xy.size(); }
    bool empty() const { return xy.empty(); }
    void set_timespan() {t_start = t.front(); t_end = t.back();}
    void clear()
    {
      xy.clear();
      t.clear();
      polarity.clear();
      coord_3d.clear();
      depth.clear();
      t_start = t_end = 0.0;
    }
    void resize(size_t n)
    {
      xy.resize(n);
      t.resize(n);
      polarity.resize(n);
      coord_3d.resize(n);
      depth.resize(n);
    }
    EventPacket operator+=(const EventPacket &other)
    {
      xy.insert(xy.end(), other.xy.begin(), other.xy.end());
      t.insert(t.end(), other.t.begin(), other.t.end());
      polarity.insert(polarity.end(), other.polarity.begin(), other.polarity.end());
      coord_3d.insert(coord_3d.end(), other.coord_3d.begin(), other.coord_3d.end());
      depth.insert(depth.end(), other.depth.begin(), other.depth.end());
      t_end = other.t_end;
      return *this;
    }
    EventPacket &operator=(const EventPacket &other)
    {
      xy = other.xy;
      t = other.t;
      polarity = other.polarity;
      coord_3d = other.coord_3d;
      depth = other.depth;
      t_start = other.t_start;
      t_end = other.t_end;
      return *this;
    }
  };
} // namespace spatiotemporal_vo
