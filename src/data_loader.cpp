#include "spatiotemporal_vo/data_loader.hpp"
#include <sstream>
#include <iostream>
#include <algorithm>

namespace spatiotemporal_vo
{

  DataLoader::DataLoader()
      : current_index_(0), current_time_(0.0)
  {
    if (all_events_.is_open())
    {
      all_events_.close();
    }
  }

  DataLoader::DataLoader(const std::string &dataset_path)
      : dataset_path_(dataset_path), current_index_(0), current_time_(0.0)
  {
    if (all_events_.is_open())
    {
      all_events_.close();
    }
  }

  bool DataLoader::loadEvents(const std::string &filepath)
  {
    all_events_.close();        // Close the stream if it was already open
    all_events_.open(filepath); // Open the file directly with all_events_

    if (!all_events_.is_open())
    {
      std::cerr << "Failed to open event file: " << filepath << std::endl;
      return false;
    }
    std::cout << "Loaded events from " << filepath << std::endl;
    return true;
  }

  bool DataLoader::loadGroundTruth(const std::string &filepath)
  {
    std::ifstream ifs(filepath);
    if (!ifs.is_open())
    {
      std::cerr << "Failed to open ground truth file: " << filepath << std::endl;
      return false;
    }

    gt_timestamps_.clear();
    gt_poses_.clear();
    std::string line;

    while (std::getline(ifs, line))
    {
      if (line.empty() || line[0] == '#')
        continue;

      std::istringstream iss(line);
      double t, tx, ty, tz, qx, qy, qz, qw;

      // TUM 格式: timestamp tx ty tz qx qy qz qw
      if (iss >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw)
      {
        gt_timestamps_.push_back(t);
        Eigen::Quaterniond q(qw, qx, qy, qz);
        Eigen::Vector3d trans(tx, ty, tz);
        gt_poses_.push_back(SE3(q, trans));
      }
    }

    ifs.close();
    std::cout << "Loaded " << gt_poses_.size() << " ground truth poses from " << filepath << std::endl;
    return !gt_poses_.empty();
  }

  EventPacket DataLoader::getEventsInWindow(double t_start, double t_end, int event_count, int select_num)
  {
    EventPacket packet;
    std::string line;

    if (t_start > t_end)
    {
      while (std::getline(all_events_, line) && packet.size() < event_count)
      {
        if (line.empty() || line[0] == '#' || line[0] == '%')
          continue;

        std::istringstream iss(line);
        double t, x, y;
        int p;

        // 假设格式为: t x y p
        if (iss >> t >> x >> y >> p)
        {
          packet.t.push_back(t);
          packet.xy.push_back(Eigen::Vector2d(x, y));
          packet.polarity.push_back(p > 0 ? 1 : -1);
        }
      }
    }
    else
    {
      while (std::getline(all_events_, line))
      {
        if (line.empty() || line[0] == '#' || line[0] == '%')
          continue;
        std::istringstream iss(line);
        double t, x, y;
        int p;

        // 假设格式为: t x y p
        if (iss >> t >> x >> y >> p)
        {
          packet.t.push_back(t);
          packet.xy.push_back(Eigen::Vector2d(x, y));
          packet.polarity.push_back(p > 0 ? 1 : -1);
        }
        if (t > t_end)
        {
          break;
        }
      }
    }
    // if (select_num > 0)
    // {
    //   if (static_cast<int>(packet.t.size()) > select_num)
    //   {
    //     EventPacket trimmed_packet;
    //     int total = packet.t.size();
    //     std::vector<int> indices(total);
    //     std::iota(indices.begin(), indices.end(), 0);

    //     // 随机打乱索引
    //     std::random_device rd;
    //     std::mt19937 gen(rd());
    //     std::shuffle(indices.begin(), indices.end(), gen);

    //     // 取前select_num个索引并排序（保证输出顺序与原始一致）
    //     indices.resize(select_num);
    //     std::sort(indices.begin(), indices.end());

    //     for (int idx : indices)
    //     {
    //       trimmed_packet.t.push_back(packet.t[idx]);
    //       trimmed_packet.xy.push_back(packet.xy[idx]);
    //       trimmed_packet.polarity.push_back(packet.polarity[idx]);
    //     }
    //     packet = std::move(trimmed_packet);
    //   }
    // }

    return packet;
  }

  void DataLoader::undistortEvents(EventPacket &packet, const CameraConfig &cam_cfg)
  {
    if (packet.empty())
      return;

    // 构造 OpenCV 畸变参数
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << cam_cfg.fx, 0, cam_cfg.cx,
                            0, cam_cfg.fy, cam_cfg.cy,
                            0, 0, 1);
    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << cam_cfg.distortion_args[0],
                          cam_cfg.distortion_args[1],
                          cam_cfg.distortion_args[2],
                          cam_cfg.distortion_args[3],
                          cam_cfg.distortion_args[4]);

    // 转换为 OpenCV 点
    std::vector<cv::Point2f> pts;
    pts.reserve(packet.xy.size());
    for (const auto &p : packet.xy)
      pts.emplace_back(p.x(), p.y());

    // 去畸变
    std::vector<cv::Point2f> undistorted_pts;
    cv::undistortPoints(pts, undistorted_pts, cameraMatrix, distCoeffs, cv::noArray(), cameraMatrix);

    // 更新原始数据
    for (size_t i = 0; i < packet.xy.size(); ++i)
      packet.xy[i] = Vec2(undistorted_pts[i].x, undistorted_pts[i].y);
  }

  bool DataLoader::hasNext()
  {
    return all_events_.good() && (all_events_.peek() != EOF);
  }

  SE3 DataLoader::getGroundTruthPose(double timestamp) const
  {
    if (gt_timestamps_.empty())
    {
      return SE3();
    }

    // 找到最近的两个时间戳进行插值
    auto it = std::lower_bound(gt_timestamps_.begin(), gt_timestamps_.end(), timestamp);

    if (it == gt_timestamps_.begin())
    {
      return gt_poses_.front();
    }
    if (it == gt_timestamps_.end())
    {
      return gt_poses_.back();
    }

    size_t idx = std::distance(gt_timestamps_.begin(), it);
    double t1 = gt_timestamps_[idx - 1];
    double t2 = gt_timestamps_[idx];
    double alpha = (timestamp - t1) / (t2 - t1);

    // 线性插值平移，球面插值旋转
    const SE3 &pose1 = gt_poses_[idx - 1];
    const SE3 &pose2 = gt_poses_[idx];

    Eigen::Vector3d trans = (1 - alpha) * pose1.translation() + alpha * pose2.translation();
    Eigen::Quaterniond q = pose1.unit_quaternion().slerp(alpha, pose2.unit_quaternion());

    return SE3(q, trans);
  }

} // namespace spatiotemporal_vo