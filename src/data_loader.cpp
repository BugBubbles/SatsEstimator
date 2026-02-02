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

} // namespace spatiotemporal_vo