#include "spatiotemporal_vo/data_loader.hpp"
#include <sstream>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <numeric>

namespace spatiotemporal_vo
{

  DataLoader::DataLoader()

  {
    if (all_events_.is_open())
    {
      all_events_.close();
    }
    events_buffer_.clear();
  }

  DataLoader::DataLoader(Config &config)
  {
    std::string filepath = config.paths.data_root + "/" +
                           config.paths.sequences + "/" +
                           config.paths.event_file;
    all_events_.open(filepath); // Open the file directly with all_events_

    events_buffer_.clear();
    grid_size_ = config.data.grid_size;
    grid_w_ = config.camera.width / grid_size_;
    // 初始化 t_ppf_ 用于后续计算， 其中 alpha =0.9, 自由度从2开始取，表格中第一个位置对应自由度为1
    t_ppf_ = {3.0776835372078066, 1.8856180831641507, 1.6377443536962095, 1.5332062740589432, 1.4758840488558216, 1.4397557472577693, 1.4149239276488585, 1.396815309743419, 1.3830287383964925, 1.3721836411102863, 1.3634303180205214, 1.3562173340231976, 1.3501712887800512, 1.345030374454649, 1.3406056078504547, 1.3367571673273142, 1.3333793897216268, 1.330390943569909};
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

  EventPacket DataLoader::getEventsInWindow(int event_count, bool use_depth)
  {
    events_buffer_.clear();
    std::string line;

    if (use_depth)
    {
      while (std::getline(all_events_, line) && events_buffer_.size() < event_count)
      {
        if (line.empty() || line[0] == '#' || line[0] == '%')
          continue;

        std::istringstream iss(line);
        double t, x, y, d;
        int p;

        // 假设格式为: t x y p
        if (iss >> t >> x >> y >> p >> d)
        {
          events_buffer_.t.push_back(t);
          events_buffer_.xy.push_back(Eigen::Vector2d(x, y));
          events_buffer_.depth.push_back(d);
          events_buffer_.polarity.push_back(p > 0 ? 1 : -1);
        }
      }
    }
    else
    {
      while (std::getline(all_events_, line) && events_buffer_.size() < event_count)
      {
        if (line.empty() || line[0] == '#' || line[0] == '%')
          continue;

        std::istringstream iss(line);
        double t, x, y;
        int p;

        // 假设格式为: t x y p
        if (iss >> t >> x >> y >> p)
        {
          events_buffer_.t.push_back(t);
          events_buffer_.xy.push_back(Eigen::Vector2d(x, y));
          events_buffer_.polarity.push_back(p > 0 ? 1 : -1);
        }
      }
    }
    events_buffer_.set_timespan();
    return events_buffer_;
  }

  EventPacket DataLoader::getEventsInWindow(double min_timespan, int event_count, int unit_count, bool adaptive_split, bool use_depth)
  {
    std::string line;

    if (!adaptive_split)
    { // 此时只需要保证划分事件个数都取到最大的event_count即可，不需要考虑时间跨度
      events_buffer_.clear();
      while (std::getline(all_events_, line) && events_buffer_.size() < event_count)
      {
        if (line.empty() || line[0] == '#' || line[0] == '%')
          continue;
        std::istringstream iss(line);
        double t, x, y, d;
        int p;
        // 假设格式为: t x y p
        if (use_depth)
        {
          if (iss >> t >> x >> y >> p >> d)
          {
            events_buffer_.t.push_back(t);
            events_buffer_.xy.push_back(Eigen::Vector2d(x, y));
            events_buffer_.depth.push_back(d);
            events_buffer_.polarity.push_back(p > 0 ? 1 : -1);
          }
        }
        else
        {
          if (iss >> t >> x >> y >> p)
          {
            events_buffer_.t.push_back(t);
            events_buffer_.xy.push_back(Eigen::Vector2d(x, y));
            events_buffer_.polarity.push_back(p > 0 ? 1 : -1);
          }
        }
      }
      events_buffer_.set_timespan();
      return events_buffer_;
    }
    else
    {
      // 自适应划分，每次获取 unit_count 个事件，计算由这些事件构成的状态量
      // 保证时间跨度不少于 min_delta_time，且事件数在[min_count * unit_cout, event_count]范围内
      // 当状态量的变化超过粗大误差的t分布置信区间时，停止采样
      EventPacket packet;
      while (hasNext())
      {
        packet.clear();
        // 采集一个数量为 unit_count 的初始包
        while (std::getline(all_events_, line) && packet.size() < unit_count)
        {
          if (line.empty() || line[0] == '#' || line[0] == '%')
            continue;
          std::istringstream iss(line);
          double t, x, y, d;
          int p;
          // 假设格式为: t x y p
          if (use_depth)
          {
            if (iss >> t >> x >> y >> p >> d)
            {
              packet.t.push_back(t);
              packet.xy.push_back(Eigen::Vector2d(x, y));
              packet.depth.push_back(d);
              packet.polarity.push_back(p > 0 ? 1 : -1);
            }
          }
          else
          {
            if (iss >> t >> x >> y >> p)
            {
              packet.t.push_back(t);
              packet.xy.push_back(Eigen::Vector2d(x, y));
              packet.polarity.push_back(p > 0 ? 1 : -1);
            }
          }
        }
        packet.set_timespan();
        // 计算初始包的状态量
        double metric = compute_grid_metric(packet);
        if (score_.size() <= 2)
        {
          events_buffer_ += packet;
          score_.push_back(metric);
          score_var_.push_back(metric * metric);
        }
        else
        {
          // 计算均值和方差
          double num_scores = static_cast<double>(score_.size());
          double mean = std::accumulate(score_.begin(), score_.end(), 0.0) / num_scores;
          double std = std::sqrt(std::accumulate(score_var_.begin(), score_var_.end(), 0.0) / num_scores - mean * mean);
          // 计算当前自由度下的t分布置信区间
          double t_threshold = t_ppf_[std::min(score_.size() - 2, t_ppf_.size() - 1)];
          if ((std::abs(metric - mean) < t_threshold * std * min_timespan / (packet.t_end - packet.t_start) && score_.size() < event_count / unit_count) || (events_buffer_.t_end - events_buffer_.t_start < min_timespan))
          {
            events_buffer_ += packet;
            score_.push_back(metric);
            score_var_.push_back(metric * metric);
          }
          else
          {
            std::swap(packet, events_buffer_);
            score_ = {metric};
            score_var_ = {metric * metric};
            break;
          }
        }
      }
      events_buffer_.set_timespan();
      packet.set_timespan();
      return packet;
    }
  }

  double DataLoader::compute_grid_metric(const EventPacket &packet)
  {
    // 计算空间归一化速率
    if (packet.empty() || packet.t_end - packet.t_start <= 1e-7)
      return 0.0;

    std::unordered_set<int> unique_indices;
    for (const auto &xy : packet.xy)
    {
      int xs = static_cast<int>(xy.x());
      int ys = static_cast<int>(xy.y());
      int g_xs = xs / grid_size_;
      int g_ys = ys / grid_size_;
      int flat_index = g_ys * grid_w_ + g_xs;
      unique_indices.insert(flat_index);
    }

    double dt = packet.t_end - packet.t_start;
    return static_cast<double>(unique_indices.size()) * grid_size_ * grid_size_ / dt;
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

  void DataLoader::projectEventsTo3D(EventPacket &packet, const CameraConfig &cam_cfg)
  {
    int n = packet.size();
    packet.coord_3d.resize(n);
    bool depth_provided = !packet.depth.empty();

    double fx = cam_cfg.fx;
    double fy = cam_cfg.fy;
    double cx = cam_cfg.cx;
    double cy = cam_cfg.cy;

    for (int i = 0; i < n; ++i)
    {
      double depth = depth_provided ? packet.depth[i] : 1.0; // 如果没有深度信息，假设深度为1.0
      double x = (packet.xy[i](0) - cx) / fx * depth;
      double y = (packet.xy[i](1) - cy) / fy * depth;
      packet.coord_3d[i] = Vec3(x, y, depth);
    }
  }

  void DataLoader::computeReferenceTime(EventPacket &packet, std::string method)
  {
    if (packet.empty())
      return;
    double t_ref = 0.0;
    if (method == "front")
      t_ref = packet.t.front();
    else if (method == "middle")
      t_ref = 0.5 * (packet.t.front() + packet.t.back());
    else if (method == "mean")
      t_ref = std::accumulate(packet.t.begin(), packet.t.end(), 0.0) / packet.size();
    for (auto &t : packet.t)
    {
      t -= t_ref;
    }
  }

  bool DataLoader::hasNext()
  {
    return all_events_.good() && (all_events_.peek() != EOF);
  }

} // namespace spatiotemporal_vo