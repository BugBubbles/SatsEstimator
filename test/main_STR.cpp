#include <iostream>
#include <filesystem>
#include "spatiotemporal_vo/config.hpp"
#include "spatiotemporal_vo/data_loader.hpp"
#include "methods/icp_estimator.hpp"
#include "spatiotemporal_vo/logger.hpp"

using namespace spatiotemporal_vo;

int main(int argc, char **argv)
{
  std::cout << "=== ICP Visual Odometry (C++) ===" << std::endl;

  // 1. 加载配置
  std::string config_file = (argc > 1) ? argv[1] : "../configs/default.yaml";
  Config config(config_file);
  config.print();

  // 2. 初始化数据加载器
  DataLoader data_loader;

  std::string event_file = config.paths.data_root + "/" + config.paths.sequences + "/" + config.paths.event_file;
  if (!data_loader.loadEvents(event_file))
  {
    std::cerr << "Failed to load events from: " << event_file << std::endl;
    std::cerr << "Exiting." << std::endl;
    return -1;
  }

  // 3. 初始化配准模块
  SpatioTemporalRegistration registration(
      config.camera.fx,
      config.camera.fy,
      config.camera.cx,
      config.camera.cy,
      config);

  // 4. 初始化可视化和轨迹记录
  ResultLogger logger(config.paths.output_root, "str", config.paths.suffix);

  // 5. 主循环
  SE3 current_pose; // 初始为单位矩阵
  int frame_count = 0;
  double t_start = 0;

  while (data_loader.hasNext())
  {
    // 获取一批事件
    EventPacket events = data_loader.getEventsInWindow(t_start, t_start + config.data.time_window,
                                                       config.data.event_count);
    // 去除畸变
    data_loader.undistortEvents(events, config.camera);

#ifdef DEBUG
    // std::ofstream undistorted_events_file(config.paths.output_root + "/temp.txt");
    // // 将去畸变后的事件点写入文件
    // for (size_t i = 0; i < events.size(); ++i)
    // {
    //   undistorted_events_file << events.xy[i].x() << " " << events.xy[i].y() << " "
    //                           << events.t[i] << " " << events.polarity[i] << "\n";
    // }
    // undistorted_events_file.flush();
    std::cout << "[DEBUG] events number is " << events.size() << std::endl;
#endif
    if (events.empty())
      continue;

    std::cout << "\rFrame " << frame_count << ": " << events.size() << " events" << std::endl;

    // 执行配准
    std::vector<double> dummy_depth; // 空深度，使用默认值 1.0
    RegistrationOutput<3> result = registration.run(
        events.xy,
        events.t,
        events.polarity,
        dummy_depth);

    frame_count++;
    t_start += (events.t_end - events.t_start);
    // 6. 保存结果

    if (logger.is_initialized())
    {
      logger.log<3>(result);
    }
  }

  std::cout << "\n\nProcessed " << frame_count << " frames." << std::endl;
  return 0;
}