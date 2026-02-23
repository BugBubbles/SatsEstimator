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
  DataLoader data_loader(config);

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
  int frame_count = 0;
  double t_start = 0;

  while (data_loader.hasNext())
  {
    EventPacket events = data_loader.getEventsInWindow(
        config.data.time_window,
        config.data.event_count, 
        config.data.unit_count,
        config.data.adaptive_split);

    data_loader.undistortEvents(events, config.camera);

    data_loader.projectEventsTo3D(events, config.camera);

#ifdef DEBUG
    std::cout << "[DEBUG] events number is " << events.size() << std::endl;
#endif
    if (events.empty())
      continue;

    std::cout << "\rFrame " << frame_count << ": " << events.size() << " events" << std::endl;

    // 执行配准
    auto result = registration.run<3>(events.xy, events.t, events.polarity, events.depth, "rotation");

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