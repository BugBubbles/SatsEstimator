#include <iostream>
#include <filesystem>
#include "spatiotemporal_vo/config.hpp"
#include "spatiotemporal_vo/data_loader.hpp"
#include "methods/poly_estimator.hpp"
#include "spatiotemporal_vo/logger.hpp"

using namespace spatiotemporal_vo;

int main(int argc, char **argv)
{
  std::cout << "=== Poly Visual Odometry (C++) ===" << std::endl;

  // 1. 加载配置
  std::string config_file = (argc > 1) ? argv[1] : "../configs/default.yaml";
  Config config(config_file);
  config.print();

  // 2. 初始化数据加载器
  DataLoader data_loader(config);

  if (!data_loader.hasNext())
  {
    std::cerr << "Failed to load events." << std::endl;
    return -1;
  }

  // 3. 初始化 PEME 估计器
  PolyEstimator estimator(
      config.camera.fx,
      config.camera.fy,
      config.camera.cx,
      config.camera.cy,
      config.camera.width,
      config.camera.height,
      config);

  // 4. 初始化日志记录器
  ResultLogger logger(config.paths.output_root, "poly", config.paths.suffix);

  // 5. 主循环
  int frame_count = 0;
  double t_start = 0;

  while (data_loader.hasNext())
  {
    EventPacket events = data_loader.getEventsInWindow(
        t_start,
        t_start + config.data.time_window,
        config.data.event_count);

    data_loader.undistortEvents(events, config.camera);

    if (events.empty())
      continue;

    std::cout << "Frame " << frame_count << ": " << events.size() << " events" << std::endl;

    // 运行 PEME 估计
    std::vector<double> dummy_depth; // 空深度，使用默认值 1.0
    RegistrationOutput result = estimator.run(events.xy, events.t, events.polarity, dummy_depth);
    if (logger.is_initialized())
    {
      logger.log(result);
    }

    t_start += config.data.time_window;
    frame_count++;
  }

  std::cout << "\n\nProcessed " << frame_count << " frames." << std::endl;
  return 0;
}   