#include <iostream>
#include <filesystem>
#include "spatiotemporal_vo/config.hpp"
#include "spatiotemporal_vo/data_loader.hpp"
#include "methods/rotation/cmax.hpp"
#include "spatiotemporal_vo/logger.hpp"

using namespace spatiotemporal_vo;

int main(int argc, char **argv)
{
  std::cout << "=== CMax Visual Odometry (C++) ===" << std::endl;

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


  // 4. 初始化日志记录器
  ResultLogger logger(config.paths.output_root, "cmax", config.paths.suffix);

  // 5. 主循环
  int frame_count = 0;
  double t_start = 0;

  while (data_loader.hasNext())
  {
    EventPacket events = data_loader.getEventsInWindow(
        config.data.time_window,
        config.data.max_count, 
        config.data.unit_count,
        config.data.adaptive_split);

    data_loader.undistortEvents(events, config.camera);

    std::vector<double> depth;
    data_loader.projectEventsTo3D(events, config.camera, depth);

    data_loader.computeReferenceTime(events, "front");

    double time_span = events.t.back() - events.t.front();
    if (events.empty())
      continue;

    std::cout << "Frame " << frame_count << ": " << events.size() << " events" << std::endl;
  //  初始化 PEME 估计器
  CMaxRotation estimator(
      config.camera.fx,
      config.camera.fy,
      config.camera.cx,
      config.camera.cy,
      config.camera.width,
      config.camera.height,
      events
      );
    // 运行 PEME 估计
    RegistrationOutput result = estimator.run(config);
    result.timespan = {t_start, t_start + time_span};

    if (logger.is_initialized())
    {
      logger.log(result);
    }
    t_start += time_span;
    frame_count++;
  }

  std::cout << "\n\nProcessed " << frame_count << " frames." << std::endl;
  return 0;
}   