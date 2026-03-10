#include <iostream>
#include "spatiotemporal_vo/config.hpp"
#include "spatiotemporal_vo/data_loader.hpp"
#include "methods/6dof_inc/cmax.hpp"
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
  // 3. 初始化 PEME 估计器
  CMaxSixDofInc estimator(
      config.camera.fx,
      config.camera.fy,
      config.camera.cx,
      config.camera.cy,
      config.camera.width,
      config.camera.height
      );
  // 4. 初始化日志记录器
  ResultLogger logger(config.paths.output_root, "cmax_inc", config.paths.suffix);

  // 5. 主循环
  int frame_count = 0;
  double t_start = 0;

  while (data_loader.hasNext())
  {
    EventPacket events = data_loader.getEventsInWindow(
        config.data.time_window,
        config.data.event_count, 
        config.data.unit_count,
        config.data.adaptive_split,
      true);

    data_loader.undistortEvents(events, config.camera);

    data_loader.projectEventsTo3D(events, config.camera);

    data_loader.computeReferenceTime(events, "mean");

    double time_span = events.t_end - events.t_start;
    if (events.empty())
      continue;

    std::cout << "Frame " << frame_count << ": " << events.size() << " events" << std::endl;

    // 运行 PEME 估计
    RegistrationOutput<6> result = estimator.run(config, events);
    result.timespan = {events.t_start, events.t_end};

    if (logger.is_initialized())
    {
      logger.log<6>(result);
    }
    t_start += time_span;
    frame_count++;
  }

  std::cout << "\n\nProcessed " << frame_count << " frames." << std::endl;
  return 0;
}