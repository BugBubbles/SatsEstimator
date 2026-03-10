#include <iostream>
#include "spatiotemporal_vo/config.hpp"
#include "spatiotemporal_vo/data_loader.hpp"
#include "methods/peme_estimator.hpp"
#include "spatiotemporal_vo/logger.hpp"

using namespace spatiotemporal_vo;

int main(int argc, char **argv)
{
  std::cout << "=== PEME Visual Odometry with Dynamic Batch (C++) ===" << std::endl;

  // 1. 加载配置
  std::string config_file = (argc > 1) ? argv[1] : "../configs/default.yaml";
  Config config(config_file);
  config.print();

  // 2. 初始化数据加载器
  DataLoader data_loader(config);
  std::string event_file = config.paths.data_root + "/" +
                           config.paths.sequences + "/" +
                           config.paths.event_file;

  if (!data_loader.loadEvents(event_file))
  {
    std::cerr << "Failed to load events from: " << event_file << std::endl;
    return -1;
  }

  // 3. 初始化 PEME 估计器
  PEMEEstimator estimator(
      config.camera.fx,
      config.camera.fy,
      config.camera.cx,
      config.camera.cy,
      config.camera.width,
      config.camera.height,
      config);

  // 4. 初始化日志记录器
  ResultLogger logger(config.paths.output_root, "peme_db", config.paths.suffix);

  // 5. 主循环
  int frame_count = 0;
  double t_start = 0, overlap, overlap_past = 0, time_taken = 0, time_start;
  EventPacket events, past_events_backup;
  // 加载初始事件片段
  EventPacket past_events = data_loader.getEventsInWindow(config.data.unit_count, false);

  data_loader.undistortEvents(past_events, config.camera);

  data_loader.projectEventsTo3D(past_events, config.camera);

  // data_loader.computeReferenceTime(past_events, "front");

  while (data_loader.hasNext())
  {
    events = data_loader.getEventsInWindow(config.data.unit_count, false);

    data_loader.undistortEvents(events, config.camera);

    data_loader.projectEventsTo3D(events, config.camera);

    // data_loader.computeReferenceTime(events, "front");

    if (events.empty())
      continue;

    // 运行 PEME 估计
    past_events_backup = past_events;
    data_loader.computeReferenceTime(past_events_backup, "front");
    RegistrationOutput<3> result = estimator.runRotation(past_events_backup);
    // 计算重叠度
    time_start = omp_get_wtime();
    overlap = estimator.overlapRatio(Vec3::Zero(), result.params, past_events, events);
    std::cout << "Overlap: " << overlap << std::endl;
    if (overlap > overlap_past)
    {
      overlap_past = overlap;
      past_events += events;
      time_taken += result.stats.time_taken;
      time_start = omp_get_wtime() - time_start;
      time_taken += time_start;
      continue;
    }
    // 不满足重叠度条件，退出动态batch循环，输出结果
    std::cout << "Frame " << frame_count << ": " << past_events.size() << " events" << std::endl;

    result.timespan = {past_events.t_start, past_events.t_end};
    result.stats.time_taken = time_taken + result.stats.time_taken;
    result.stats.num_iterations = frame_count;
    // 用时将包括几次优化的总时间以及计算重叠度的时间
    if (logger.is_initialized())
    {
      logger.log<3>(result);
    }
    t_start += (past_events.t_end - past_events.t_start);
    frame_count++;
    past_events = events;
    overlap_past = 0;
    time_taken = 0;
  }

  std::cout << "\n\nProcessed " << frame_count << " frames." << std::endl;
  return 0;
}