#pragma once

#include "spatiotemporal_vo/common.hpp"
#include "spatiotemporal_vo/config.hpp"
#include <string>
#include <vector>
#include <Eigen/Core>
#include <fstream>

namespace spatiotemporal_vo
{

  /**
   * @brief 事件数据加载器
   */
  class DataLoader
  {
  public:
    DataLoader();
    DataLoader(Config &config);
    /**
     * @brief 加载事件数据文件，仅懒加载
     * @param filepath 事件文件路径
     * @return 是否加载成功
     */
    bool loadEvents(const std::string &filepath);

    /**
     * @brief 获取指定时间窗口，或指定个数的事件
     * @param t_start 起始时间
     * @param t_end 结束时间
     * @param event_count 事件个数
     * @param select_num 挑选的事件个数
     * @return 事件数据包
     */
    EventPacket getEventsInWindow(double t_start, double t_end, int event_count);
    /**
     * @brief 自适应地获取事件数据包，保证每个包内的时间跨度不少于 min_delta_time 秒，且事件数在[min_count, max_count]范围内
     * @param min_delta_time 最小时间跨度
     * @param event_count 最大事件数
     * @param unit_count 基础事件数单位
     * @param adaptive_split 是否自适应分割
     * @param use_depth 是否使用深度信息（假设事件文件中包含深度数据）
     */
    EventPacket getEventsInWindow(double min_delta_time, int event_count, int unit_count, bool adaptive_split, bool use_depth = false);

    /**
     * @brief 检查是否还有数据
     */
    bool hasNext();

    /**
     * @brief 事件点去畸变
     * @param
     */
    void undistortEvents(EventPacket &packet, const CameraConfig &cam_cfg);

    /**
     * @brief 将事件点投影到3D空间
     * @param packet 事件包
     * @param cam_cfg 相机配置
     */
    void projectEventsTo3D(EventPacket &packet, const CameraConfig &cam_cfg);

    /**
     * @brief 计算参考时间并更新全部事件的时间戳
     * @param packet 事件包
     * @param method 参考时间计算方法（"front", "middle", "mean"）
     */
    void computeReferenceTime(EventPacket &packet, std::string method = "front");

  private:
    std::ifstream all_events_;
    EventPacket events_buffer_;
    std::vector<double> score_, score_var_;
    int grid_size_, grid_w_;
    double compute_grid_metric(const EventPacket &packet);
    std::vector<double> t_ppf_;
  };

} // namespace spatiotemporal_vo
