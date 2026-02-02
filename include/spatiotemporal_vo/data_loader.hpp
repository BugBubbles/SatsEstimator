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
   * @brief 事件数据结构
   */
  struct EventPacket
  {
    std::vector<Eigen::Vector2d> xy; // 事件坐标
    std::vector<double> t;           // 时间戳
    std::vector<int> polarity;       // 极性 (1 或 -1)

    size_t size() const { return xy.size(); }
    bool empty() const { return xy.empty(); }
    double timespan() const { return t.back() - t.front(); };
    void clear()
    {
      xy.clear();
      t.clear();
      polarity.clear();
    }
  };

  /**
   * @brief 事件数据加载器
   */
  class DataLoader
  {
  public:
    DataLoader();
    explicit DataLoader(const std::string &data_root);

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
    EventPacket getEventsInWindow(double t_start, double t_end, int event_count, int select_num = 0);

    /**
     * @brief 检查是否还有数据
     */
    bool hasNext();

    /**
     * @brief 事件点去畸变
     * @param
     */
    void undistortEvents(EventPacket &packet, const CameraConfig &cam_cfg);

  private:
    std::string dataset_path_;
    std::ifstream all_events_;
    std::vector<double> gt_timestamps_;
    size_t current_index_;
    double current_time_;
  };

} // namespace spatiotemporal_vo
