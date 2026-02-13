#pragma once

#include "spatiotemporal_vo/common.hpp"
#include <string>
#include <fstream>
namespace spatiotemporal_vo
{
  /**
   * @brief 简化的可视化器 (仅使用 OpenCV)
   */
  class ResultLogger
  {
  public:
    /**
     * @brief 初始化日志记录器
     * @param filepath 日志路径
     * @param method 方法名称前缀
     * @param suffix 日志文件后缀
     */
    ResultLogger(const std::string &filepath, const std::string &method, const std::string &suffix);
    /**
     * @brief 获取日志记录器状态
     */
    bool is_initialized() const { return is_initialized_; };
    /**
     * @brief 记录日志
     * @param output 优化器输出结果
     */
    template<int N>
    void log(const RegistrationOutput<N> &output)
      {
    if (!is_initialized_ || !log_file_.is_open())
    {
      std::cerr << "Logger not initialized!" << std::endl;
      return;
    }

    // 格式: idx time1 time2 runtime (trans1 trans2 trans3) twist1 twist2 twist3
    log_file_ << std::fixed << output.stats.num_iterations << " "
              << std::setprecision(9)
              << output.timespan.first << " "
              << output.timespan.second << " "
              << output.stats.time_taken << " " 
              << output.params.transpose() << std::endl;
    // 刷新一次
    log_file_.flush();
  };
    /**
     * @brief 关闭日志文件
     */
    void close();

  private:
    std::ofstream log_file_;
    bool is_initialized_;
    std::string suffix_, prefix_;
    bool createDirectory(const std::string &path);
    std::string generateFilename() const;
  };

} // namespace spatiotemporal_vo
