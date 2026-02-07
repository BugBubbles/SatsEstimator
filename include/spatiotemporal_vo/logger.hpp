#pragma once

#include "spatiotemporal_vo/common.hpp"
#include <string>
#include <fstream>
namespace spatiotemporal_vo
{
  enum LogMode
  {
    Rotation,
    SixDof,
    Translation
  };

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
    void log(const RegistrationOutput &output, LogMode mode);
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
