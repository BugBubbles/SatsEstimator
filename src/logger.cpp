#include "spatiotemporal_vo/logger.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <sys/stat.h>

namespace spatiotemporal_vo
{

  ResultLogger::ResultLogger(const std::string &output_path, const std::string &method, const std::string &suffix)
      : prefix_(method), suffix_(suffix), is_initialized_(false)
  {
    // 创建输出目录（如果不存在）
    createDirectory(output_path);

    // 生成带时间戳的文件名
    std::string filename = generateFilename();
    std::string filepath = output_path + "/" + filename;

    // 打开文件
    log_file_.open(filepath, std::ios::out | std::ios::trunc);
    if (!log_file_.is_open())
    {
      std::cerr << "Failed to open log file: " << filepath << std::endl;
    }

    is_initialized_ = true;
    std::cout << "Logger initialized: " << filepath << std::endl;
  }

  void ResultLogger::log(const RegistrationOutput &output)
  {
    if (!is_initialized_ || !log_file_.is_open())
    {
      std::cerr << "Logger not initialized!" << std::endl;
      return;
    }

    // 格式: idx time1 time2 runtime twist1 twist2 twist3
    log_file_ << std::fixed << output.stats.num_iterations << " "
              << std::setprecision(9)
              << output.timespan.first << " "
              << output.timespan.second << " "
              << output.stats.time_taken << " "
              << output.twist(0) << " "
              << output.twist(1) << " "
              << output.twist(2) << std::endl;

    // 刷新一次
    log_file_.flush();
  }

  bool ResultLogger::createDirectory(const std::string &path)
  {
    // 简单的目录创建（仅支持单级目录）
    struct stat st;
    if (stat(path.c_str(), &st) != 0)
    {
// 目录不存在，尝试创建
#ifdef _WIN32
      int ret = mkdir(path.c_str());
#else
      int ret = mkdir(path.c_str(), 0755);
#endif
      if (ret != 0)
      {
        std::cerr << "Failed to create directory: " << path << std::endl;
        return false;
      }
      std::cout << "Created directory: " << path << std::endl;
    }
    return true;
  }
  std::string ResultLogger::generateFilename() const
  {
    // 生成带时间戳的文件名
    auto now = std::time(nullptr);
    auto *tm = std::localtime(&now);

    std::ostringstream oss;
    oss << prefix_ << "_"
        << std::put_time(tm, "%Y%m%d_%H%M%S")
        << "_" << suffix_
        << ".txt";

    return oss.str();
  }
  void ResultLogger::close()
  {
    if (log_file_.is_open())
    {
      log_file_.close();
      std::cout << "Logger closed." << std::endl;
    }
    is_initialized_ = false;
  }
}