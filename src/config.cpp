#include "spatiotemporal_vo/config.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cctype>

namespace spatiotemporal_vo
{

  Config::Config()
  {
    setDefaults();
  }

  Config::Config(const std::string &config_file)
  {
    setDefaults();
    if (!config_file.empty())
    {
      load(config_file);
    }
  }

  void Config::setDefaults()
  {
    // Solver 默认配置
    solver.max_iters = 10;
    solver.lr = 1.0;
    solver.epsilon_T = 0.05;
    solver.trim_ratio = 0.9;
    solver.convergence_thresh = 1e-6;

    // Camera 默认配置
    camera.fx = 500.0;
    camera.fy = 500.0;
    camera.cx = 173.0;
    camera.cy = 130.0;
    camera.width = 346;
    camera.height = 260;

    // 路径默认配置
    paths.data_root = ".";
    paths.calib_path = "calib.txt";
    paths.event_file = "events.txt";
    paths.gt_file = "groundtruth.txt";
    paths.output_root = "./results";
    paths.suffix = "temp";

    // Data 默认配置
    data.time_window = 0.05;
    data.max_events = 100;
    data.event_count = 30000;
  }

  bool Config::loadCalibration(const std::string &filepath)
  {
    std::ifstream ifs(filepath);
    if (!ifs.is_open())
    {
      std::cerr << "Failed to open calibration file: " << filepath << std::endl;
      return false;
    }

    std::string line;

    while (std::getline(ifs, line))
    {
      // 跳过空行和注释
      if (line.empty() || line[0] == '#')
        continue;

      std::istringstream iss(line);
      double fx, fy, cx, cy, k1, k2, k3, p1, p2;

      // 假设校准文件格式为: fx fy cx cy k1 k2 k3 k4
      if (iss >> fx >> fy >> cx >> cy >> k1 >> k2 >> p1 >> p2 >> k3)
      {
        camera.fx = fx;
        camera.fy = fy;
        camera.cx = cx;
        camera.cy = cy;
        camera.distortion_args << k1, k2, p1, p2, k3;
      }
      else
      {
        std::cerr << "Invalid calibration file format: " << filepath << std::endl;
        return false;
      }
    }

    ifs.close();
    std::cout << "Camera intrinsics and distortion parameters loaded from: " << filepath << std::endl;
    return true;
  }

  bool Config::load(const std::string &config_file)
  {
    std::ifstream ifs(config_file);
    if (!ifs.is_open())
    {
      std::cerr << "Failed to open config file: " << config_file << std::endl;
      return false;
    }

    std::string line;
    std::string current_section;

    while (std::getline(ifs, line))
    {
      // 跳过空行和注释
      if (line.empty() || line[0] == '#')
        continue;

      // 去除前后空格
      size_t start = line.find_first_not_of(" \t");
      size_t end = line.find_last_not_of(" \t");
      if (start == std::string::npos)
        continue;
      line = line.substr(start, end - start + 1);

      // 检查是否是 section (如 "solver:", "camera:")
      if (line.back() == ':' && line.find(':') == line.length() - 1)
      {
        current_section = line.substr(0, line.length() - 1);
        continue;
      }

      // 解析 key: value
      size_t colon_pos = line.find(':');
      if (colon_pos == std::string::npos)
        continue;

      std::string key = line.substr(0, colon_pos);
      std::string value = line.substr(colon_pos + 1);

      // 去除 key 和 value 的空格
      key.erase(0, key.find_first_not_of(" \t"));
      key.erase(key.find_last_not_of(" \t") + 1);
      value.erase(0, value.find_first_not_of(" \t"));
      value.erase(value.find_last_not_of(" \t") + 1);

      try
      {
        // 根据 section 设置对应的值
        if (current_section == "solver")
        {
          if (key == "max_iters")
            solver.max_iters = std::stoi(value);
          else if (key == "lr")
            solver.lr = std::stod(value);
          else if (key == "epsilon_T")
            solver.epsilon_T = std::stod(value);
          else if (key == "trim_ratio")
            solver.trim_ratio = std::stod(value);
          else if (key == "convergence_thresh")
            solver.convergence_thresh = std::stod(value);
          else if (key == "regularization")
            solver.regularization = std::stod(value);
        }
        else if (current_section == "camera")
        {
          if (key == "width")
            camera.width = std::stoi(value);
          else if (key == "height")
            camera.height = std::stoi(value);
        }
        else if (current_section == "paths")
        {
          if (key == "data_root")
            paths.data_root = value;
          else if (key == "sequences")
            paths.sequences = value;
          else if (key == "event_file")
            paths.event_file = value;
          else if (key == "gt_file")
            paths.gt_file = value;
          else if (key == "calib_file")
            paths.calib_path = value;
          else if (key == "output_root")
            paths.output_root = value;
          else if (key == "suffix")
            paths.suffix = value;
        }
        else if (current_section == "data")
        {
          if (key == "time_window")
            data.time_window = std::stod(value);
          else if (key == "max_events")
            data.max_events = std::stoll(value);
          else if (key == "event_count")
            data.event_count = std::stoi(value);
          else if (key == "grid_size")
            data.grid_size = std::stoi(value);
          else if (key == "max_count")
            data.max_count = std::stoi(value);
          else if (key == "unit_count")
            data.unit_count = std::stoi(value);
          else if (key == "adaptive_split")
            data.adaptive_split = (value == "true" || value == "1");
        }
      }
      catch (const std::invalid_argument &e)
      {
        std::cerr << "Invalid value for key: " << key << ". Skipping." << std::endl;
      }
      catch (const std::out_of_range &e)
      {
        std::cerr << "Value out of range for key: " << key << ". Skipping." << std::endl;
      }
    }

    // 加载校准文件
    loadCalibration(get_file_path(paths.calib_path));

    ifs.close();
    std::cout << "Config loaded from: " << config_file << std::endl;
    return true;
  }

  bool Config::save(const std::string &config_file) const
  {
    std::ofstream ofs(config_file);
    if (!ofs.is_open())
    {
      std::cerr << "Failed to open config file for writing: " << config_file << std::endl;
      return false;
    }

    ofs << "# Spatiotemporal VO Configuration\n\n";

    ofs << "solver:\n";
    ofs << "  max_iters: " << solver.max_iters << "\n";
    ofs << "  lr: " << solver.lr << "\n";
    ofs << "  epsilon_T: " << solver.epsilon_T << "\n";
    ofs << "  trim_ratio: " << solver.trim_ratio << "\n";
    ofs << "  convergence_thresh: " << solver.convergence_thresh << "\n\n";

    ofs << "camera:\n";
    ofs << "  fx: " << camera.fx << "\n";
    ofs << "  fy: " << camera.fy << "\n";
    ofs << "  cx: " << camera.cx << "\n";
    ofs << "  cy: " << camera.cy << "\n";
    ofs << "  width: " << camera.width << "\n";
    ofs << "  height: " << camera.height << "\n";
    ofs << "  distortion_args: " << camera.distortion_args.transpose() << "\n\n";

    ofs << "paths:\n";
    ofs << "  data_root: " << paths.data_root << "\n";
    ofs << "  calib_path: " << paths.calib_path << "\n";
    ofs << "  event_file: " << paths.event_file << "\n";
    ofs << "  gt_file: " << paths.gt_file << "\n";
    ofs << "  output_root: " << paths.output_root << "\n";
    ofs << "  suffix: " << paths.suffix << "\n\n";

    ofs << "data:\n";
    ofs << "  time_window: " << data.time_window << "\n";
    ofs << "  max_events: " << data.max_events << "\n";
    ofs << "  event_count: " << data.event_count << "\n\n";
    ofs << "  grid_size: " << data.grid_size << "\n";
    ofs << "  max_count: " << data.max_count << "\n";
    ofs << "  unit_count: " << data.unit_count << "\n";
    ofs << "  adaptive_split: " << (data.adaptive_split ? "true" : "false") << "\n";

    ofs.close();
    std::cout << "Config saved to: " << config_file << std::endl;
    return true;
  }

  void Config::print() const
  {
    std::cout << "========== Configuration ==========" << std::endl;

    std::cout << "\n[Solver]" << std::endl;
    std::cout << "  max_iters: " << solver.max_iters << std::endl;
    std::cout << "  lr: " << solver.lr << std::endl;
    std::cout << "  epsilon_T: " << solver.epsilon_T << std::endl;
    std::cout << "  trim_ratio: " << solver.trim_ratio << std::endl;
    std::cout << "  convergence_thresh: " << solver.convergence_thresh << std::endl;
    std::cout << "  regularization: " << solver.regularization << std::endl;

    std::cout << "\n[Camera]" << std::endl;
    std::cout << "  fx: " << camera.fx << std::endl;
    std::cout << "  fy: " << camera.fy << std::endl;
    std::cout << "  cx: " << camera.cx << std::endl;
    std::cout << "  cy: " << camera.cy << std::endl;
    std::cout << "  width: " << camera.width << std::endl;
    std::cout << "  height: " << camera.height << std::endl;
    std::cout << "  distortion_args: " << camera.distortion_args.transpose() << std::endl;

    std::cout << "\n[Paths]" << std::endl;
    std::cout << "  data_root: " << paths.data_root << std::endl;
    std::cout << "  calib_path: " << paths.calib_path << std::endl;
    std::cout << "  event_file: " << paths.event_file << std::endl;
    std::cout << "  gt_file: " << paths.gt_file << std::endl;
    std::cout << "  output_root: " << paths.output_root << std::endl;
    std::cout << "  suffix: " << paths.suffix << std::endl;

    std::cout << "\n[Data]" << std::endl;
    std::cout << "  time_window: " << data.time_window << std::endl;
    std::cout << "  max_events: " << data.max_events << std::endl;
    std::cout << "  event_count: " << data.event_count << std::endl;
    std::cout << "  grid_size: " << data.grid_size << std::endl;
    std::cout << "  max_count: " << data.max_count << std::endl;
    std::cout << "  unit_count: " << data.unit_count << std::endl;
    std::cout << "  adaptive_split: " << (data.adaptive_split ? "true" : "false") << std::endl;

    std::cout << "===================================" << std::endl;
  }

} // namespace spatiotemporal_vo