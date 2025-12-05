#pragma once

#include <string>
#include "spatiotemporal_vo/common.hpp"
#include <Eigen/Core>

namespace spatiotemporal_vo {

/**
 * @brief 求解器配置参数
 */
struct SolverConfig {
    int max_iters = 10;         // 最大迭代次数
    double lr = 1.0;            // 学习率
    double epsilon_T = 0.05;    // 时间邻域阈值
    bool trimming = false;      // 是否使用 trimming
    double trim_ratio = 0.9;    // trimming 比例
    double convergence_thresh = 1e-6;  // 收敛阈值
};

/**
 * @brief 相机内参配置
 */
struct CameraConfig {
    double fx = 500.0;          // x 方向焦距
    double fy = 500.0;          // y 方向焦距
    double cx = 173.0;          // x 方向主点
    double cy = 130.0;          // y 方向主点
    int width = 346;            // 图像宽度
    int height = 260;           // 图像高度
    //畸变参数
    Eigen::Matrix<double, 5, 1> distortion_args;
};

/*
 * @brief 路径配置
*/
struct PathConfig
{
    std::string data_root = ".";              // 数据集路径
    std::string calib_path = "calib.txt";    // 相机内参路径
    std::string sequences = "boxes_rotation"; // 默认数据序列
    std::string event_file = "events.txt";   // 事件文件名
    std::string gt_file = "groundtruth.txt"; // 真值文件名
    std::string output_root = "./result";    // 输出路径
    std::string suffix = "temp";             // 输出文件后缀
};

/**
 * @brief 数据加载配置
 */
struct DataConfig {
    double time_window = 0.05;               // 时间窗口 (秒)
    long long max_events = 100000;                 // 最大事件数
    int event_count = event_count;           // 数量窗口
};

/**
 * @brief 主配置类
 */
class Config {
public:
    Config();
    explicit Config(const std::string& config_file);

    /**
     * @brief 从 YAML 文件加载配置
     * @param config_file YAML 配置文件路径
     * @return 是否加载成功
     */
    bool load(const std::string& config_file);
    
    /**
     * @brief 保存配置到 YAML 文件
     * @param config_file YAML 配置文件路径
     * @return 是否保存成功
     */
    bool save(const std::string& config_file) const;

    /**
     * @brief 打印当前配置
     */
    void print() const;

    // 各模块配置
    SolverConfig solver;
    CameraConfig camera;
    DataConfig data;
    PathConfig paths;

  private:
    /**
     * @brief 设置默认值
     */
    void setDefaults();
    /**
     * @brief 加载相机标定文件
     * @param filepath 真值文件路径
     * @return 是否加载成功
     */
    bool loadCalibration(const std::string& filepath);
    inline std::string get_file_path(const std::string &filepath) { return paths.data_root + "/" + paths.sequences + "/" + filepath; };
};

} // namespace spatiotemporal_vo
