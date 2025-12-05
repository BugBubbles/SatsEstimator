#pragma once

#include "spatiotemporal_vo/common.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <vector>

namespace spatiotemporal_vo {

/**
 * @brief 计算反对称矩阵 (skew-symmetric matrix)
 * @param v 3维向量
 * @return 3x3 反对称矩阵
 */
Eigen::Matrix3d skew(const Eigen::Vector3d& v);

/**
 * @brief SO3 指数映射：将旋转向量转换为旋转矩阵
 * @param omega 旋转向量 (3维)
 * @return 旋转矩阵 (SO3)
 */
SO3 so3_exp(const Eigen::Vector3d& omega);

/**
 * @brief SO3 对数映射：将旋转矩阵转换为旋转向量
 * @param R 旋转矩阵 (SO3)
 * @return 旋转向量 (3维)
 */
Eigen::Vector3d so3_log(const SO3& R);

/**
 * @brief SE3 指数映射：将 twist 向量转换为 SE3 变换矩阵
 * @param xi twist 向量 (6维: [rho, omega] 或 [omega, rho])
 * @return SE3 变换矩阵
 */
SE3 se3_exp(const Eigen::Matrix<double, 6, 1>& xi);

/**
 * @brief SE3 对数映射：将 SE3 变换矩阵转换为 twist 向量
 * @param T SE3 变换矩阵
 * @return twist 向量 (6维)
 */
Eigen::Matrix<double, 6, 1> se3_log(const SE3& T);

/**
 * @brief 将 2D 像素点反投影到 3D 相机坐标系
 * @param K 相机内参 [fx, fy, cx, cy]
 * @param uv 2D 像素坐标
 * @param depth 深度值
 * @return 3D 点 (相机坐标系)
 */
Eigen::Vector3d backproject_point(
    const Eigen::Vector4d& K,
    const Eigen::Vector2d& uv,
    double depth
);

/**
 * @brief 批量反投影 2D 点到 3D
 * @param K 相机内参 [fx, fy, cx, cy]
 * @param uvs 2D 像素坐标数组
 * @param depths 深度值数组
 * @return 3D 点数组 (相机坐标系)
 */
std::vector<Eigen::Vector3d> backproject_points(
    const Eigen::Vector4d& K,
    const std::vector<Eigen::Vector2d>& uvs,
    const std::vector<double>& depths
);

/**
 * @brief 将 3D 点投影到 2D 像素平面
 * @param K 相机内参 [fx, fy, cx, cy]
 * @param point 3D 点 (相机坐标系)
 * @return 2D 像素坐标
 */
Eigen::Vector2d project_point(
    const Eigen::Vector4d& K,
    const Eigen::Vector3d& point
);

/**
 * @brief 批量投影 3D 点到 2D
 * @param K 相机内参 [fx, fy, cx, cy]
 * @param points 3D 点数组
 * @return 2D 像素坐标数组
 */
std::vector<Eigen::Vector2d> project_points(
    const Eigen::Vector4d& K,
    const std::vector<Eigen::Vector3d>& points
);

/**
 * @brief 使用位姿变换 3D 点
 * @param pose SE3 位姿
 * @param point 3D 点
 * @return 变换后的 3D 点
 */
Eigen::Vector3d transform_point(const SE3& pose, const Eigen::Vector3d& point);

/**
 * @brief 批量使用位姿变换 3D 点
 * @param pose SE3 位姿
 * @param points 3D 点数组
 * @return 变换后的 3D 点数组
 */
std::vector<Eigen::Vector3d> transform_points(
    const SE3& pose,
    const std::vector<Eigen::Vector3d>& points
);

/**
 * @brief 计算相机内参矩阵
 * @param fx x 方向焦距
 * @param fy y 方向焦距
 * @param cx x 方向主点
 * @param cy y 方向主点
 * @return 3x3 内参矩阵 K
 */
Eigen::Matrix3d make_intrinsic_matrix(double fx, double fy, double cx, double cy);

/**
 * @brief 计算投影点关于 SO3 参数的雅可比矩阵
 * @param K 相机内参 [fx, fy, cx, cy]
 * @param p_cam 相机坐标系下的 3D 点
 * @return 2x3 雅可比矩阵
 */
Eigen::Matrix<double, 2, 3> jacobian_project_wrt_point(
    const Eigen::Vector4d& K,
    const Eigen::Vector3d& p_cam
);

/**
 * @brief 计算旋转后点关于旋转向量的雅可比矩阵
 * @param p 3D 点
 * @return 3x3 雅可比矩阵 (-skew(R*p))
 */
Eigen::Matrix3d jacobian_rotation_wrt_omega(const Eigen::Vector3d& p);

} // namespace spatiotemporal_vo
