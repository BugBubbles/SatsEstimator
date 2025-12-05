#include "spatiotemporal_vo/geometry.hpp"
#include <cmath>

namespace spatiotemporal_vo {

// 将点云根据 SE3 位姿进行变换
PointCloudPtr transformPointCloud(PointCloudConstPtr cloud, const SE3& pose) {
    PointCloudPtr transformed_cloud(new PointCloud);
    Eigen::Matrix4d transform_matrix = pose.matrix();

    // 使用 PCL 的 transformPointCloud 函数，但传入 Eigen 矩阵
    pcl::transformPointCloud(*cloud, *transformed_cloud, transform_matrix);
    
    return transformed_cloud;
}

// Sophus SE3 到 Eigen Matrix4d 的转换示例 (Sophus 自带 .matrix() 方法)
Eigen::Matrix4d se3ToEigen(const SE3& se3) {
    return se3.matrix();
}

// 旋转矩阵的 skew 对应
Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S <<     0, -v(2),  v(1),
          v(2),     0, -v(0),
         -v(1),  v(0),     0;
    return S;
}

// SO3 的指数映射
SO3 so3_exp(const Eigen::Vector3d& omega) {
    return SO3::exp(omega);
}

// SO3 的对数映射
Eigen::Vector3d so3_log(const SO3& R) {
    return R.log();
}

// SE3 的指数映射
SE3 se3_exp(const Eigen::Matrix<double, 6, 1>& xi) {
    // Sophus 使用的约定：前 3 维是平移，后 3 维是旋转
    // xi = [rho (translation), omega (rotation)]
    return SE3::exp(xi);
}

// SE3 的对数映射
Eigen::Matrix<double, 6, 1> se3_log(const SE3& T) {
    return T.log();
}

// 从图像坐标反投影到 3D 点
Eigen::Vector3d backproject_point(
    const Eigen::Vector4d& K,
    const Eigen::Vector2d& uv,
    double depth
) {
    double fx = K(0);
    double fy = K(1);
    double cx = K(2);
    double cy = K(3);

    double x = (uv(0) - cx) / fx * depth;
    double y = (uv(1) - cy) / fy * depth;
    double z = depth;

    return Eigen::Vector3d(x, y, z);
}

// 从图像坐标反投影到 3D 点的向量
std::vector<Eigen::Vector3d> backproject_points(
    const Eigen::Vector4d& K,
    const std::vector<Eigen::Vector2d>& uvs,
    const std::vector<double>& depths
) {
    std::vector<Eigen::Vector3d> points;
    points.reserve(uvs.size());

    for (size_t i = 0; i < uvs.size(); ++i) {
        double d = (i < depths.size()) ? depths[i] : 1.0;
        points.push_back(backproject_point(K, uvs[i], d));
    }

    return points;
}

// 从 3D 点投影到图像坐标
Eigen::Vector2d project_point(
    const Eigen::Vector4d& K,
    const Eigen::Vector3d& point
) {
    double fx = K(0);
    double fy = K(1);
    double cx = K(2);
    double cy = K(3);

    if (std::abs(point(2)) < 1e-8) {
        return Eigen::Vector2d(cx, cy); // 避免除以零
    }

    double u = fx * point(0) / point(2) + cx;
    double v = fy * point(1) / point(2) + cy;

    return Eigen::Vector2d(u, v);
}

// 从 3D 点投影到图像坐标
std::vector<Eigen::Vector2d> project_points(
    const Eigen::Vector4d& K,
    const std::vector<Eigen::Vector3d>& points
) {
    std::vector<Eigen::Vector2d> uvs;
    uvs.reserve(points.size());

    for (const auto& p : points) {
        uvs.push_back(project_point(K, p));
    }

    return uvs;
}

// 位姿变换点
Eigen::Vector3d transform_point(const SE3& pose, const Eigen::Vector3d& point) {
    return pose * point;
}

// 位姿变换点
std::vector<Eigen::Vector3d> transform_points(
    const SE3& pose,
    const std::vector<Eigen::Vector3d>& points
) {
    std::vector<Eigen::Vector3d> transformed;
    transformed.reserve(points.size());

    for (const auto& p : points) {
        transformed.push_back(pose * p);
    }

    return transformed;
}

// 内参矩阵
Eigen::Matrix3d make_intrinsic_matrix(double fx, double fy, double cx, double cy) {
    Eigen::Matrix3d K;
    K << fx,  0, cx,
          0, fy, cy,
          0,  0,  1;
    return K;
}

// 从图像坐标到点的雅可比
Eigen::Matrix<double, 2, 3> jacobian_project_wrt_point(
    const Eigen::Vector4d& K,
    const Eigen::Vector3d& p_cam
) {
    double fx = K(0);
    double fy = K(1);
    double x = p_cam(0);
    double y = p_cam(1);
    double z = p_cam(2);

    if (std::abs(z) < 1e-8) {
        return Eigen::Matrix<double, 2, 3>::Zero();
    }

    double z_inv = 1.0 / z;
    double z_inv2 = z_inv * z_inv;

    Eigen::Matrix<double, 2, 3> J;
    // du/d[x, y, z]
    J(0, 0) = fx * z_inv;
    J(0, 1) = 0;
    J(0, 2) = -fx * x * z_inv2;
    // dv/d[x, y, z]
    J(1, 0) = 0;
    J(1, 1) = fy * z_inv;
    J(1, 2) = -fy * y * z_inv2;

    return J;
}

// 旋转矩阵对 omega 的雅可比
Eigen::Matrix3d jacobian_rotation_wrt_omega(const Eigen::Vector3d& p) {
    // d(R*p)/d(omega) = -skew(R*p) 当 omega 接近 0 时
    return -skew(p);
}

} // namespace spatiotemporal_vo