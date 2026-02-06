// #include "spatiotemporal_vo/common.hpp"
// #include "methods/cmax_estimator.hpp"
// #include <sophus/so3.hpp>
// #include <algorithm>
// #include <numeric>
// #include <random>
// #include <chrono>
// #include <iostream>
// #include <ceres/ceres.h>
// #include <ceres/cubic_interpolation.h>
// #include <cmath>
// #include <pcl/kdtree/kdtree_flann.h>
// #include <pcl/point_types.h>
// #include <pcl/point_cloud.h>
// #include <omp.h>

// namespace spatiotemporal_vo
// {

//   class CMaxOptimizer
//   {
//   private:
//     // 你的数据成员（和原CMaxFunction一致）
//     std::vector<Vec3> points_;
//     int width_, height_;
//     Eigen::Matrix3d intrinsic_;
//     Eigen::Matrix<double, Eigen::Dynamic, 13, Eigen::RowMajor> iwe_data_;
//     EigenMat warped_image_, pixel_grad_, pixel_hess_;
//     std::vector<EigenMat> iwe_grad_;
//     std::vector<EigenMat> iwe_hess_;
//     std::vector<double> delta_time_;
//     std::vector<int> polarity_;
//     pcl::PointCloud<pcl::PointXY>::Ptr cloud_;
//     int n_points_;
//     int padding_ = 100;
//     const float radius = 4.0f;
//     // 添加 final_loss_ 成员
//     double final_loss_ = 0.0;
//     Vec3 est_angleAxis_;

//   public:
//     void getEstimatedParams(Vec3 &est_angleAxis) const { est_angleAxis = est_angleAxis_; }
//     double getFinalLoss() const { return final_loss_; }
//     CMaxOptimizer(const std::vector<Vec3> &points,
//                   const std::vector<double> &delta_time,
//                   const std::vector<int> &polarity,
//                   int width, int height,
//                   const Eigen::Matrix3d &K,
//                   const Vec3 &initial_angleAxis)
//         : points_(points), delta_time_(delta_time), polarity_(polarity),
//           width_(width), height_(height), intrinsic_(K), est_angleAxis_(initial_angleAxis)
//     {
//       n_points_ = points_.size();
//       warped_image_.resize(height_ + 2 * padding_, width_ + 2 * padding_);
//       pixel_grad_.resize(height_ + 2 * padding_, width_ + 2 * padding_);
//       pixel_hess_.resize(height_ + 2 * padding_, width_ + 2 * padding_);
//       iwe_data_.resize((height_ + 2 * padding_) * (width_ + 2 * padding_), 13); // 13 channel for 1 loss + 3 grad + 9 hess
//       iwe_grad_.resize(3);
//       iwe_hess_.resize(9);
//       for (int i = 0; i < 3; ++i)
//       {
//         iwe_grad_[i].resize(height_ + 2 * padding_, width_ + 2 * padding_);
//       }
//       for (int i = 0; i < 9; ++i)
//       {
//         iwe_hess_[i].resize(height_ + 2 * padding_, width_ + 2 * padding_);
//       }
//     }

//     void compute_loss(const Vec3 &parameters, double &loss)
//     {

//       Eigen::Matrix<double, 3, 1> delta_points_T, delta_second_points_T;
//       Eigen::Matrix<double, 3, 1> points_T;
//       // 创建翘曲事件图像
//       warped_image_.setZero();
//       // 构建二维PCL点云
//       // 释放cloud旧内存
//       cloud_.reset(new pcl::PointCloud<pcl::PointXY>());
//       cloud_->resize(n_points_);
//       for (int i = 0; i < n_points_; ++i)
//       {
//         auto point_ = points_[i];
//         // Taylor 二阶展开: points * skew_matrix * skew_matrix
//         delta_points_T(0) = -parameters[2] * point_(1) + parameters[1] * point_(2);
//         delta_points_T(1) = parameters[2] * point_(0) - parameters[0] * point_(2);
//         delta_points_T(2) = -parameters[1] * point_(0) + parameters[0] * point_(1);

//         // delta_second_points_T(0) = -parameters[2] * delta_points_T(1) + parameters[1] * delta_points_T(2);
//         // delta_second_points_T(1) = parameters[2] * delta_points_T(0) - parameters[0] * delta_points_T(2);
//         // delta_second_points_T(2) = -parameters[1] * delta_points_T(0) + parameters[0] * delta_points_T(1);
//         // 计算 warp 后的 3D 点
//         double dt_early = delta_time_[i];
//         // double dt_sq = 0.5 * dt_early * dt_early;
//         // points_T(0) = point_(0) + delta_points_T(0) * dt_early + delta_second_points_T(0) * dt_sq;
//         // points_T(1) = point_(1) + delta_points_T(1) * dt_early + delta_second_points_T(1) * dt_sq;
//         // points_T(2) = point_(2) + delta_points_T(2) * dt_early + delta_second_points_T(2) * dt_sq;

//         points_T(0) = point_(0) + delta_points_T(0) * dt_early;
//         points_T(1) = point_(1) + delta_points_T(1) * dt_early;
//         points_T(2) = point_(2) + delta_points_T(2) * dt_early;
//         // 投影到 2D
//         (*cloud_)[i].x = points_T(0) / points_T(2) * intrinsic_(0, 0) + intrinsic_(0, 2);
//         (*cloud_)[i].y = points_T(1) / points_T(2) * intrinsic_(1, 1) + intrinsic_(1, 2);
//         // 使用双线性插值将当前事件翘曲至图像
//         int x = static_cast<int>((*cloud_)[i].x);
//         int y = static_cast<int>((*cloud_)[i].y);
//         double dx = (*cloud_)[i].x - x;
//         double dy = (*cloud_)[i].y - y;

//         if (x >= -padding_ && x < width_ + padding_ - 1 && y >= -padding_ && y < height_ + padding_ - 1)
//         {
//           double w4 = dx * dy;
//           double w3 = (1.0 - dx) * dy;
//           double w2 = dx * (1.0 - dy);
//           double w1 = (1.0 - dx) * (1.0 - dy);

//           warped_image_(y + padding_, x + padding_) += w1;
//           warped_image_(y + padding_, x + 1 + padding_) += w2;
//           warped_image_(y + padding_ + 1, x + padding_) += w3;
//           warped_image_(y + padding_ + 1, x + 1 + padding_) += w4;
//         }
//       }
//       // 平滑翘曲图像
//       // Eigen 转为 OpenCV 格式
//       cv::Mat warped_cv(height_ + 2 * padding_, width_ + 2 * padding_, CV_64F, warped_image_.data());
//       cv::GaussianBlur(warped_cv, warped_cv, cv::Size(5, 5), 1.0);
//       // 计算残差（方差）
//       // double mean = warped_image_.mean();
//       // loss = -(warped_image_.array() - mean).square().mean();
//       cv::Scalar mean, var;
//       cv::meanStdDev(warped_cv, mean, var);
//       loss = -(var[0] * var[0]);
//     }
//     void compute_gradient(const Vec3 &parameters, Vec3 &gradient, Eigen::Matrix3d &hessian, bool use_prev) const
//     {
//       pcl::KdTreeFLANN<pcl::PointXY> kdtree;
//       kdtree.setInputCloud(cloud_);
//       gradient.setZero();
//       if (!use_prev)
//         hessian.setZero();

//       // 局部累加器
//       int num_threads = omp_get_max_threads();
//       std::vector<Vec3> grad_private(num_threads, Vec3::Zero());
//       std::vector<Eigen::Matrix3d> hess_private(num_threads, Eigen::Matrix3d::Zero());

// #pragma omp parallel for
//       for (int k = 0; k < n_points_; ++k)
//       {
//         int tid = omp_get_thread_num();
//         std::vector<int> neighbor_indices;
//         std::vector<float> neighbor_dists;

//         pcl::PointXY search_point = (*cloud_)[k];
//         auto k1 = (*cloud_)[k].x - intrinsic_(0, 2);
//         auto k2 = (*cloud_)[k].y - intrinsic_(1, 2);
//         kdtree.radiusSearch(search_point, radius, neighbor_indices, neighbor_dists);

//         for (size_t idx = 0; idx < neighbor_indices.size(); ++idx)
//         {
//           int l = neighbor_indices[idx];
//           if (l == k)
//             continue;
//           float dist2 = neighbor_dists[idx];
//           double weight = 0.079577 * std::exp(-dist2 * dist2 / 4.0);
//           double tl = delta_time_[l];
//           double tk = delta_time_[k];
//           auto l1 = (*cloud_)[l].x - intrinsic_(0, 2);
//           auto l2 = (*cloud_)[l].y - intrinsic_(1, 2);

//           grad_private[tid][0] += weight * tk * ((intrinsic_(1, 1) + k2 * k2 / intrinsic_(1, 1)) * (-k2 + l2) + k1 * k2 * (-k1 + l1) / intrinsic_(1, 1));
//           grad_private[tid][1] += weight * tk * ((-intrinsic_(0, 0) - k1 * k1 / intrinsic_(0, 0)) * (-k1 + l1) - k1 * k2 * (-k2 + l2) / intrinsic_(0, 0));
//           grad_private[tid][2] += weight * tk * (-k1 * (-k2 + l2) + k2 * (-k1 + l1));
//           if (!use_prev)
//           {
//             hess_private[tid](0, 0) += weight * ((intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (pow(tk, 2) * (intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) - k1 * k2 * pow(tk, 2) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) / intrinsic_(1, 1)) + (intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (-tk * tl * (intrinsic_(1, 1) + pow(l2, 2) / intrinsic_(1, 1)) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) + l1 * l2 * tk * tl * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) / intrinsic_(1, 1)) + k1 * k2 * (-pow(tk, 2) * (intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) + k1 * k2 * pow(tk, 2) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) / intrinsic_(1, 1)) / intrinsic_(1, 1) + k1 * k2 * (tk * tl * (intrinsic_(1, 1) + pow(l2, 2) / intrinsic_(1, 1)) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) - l1 * l2 * tk * tl * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) / intrinsic_(1, 1)) / intrinsic_(1, 1));
//             hess_private[tid](0, 1) += weight * ((-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (-pow(tk, 2) * (intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) + k1 * k2 * pow(tk, 2) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) / intrinsic_(1, 1)) + (-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (tk * tl * (intrinsic_(1, 1) + pow(l2, 2) / intrinsic_(1, 1)) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) - l1 * l2 * tk * tl * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) / intrinsic_(1, 1)) - k1 * k2 * (pow(tk, 2) * (intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) - k1 * k2 * pow(tk, 2) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) / intrinsic_(1, 1)) / intrinsic_(0, 0) - k1 * k2 * (-tk * tl * (intrinsic_(1, 1) + pow(l2, 2) / intrinsic_(1, 1)) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) + l1 * l2 * tk * tl * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) / intrinsic_(1, 1)) / intrinsic_(0, 0));
//             hess_private[tid](0, 2) += weight * (-k1 * (pow(tk, 2) * (intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) - k1 * k2 * pow(tk, 2) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) / intrinsic_(1, 1)) - k1 * (-tk * tl * (intrinsic_(1, 1) + pow(l2, 2) / intrinsic_(1, 1)) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) + l1 * l2 * tk * tl * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) / intrinsic_(1, 1)) + k2 * (-pow(tk, 2) * (intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) + k1 * k2 * pow(tk, 2) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) / intrinsic_(1, 1)) + k2 * (tk * tl * (intrinsic_(1, 1) + pow(l2, 2) / intrinsic_(1, 1)) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) - l1 * l2 * tk * tl * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) / intrinsic_(1, 1)));
//             // hess_private[tid](1, 0) += weight * ((intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (-pow(tk, 2) * (-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) - k1 * k2 * pow(tk, 2) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) / intrinsic_(0, 0)) + (intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (tk * tl * (-intrinsic_(0, 0) - pow(l1, 2) / intrinsic_(0, 0)) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) + l1 * l2 * tk * tl * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) / intrinsic_(0, 0)) + k1 * k2 * (pow(tk, 2) * (-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) + k1 * k2 * pow(tk, 2) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) / intrinsic_(0, 0)) / intrinsic_(1, 1) + k1 * k2 * (-tk * tl * (-intrinsic_(0, 0) - pow(l1, 2) / intrinsic_(0, 0)) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) - l1 * l2 * tk * tl * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) / intrinsic_(0, 0)) / intrinsic_(1, 1));
//             hess_private[tid](1, 1) += weight * ((-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (pow(tk, 2) * (-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) + k1 * k2 * pow(tk, 2) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) / intrinsic_(0, 0)) + (-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (-tk * tl * (-intrinsic_(0, 0) - pow(l1, 2) / intrinsic_(0, 0)) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) - l1 * l2 * tk * tl * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) / intrinsic_(0, 0)) - k1 * k2 * (-pow(tk, 2) * (-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) - k1 * k2 * pow(tk, 2) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) / intrinsic_(0, 0)) / intrinsic_(0, 0) - k1 * k2 * (tk * tl * (-intrinsic_(0, 0) - pow(l1, 2) / intrinsic_(0, 0)) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) + l1 * l2 * tk * tl * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) / intrinsic_(0, 0)) / intrinsic_(0, 0));
//             hess_private[tid](1, 2) += weight * (-k1 * (-pow(tk, 2) * (-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) - k1 * k2 * pow(tk, 2) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) / intrinsic_(0, 0)) - k1 * (tk * tl * (-intrinsic_(0, 0) - pow(l1, 2) / intrinsic_(0, 0)) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2) + l1 * l2 * tk * tl * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) / intrinsic_(0, 0)) + k2 * (pow(tk, 2) * (-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) + k1 * k2 * pow(tk, 2) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) / intrinsic_(0, 0)) + k2 * (-tk * tl * (-intrinsic_(0, 0) - pow(l1, 2) / intrinsic_(0, 0)) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5) - l1 * l2 * tk * tl * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) / intrinsic_(0, 0)));
//             // hessian(2, 0) += (intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (-k1 * pow(tk, 2) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) - k2 * pow(tk, 2) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2)) + (intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1)) * (l1 * tk * tl * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) + l2 * tk * tl * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2)) + k1 * k2 * (k1 * pow(tk, 2) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) + k2 * pow(tk, 2) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5)) / intrinsic_(1, 1) + k1 * k2 * (-l1 * tk * tl * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) - l2 * tk * tl * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5)) / intrinsic_(1, 1);
//             // hessian(2, 1) += (-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (k1 * pow(tk, 2) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) + k2 * pow(tk, 2) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5)) + (-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) * (-l1 * tk * tl * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) - l2 * tk * tl * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5)) - k1 * k2 * (-k1 * pow(tk, 2) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) - k2 * pow(tk, 2) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2)) / intrinsic_(0, 0) - k1 * k2 * (l1 * tk * tl * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) + l2 * tk * tl * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2)) / intrinsic_(0, 0);
//             hess_private[tid](2, 2) += weight * (-k1 * (-k1 * pow(tk, 2) * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) - k2 * pow(tk, 2) * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2)) - k1 * (l1 * tk * tl * (-(-k2 + l2) * (-0.25 * k2 + 0.25 * l2) + 0.5) + l2 * tk * tl * (-0.25 * k1 + 0.25 * l1) * (-k2 + l2)) + k2 * (k1 * pow(tk, 2) * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) + k2 * pow(tk, 2) * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5)) + k2 * (-l1 * tk * tl * (-k1 + l1) * (-0.25 * k2 + 0.25 * l2) - l2 * tk * tl * (-(-k1 + l1) * (-0.25 * k1 + 0.25 * l1) + 0.5)));
//           }
//         }
//       }
//       // 归约
//       for (int t = 0; t < num_threads; ++t)
//       {
//         gradient += grad_private[t];
//         if (!use_prev)
//           hessian += hess_private[t];
//       }
//       double num = (width_ + 2 * padding_) * (height_ + 2 * padding_);
//       gradient /= num;
//       if (!use_prev)
//       {
//         hessian(1, 0) = hessian(0, 1);
//         hessian(2, 0) = hessian(0, 2);
//         hessian(2, 1) = hessian(1, 2);
//         hessian.array() *= (2.0 / num);
//       }
//     }
//     /**
//      * @brief 计算CMax函数对像素的一阶梯度
//      */
//     void iwe_gradient(const EigenMat &iwe, double &loss, EigenMat &grad, EigenMat &hess) const
//     {
//       // CMax函数
//       loss = -(iwe.array() - iwe.mean()).square().mean();
//       grad = -2 * (iwe.array() - iwe.mean()).matrix();
//       hess = -2 * Eigen::MatrixXd::Ones(iwe.rows(), iwe.cols());
//       // Ours
//       // loss = -((0.9416 * iwe.array() + (iwe.array() + 0.1).log()).mean());
//       // grad = 0.9416 + 1 / (iwe.array() + 0.1);
//       // hess = -1 / (iwe.array() + 0.1).square();
//     }
//     /**
//      * @brief 计算运动参数对翘曲事件点的梯度
//      * @param parameters 运动参数（角轴表示法）
//      * @param xs, ys 翘曲后的事件点（2D坐标），注意，这里的坐标是减去u0 v0后的坐标
//      * @param grad_warp 输出的梯度矩阵，大小为2x3
//      */
//     void warp_gradient(const Vec3 &parameters, const Vec3 &point, double dt, Eigen::Matrix<double, 2, 3> &grad_warp, Eigen::Matrix3d &hess_warp) const
//     {
//       double X_Z = point(0) / point(2);
//       double Y_Z = point(1) / point(2);
//       grad_warp << intrinsic_(0, 0) * X_Z * Y_Z, -intrinsic_(0, 0) * (1 + X_Z * X_Z), intrinsic_(0, 0) * Y_Z,
//           intrinsic_(1, 1) * (1 + Y_Z * Y_Z), -intrinsic_(1, 1) * X_Z * Y_Z, -intrinsic_(1, 1) * X_Z;
//       grad_warp *= dt;
//       hess_warp = grad_warp.row(0).transpose() * grad_warp.row(1) + grad_warp.row(1).transpose() * grad_warp.row(0);
//     }
//     void assign_vec(const Vec3 &vec, std::vector<EigenMat> &mat_vec, int x, int y) const
//     {
//       for (int i = 0; i < 3; ++i)
//       {
//         mat_vec[i](y, x) += vec(i);
//       }
//     }
//     void assign_mat(const Eigen::Matrix3d &mat, std::vector<EigenMat> &mat_vec, int x, int y) const
//     {
//       for (int i = 0; i < 3; ++i)
//       {
//         for (int j = 0; j < 3; ++j)
//         {
//           mat_vec[i * 3 + j](y, x) += mat(i, j);
//         }
//       }
//     }
//     void compute_gradient(const Vec3 &parameters, double &loss, Vec3 &gradient, Eigen::Matrix3d &hessian)
//     {
//       auto blur_eigen_mat = [&](EigenMat &mat)
//       {
//         // 注意：Eigen是列优先，所以rows是Eigen的cols，cols是Eigen的rows
//         cv::Mat cv_mat(mat.cols(), mat.rows(), CV_64F, mat.data());
//         // 使用与 compute_loss 中相同的模糊参数 (Size(5,5), sigma=1.0)
//         // 这一步对于模拟第一种方法的"Radius Search"至关重要
//         cv::GaussianBlur(cv_mat, cv_mat, cv::Size(5, 5), 1.0);
//       };
//       Eigen::Matrix<double, 3, 1> delta_points_T, delta_second_points_T;
//       Eigen::Matrix<double, 3, 1> points_T;
//       // 创建翘曲事件图像
//       int pad_width = width_ + 2 * padding_;
//       int pad_height = height_ + 2 * padding_;
//       warped_image_.setZero();
//       // 初始化梯度和海森矩阵图像
//       for (int i = 0; i < 3; ++i)
//       {
//         iwe_grad_[i].setZero();
//       }
//       for (int i = 0; i < 9; ++i)
//       {
//         iwe_hess_[i].setZero();
//       }
//       double warped_x, warped_y, event_x, event_y;
//       Eigen::Matrix<double, 2, 3> grad_warp;
//       Eigen::Matrix3d hess_warp;

//       // 提取内参，避免在循环中重复访问成员变量
//       const double fx = intrinsic_(0, 0);
//       const double cx = intrinsic_(0, 2);
//       const double fy = intrinsic_(1, 1);
//       const double cy = intrinsic_(1, 2);
//       for (int i = 0; i < n_points_; ++i)
//       {
//         auto point_ = points_[i];
//         // Taylor 二阶展开: points * skew_matrix * skew_matrix
//         delta_points_T(0) = -parameters[2] * point_(1) + parameters[1] * point_(2);
//         delta_points_T(1) = parameters[2] * point_(0) - parameters[0] * point_(2);
//         delta_points_T(2) = -parameters[1] * point_(0) + parameters[0] * point_(1);

//         // delta_second_points_T(0) = -parameters[2] * delta_points_T(1) + parameters[1] * delta_points_T(2);
//         // delta_second_points_T(1) = parameters[2] * delta_points_T(0) - parameters[0] * delta_points_T(2);
//         // delta_second_points_T(2) = -parameters[1] * delta_points_T(0) + parameters[0] * delta_points_T(1);
//         // 计算 warp 后的 3D 点
//         double dt = delta_time_[i];
//         // double dt_sq = 0.5 * dt * dt;
//         points_T(0) = point_(0) + delta_points_T(0) * dt;
//         points_T(1) = point_(1) + delta_points_T(1) * dt;
//         points_T(2) = point_(2) + delta_points_T(2) * dt;

//         // 投影到 2D
//         warped_x = points_T(0) / points_T(2) * fx + cx + padding_;
//         warped_y = points_T(1) / points_T(2) * fy + cy + padding_;
//         // 使用双线性插值将当前事件翘曲至图像
//         int x = static_cast<int>(floor(warped_x));
//         int y = static_cast<int>(floor(warped_y));
//         double dx = warped_x - x;
//         double dy = warped_y - y;

//         if (x >= 0 && x < pad_width - 1 && y >= 0 && y < pad_height - 1)
//         {
//           double w4 = dx * dy;
//           double w3 = (1.0 - dx) * dy;
//           double w2 = dx * (1.0 - dy);
//           double w1 = (1.0 - dx) * (1.0 - dy);

//           warped_image_(y, x) += w1;
//           warped_image_(y, x + 1) += w2;
//           warped_image_(y + 1, x) += w3;
//           warped_image_(y + 1, x + 1) += w4;
//           // 计算梯度
//           warp_gradient(parameters, point_, dt, grad_warp, hess_warp);
//           assign_vec((dy - 1) * grad_warp.row(0) + (dx - 1) * grad_warp.row(1), iwe_grad_, x, y);
//           assign_vec((1 - dy) * grad_warp.row(0) + (-dx) * grad_warp.row(1), iwe_grad_, x + 1, y);
//           assign_vec((-dy) * grad_warp.row(0) + (1 - dx) * grad_warp.row(1), iwe_grad_, x, y + 1);
//           assign_vec(dy * grad_warp.row(0) + dx * grad_warp.row(1), iwe_grad_, x + 1, y + 1);
//           // 计算海森矩阵
//           assign_mat(hess_warp, iwe_hess_, x, y);
//           assign_mat(-hess_warp, iwe_hess_, x + 1, y);
//           assign_mat(-hess_warp, iwe_hess_, x, y + 1);
//           assign_mat(hess_warp, iwe_hess_, x + 1, y + 1);
//         }
//       }
//       blur_eigen_mat(warped_image_);

// // 2. 模糊雅可比图 (最重要的改动!)
// // 这实际上是在计算：参数微扰动会导致"模糊后的图像"发生怎样的变化
// // 没有这一步，梯度就是针对"尖锐"图像的，与 Loss 不匹配
// #pragma omp parallel for
//       for (int k = 0; k < 3; ++k)
//       {
//         blur_eigen_mat(iwe_grad_[k]);
//       }

// // 3. 模糊海森图
// #pragma omp parallel for
//       for (int k = 0; k < 9; ++k)
//       {
//         blur_eigen_mat(iwe_hess_[k]);
//       }
//       // 计算最终的梯度与海森矩阵
//       iwe_gradient(warped_image_, loss, pixel_grad_, pixel_hess_);
// // 计算总梯度，各个像素梯度的加权和
// // gradient 等于将 pixel_grad 与 iwe_grad_ 第一个通道按像素位置对应相乘后求和
// #pragma omp parallel for
//       for (int i = 0; i < 3; ++i)
//       {
//         gradient(i) = (pixel_grad_.array() * iwe_grad_[i].array()).sum();
//       }
// #pragma omp parallel for
//       for (int i = 0; i < 3; ++i)
//       {
//         for (int j = 0; j < 3; ++j)
//         {
//           hessian(i, j) = (pixel_grad_.array() * iwe_hess_[3 * i + j].array()).sum() + (pixel_hess_.array() * iwe_grad_[i].array() * iwe_grad_[j].array()).sum();
//         }
//       }
//     }
//     void compute_gradient(const Vec3 &parameters, double &loss, Vec3 &gradient, Eigen::Matrix3d &hessian, bool use_fast)
//     {
//       const int pad_width = width_ + 2 * padding_;
//       const int pad_height = height_ + 2 * padding_;
//       // 使用 vector 保证内存连续，初始化为 0
//       iwe_data_.setZero();
//       Vec3 delta_points_T, delta_second_points_T, points_T;

//       double warped_x, warped_y, event_x, event_y;
//       Eigen::Matrix<double, 2, 3> grad_warp;
//       Eigen::Matrix3d hess_warp;

//       // 提取内参，避免在循环中重复访问成员变量
//       const double fx = intrinsic_(0, 0);
//       const double cx = intrinsic_(0, 2);
//       const double fy = intrinsic_(1, 1);
//       const double cy = intrinsic_(1, 2);
//       for (int i = 0; i < n_points_; ++i)
//       {
//         auto point_ = points_[i];
//         // Taylor 二阶展开: points * skew_matrix * skew_matrix
//         delta_points_T(0) = -parameters[2] * point_(1) + parameters[1] * point_(2);
//         delta_points_T(1) = parameters[2] * point_(0) - parameters[0] * point_(2);
//         delta_points_T(2) = -parameters[1] * point_(0) + parameters[0] * point_(1);

//         // delta_second_points_T(0) = -parameters[2] * delta_points_T(1) + parameters[1] * delta_points_T(2);
//         // delta_second_points_T(1) = parameters[2] * delta_points_T(0) - parameters[0] * delta_points_T(2);
//         // delta_second_points_T(2) = -parameters[1] * delta_points_T(0) + parameters[0] * delta_points_T(1);
//         // 计算 warp 后的 3D 点
//         double dt = delta_time_[i];
//         // double dt_sq = 0.5 * dt * dt;
//         points_T(0) = point_(0) + delta_points_T(0) * dt;
//         points_T(1) = point_(1) + delta_points_T(1) * dt;
//         points_T(2) = point_(2) + delta_points_T(2) * dt;

//         // 投影到 2D
//         warped_x = points_T(0) / points_T(2) * fx + cx + padding_;
//         warped_y = points_T(1) / points_T(2) * fy + cy + padding_;
//         // 使用双线性插值将当前事件翘曲至图像
//         int x = static_cast<int>(floor(warped_x));
//         int y = static_cast<int>(floor(warped_y));
//         double dx = warped_x - x;
//         double dy = warped_y - y;

//         if (x >= 0 && x < pad_width - 1 && y >= 0 && y < pad_height - 1)
//         {
//           double dx = warped_x - x;
//           double dy = warped_y - y;
//           double dx_inv = 1.0 - dx;
//           double dy_inv = 1.0 - dy;
//           // 计算梯度
//           warp_gradient(parameters, point_, dt, grad_warp, hess_warp);
//           // 注意：如果 grad_warp 是 RowMajor，data布局不同。
//           // 假设 grad_warp 是默认的 Matrix<double, 2, 3> (ColMajor):
//           // data: [r0c0, r1c0, r0c1, r1c1, r0c2, r1c2]
//           // 手动获取值比用 Eigen 接口更快
//           double g00 = grad_warp(0, 0), g01 = grad_warp(0, 1), g02 = grad_warp(0, 2);
//           double g10 = grad_warp(1, 0), g11 = grad_warp(1, 1), g12 = grad_warp(1, 2);
//           // 计算海森矩阵
//           double h00 = hess_warp(0, 0), h01 = hess_warp(0, 1), h02 = hess_warp(0, 2);
//           double h10 = hess_warp(1, 0), h11 = hess_warp(1, 1), h12 = hess_warp(1, 2);
//           double h20 = hess_warp(2, 0), h21 = hess_warp(2, 1), h22 = hess_warp(2, 2);
//           // Lambda: 更新单个像素的 13 个通道
//           auto update_pixel = [&](int offset, double w, double c1, double c2, bool is_positive)
//           {
//             // 直接获取该像素在 Eigen 矩阵中的行指针
//             // iwe_data_ 是 RowMajor，所以 row(offset) 的数据是连续的
//             double *ptr = iwe_data_.row(offset).data();

//             // [0]: Image
//             ptr[0] += w;
//             // [1-3]: Gradient
//             ptr[1] += c1 * g00 + c2 * g10;
//             ptr[2] += c1 * g01 + c2 * g11;
//             ptr[3] += c1 * g02 + c2 * g12;
//             // [4-12]: Hessian
//             if (is_positive)
//             {
//               ptr[4] += h00;
//               ptr[5] += h01;
//               ptr[6] += h02;
//               ptr[7] += h10;
//               ptr[8] += h11;
//               ptr[9] += h12;
//               ptr[10] += h20;
//               ptr[11] += h21;
//               ptr[12] += h22;
//             }
//             else
//             {
//               ptr[4] -= h00;
//               ptr[5] -= h01;
//               ptr[6] -= h02;
//               ptr[7] -= h10;
//               ptr[8] -= h11;
//               ptr[9] -= h12;
//               ptr[10] -= h20;
//               ptr[11] -= h21;
//               ptr[12] -= h22;
//             }
//           };
//           int pixel_idx = y * pad_width + x;
//           update_pixel(pixel_idx, dx_inv * dy_inv, -dy_inv, -dx_inv, true);     // TL
//           update_pixel(pixel_idx + 1, dx * dy_inv, dy_inv, -dx, false);         // TR
//           update_pixel(pixel_idx + pad_width, dx_inv * dy, -dy, dx_inv, false); // BL
//           update_pixel(pixel_idx + pad_width + 1, dx * dy, dy, dx, true);       // BR
//         }
//       }
//       cv::Mat cv_view(pad_height, pad_width, CV_MAKETYPE(CV_64F, 13), iwe_data_.data());
//       cv::GaussianBlur(cv_view, cv_view, cv::Size(5, 5), 1.0);

//       // 计算最终的梯度与海森矩阵
//       iwe_gradient(iwe_data_.col(0), loss, pixel_grad_, pixel_hess_);
//       // 计算总梯度，各个像素梯度的加权和
//       // gradient 等于将 pixel_grad 与 iwe_grad_ 第一个通道按像素位置对应相乘后求和
//       for (int i = 0; i < 3; ++i)
//       {
//         gradient(i) = (pixel_grad_.array() * iwe_data_.col(1 + i).array()).sum();
//       }
//       for (int i = 0; i < 3; ++i)
//       {
//         for (int j = 0; j < 3; ++j)
//         {
//           hessian(i, j) = (pixel_grad_.array() * iwe_data_.col(4 + i * 3 + j).array()).sum() + (pixel_hess_.array() * iwe_data_.col(1 + i).array() * iwe_data_.col(1 + j).array()).sum();
//         }
//       }
//     }
//     void MainOptimize()
//     {
//       // 使用牛顿法进行优化
//       const int max_iterations = 40;
//       const double tolerance = 1e-6;
//       Eigen::Matrix3d hessian;
//       Vec3 gradient;
//       Eigen::Matrix3d D;
//       double best_loss = std::numeric_limits<double>::max();
//       double loss = 0.0, prev_loss = best_loss;
//       Vec3 best_params = est_angleAxis_;
//       for (int iter = 0; iter < max_iterations; ++iter)
//       {
//         // compute_loss(est_angleAxis_, loss);
//         compute_gradient(est_angleAxis_, loss, gradient, hessian);
//         // compute_gradient(est_angleAxis_, loss, gradient, hessian, true);

//         // 添加阻尼项以确保Hessian矩阵是正定的
//         Eigen::Matrix3d damped_hessian = hessian;

//         // 求解线性系统
//         Eigen::Vector3d step = damped_hessian.ldlt().solve(-gradient);
//         est_angleAxis_ += step;
//         // 检查收敛性
//         if (step.norm() < tolerance || std::abs(prev_loss - loss) < tolerance)
//         {
//           break;
//         }
//         // 更新参数
//         if (best_loss > loss)
//         {
//           best_loss = loss;
//           best_params = est_angleAxis_;
//         }
//         prev_loss = loss;
// #ifdef DEBUG
//         std::cout << "Iteration " << iter << ": Loss = " << loss << ", Step Norm = " << step.norm() << std::endl;
//         std::cout << "Estimated Parameters: " << est_angleAxis_.transpose() << std::endl;
// #endif
//       }
//       // 计算最终损失
//       est_angleAxis_ = best_params;
//       final_loss_ = best_loss;
//       // compute_loss(est_angleAxis_, final_loss_);
//     }
//   };
//   // 继承 FirstOrderFunction，用于通用函数优化
//   class CMaxFunction : public ceres::FirstOrderFunction
//   {
//   private:
//     std::vector<Vec3> points_;
//     int width_, height_;
//     Eigen::Matrix3d intrinsic_;
//     std::vector<double> delta_time_;
//     std::vector<int> polarity_;

//   public:
//     CMaxFunction(const std::vector<Vec3> &points,
//                  const std::vector<double> &delta_time,
//                  const std::vector<int> &polarity,
//                  int width, int height,
//                  const Eigen::Matrix3d &K)
//         : points_(points),
//           delta_time_(delta_time),
//           intrinsic_(K),
//           width_(width),
//           height_(height),
//           polarity_(polarity)
//     {
//     }
//     virtual ~CMaxFunction() = default;
//     virtual bool Evaluate(const double *parameters,
//                           double *cost,
//                           double *gradient) const override
//     {
//       int n_points_ = points_.size();
//       int padding_ = 100;
//       Eigen::Matrix<double, 3, 1> delta_points_T, delta_second_points_T;
//       Eigen::Matrix<double, 3, 1> points_T;
//       // 创建翘曲事件图像
//       cv::Mat warped_image = cv::Mat::zeros(height_ + 2 * padding_, width_ + 2 * padding_, CV_64F);

//       // 构建二维PCL点云
//       pcl::PointCloud<pcl::PointXY>::Ptr cloud(new pcl::PointCloud<pcl::PointXY>());
//       cloud->resize(n_points_);
//       for (int i = 0; i < n_points_; ++i)
//       {
//         auto point_ = points_[i];
//         // Taylor 二阶展开: points * skew_matrix * skew_matrix
//         delta_points_T(0) = -parameters[2] * point_(1) + parameters[1] * point_(2);
//         delta_points_T(1) = parameters[2] * point_(0) - parameters[0] * point_(2);
//         delta_points_T(2) = -parameters[1] * point_(0) + parameters[0] * point_(1);

//         // delta_second_points_T(0) = -parameters[2] * delta_points_T(1) + parameters[1] * delta_points_T(2);
//         // delta_second_points_T(1) = parameters[2] * delta_points_T(0) - parameters[0] * delta_points_T(2);
//         // delta_second_points_T(2) = -parameters[1] * delta_points_T(0) + parameters[0] * delta_points_T(1);
//         // 计算 warp 后的 3D 点
//         double dt_early = delta_time_[i];
//         // double dt_sq = 0.5 * dt_early * dt_early;
//         // points_T(0) = point_(0) + delta_points_T(0) * dt_early + delta_second_points_T(0) * dt_sq;
//         // points_T(1) = point_(1) + delta_points_T(1) * dt_early + delta_second_points_T(1) * dt_sq;
//         // points_T(2) = point_(2) + delta_points_T(2) * dt_early + delta_second_points_T(2) * dt_sq;

//         points_T(0) = point_(0) + delta_points_T(0) * dt_early;
//         points_T(1) = point_(1) + delta_points_T(1) * dt_early;
//         points_T(2) = point_(2) + delta_points_T(2) * dt_early;
//         // 投影到 2D
//         (*cloud)[i].x = points_T(0) / points_T(2) * intrinsic_(0, 0) + intrinsic_(0, 2);
//         (*cloud)[i].y = points_T(1) / points_T(2) * intrinsic_(1, 1) + intrinsic_(1, 2);
//         // 使用双线性插值将当前事件翘曲至图像
//         int x = static_cast<int>((*cloud)[i].x);
//         int y = static_cast<int>((*cloud)[i].y);
//         double dx = (*cloud)[i].x - x;
//         double dy = (*cloud)[i].y - y;

//         if (x >= -padding_ && x < width_ + padding_ - 1 && y >= -padding_ && y < height_ + padding_ - 1)
//         {
//           double w4 = dx * dy;
//           double w3 = (1.0 - dx) * dy;
//           double w2 = dx * (1.0 - dy);
//           double w1 = (1.0 - dx) * (1.0 - dy);

//           warped_image.at<double>(y + padding_, x + padding_) += w1;
//           warped_image.at<double>(y + padding_, x + 1 + padding_) += w2;
//           warped_image.at<double>(y + padding_ + 1, x + padding_) += w3;
//           warped_image.at<double>(y + padding_ + 1, x + 1 + padding_) += w4;
//         }
//       }
//       // 平滑翘曲图像
//       cv::GaussianBlur(warped_image, warped_image, cv::Size(3, 3), 1.0);
//       // 计算图像梯度
//       cv::Mat grad_x, grad_y;
//       // cv::Sobel(warped_image, grad_x, CV_64F, 1, 0, 3);
//       // cv::Sobel(warped_image, grad_y, CV_64F, 0, 1, 3);
//       // 计算残差（方差）
//       cv::Scalar mean, stddev;
//       cv::meanStdDev(warped_image, mean, stddev);
//       double var = stddev[0] * stddev[0];
//       cost[0] = -var;

//       // 构建KD树
//       pcl::KdTreeFLANN<pcl::PointXY> kdtree;
//       kdtree.setInputCloud(cloud);

//       const float radius = 4.0f;
//       gradient[0] = 0.0;
//       gradient[1] = 0.0;
//       gradient[2] = 0.0;

//       // 局部累加器
//       int num_threads = omp_get_max_threads();
//       std::vector<Vec3> grad_private(num_threads, Vec3::Zero());

// #pragma omp parallel for
//       for (int k = 0; k < n_points_; ++k)
//       {
//         int tid = omp_get_thread_num();
//         std::vector<int> neighbor_indices;
//         std::vector<float> neighbor_dists;

//         pcl::PointXY search_point = (*cloud)[k];
//         auto k1 = (*cloud)[k].x - intrinsic_(0, 2);
//         auto k2 = (*cloud)[k].y - intrinsic_(1, 2);
//         // 半径搜索
//         kdtree.radiusSearch(search_point, radius, neighbor_indices, neighbor_dists);

//         for (size_t idx = 0; idx < neighbor_indices.size(); ++idx)
//         {
//           int l = neighbor_indices[idx];
//           if (l == k)
//             continue;
//           float dist2 = neighbor_dists[idx];
//           double weight = 0.079577 * std::exp(-dist2 * dist2 / 4.0) * delta_time_[k];
//           auto l1 = (*cloud)[l].x - intrinsic_(0, 2);
//           auto l2 = (*cloud)[l].y - intrinsic_(1, 2);
//           grad_private[tid][0] += weight * ((intrinsic_(1, 1) + k2 * k2 / intrinsic_(1, 1)) * (-k2 + l2) + k1 * k2 * (-k1 + l1) / intrinsic_(1, 1));
//           grad_private[tid][1] += weight * ((-intrinsic_(0, 0) - k1 * k1 / intrinsic_(0, 0)) * (-k1 + l1) - k1 * k2 * (-k2 + l2) / intrinsic_(0, 0));
//           grad_private[tid][2] += weight * (-k1 * (-k2 + l2) + k2 * (-k1 + l1));
//         }
//       }
//       for (int t = 0; t < num_threads; ++t)
//       {
//         gradient[0] += grad_private[t][0];
//         gradient[1] += grad_private[t][1];
//         gradient[2] += grad_private[t][2];
//       }
//       // #pragma omp parallel for
//       //       for (int k = 0; k < n_points_; ++k)
//       //       {
//       //         double x = (*cloud)[k].x;
//       //         double y = (*cloud)[k].y;
//       //         if (x < 0 || x >= width_ || y < 0 || y >= height_)
//       //           continue;
//       //         double dx = grad_x.at<double>(static_cast<int>(y), static_cast<int>(x)) * 2 * delta_time_[k];
//       //         double dy = grad_y.at<double>(static_cast<int>(y), static_cast<int>(x)) * 2 * delta_time_[k];
//       //         auto k1 = x - intrinsic_(0, 2);
//       //         auto k2 = y - intrinsic_(1, 2);
//       //         gradient[0] += dx * k1 * k2 / intrinsic_(1, 1) + dy * (intrinsic_(1, 1) + pow(k2, 2) / intrinsic_(1, 1));
//       //         gradient[1] += dx * (-intrinsic_(0, 0) - pow(k1, 2) / intrinsic_(0, 0)) - dy * k1 * k2 / intrinsic_(0, 0);
//       //         gradient[2] += dx * k2 - dy * k1;
//       //       }
//       gradient[0] /= n_points_;
//       gradient[1] /= n_points_;
//       gradient[2] /= n_points_;
//       return true;
//     }

//     virtual int NumParameters() const override { return 3; }
//   };

//   CMaxEstimator::CMaxEstimator(double fx, double fy, double cx, double cy,
//                                int width, int height, const Config &config)
//       : width_(width), height_(height), count_(1)
//   {
//     // 初始化相机内参矩阵
//     K_ << fx, 0, cx,
//         0, fy, cy,
//         0, 0, 1;

//     // 初始化估计状态
//     est_angleAxis_.setZero();
//     last_est_var_.setZero();
//   }

//   void CMaxEstimator::projectEventsTo3D(const std::vector<Vec2> &xy,
//                                         std::vector<Vec3> &coord_3d,
//                                         const std::vector<double> &depth)
//   {
//     int n = xy.size();
//     coord_3d.resize(n);

//     double fx = K_(0, 0), fy = K_(1, 1);
//     double cx = K_(0, 2), cy = K_(1, 2);

//     for (int i = 0; i < n; ++i)
//     {
//       double x = (xy[i](0) - cx) / fx;
//       double y = (xy[i](1) - cy) / fy;
//       coord_3d[i] = Vec3(x, y, depth.empty() ? 1.0 : depth[i]);
//       coord_3d[i].normalize();
//     }
//   }

//   void CMaxEstimator::getSampledIndices(std::vector<int> &indices, int count,
//                                         double start_ratio, double end_ratio)
//   {
//     int start_idx = static_cast<int>(eventBundle_.size * start_ratio);
//     int end_idx = static_cast<int>(eventBundle_.size * end_ratio);
//     int range = end_idx - start_idx;

//     if (range <= 0 || count <= 0)
//     {
//       indices.clear();
//       return;
//     }

//     count = std::min(count, range);
//     indices.resize(count);

//     // 随机采样
//     std::vector<int> all_indices(range);
//     std::iota(all_indices.begin(), all_indices.end(), start_idx);

//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::shuffle(all_indices.begin(), all_indices.end(), gen);

//     std::copy(all_indices.begin(), all_indices.begin() + count, indices.begin());
//     std::sort(indices.begin(), indices.end());
//   }

//   double CMaxEstimator::optimizeMotionIpopt()
//   {
//     CMaxOptimizer optimizer(
//         event_undis_Bundle_.coord_3d,
//         eventBundle_.time_delta,
//         eventBundle_.polarity,
//         width_, height_,
//         K_, est_angleAxis_);

//     // 运行优化
//     optimizer.MainOptimize();

//     // 更新结果
//     optimizer.getEstimatedParams(est_angleAxis_);
// #ifdef DEBUG
//     std::cout << "Final estimated angular velocity: " << est_angleAxis_.transpose() << std::endl;
// #endif
//     return optimizer.getFinalLoss();
//   }
//   double CMaxEstimator::optimizeMotion()
//   {
//     // ... 参数初始化 ...
//     double angleAxis[3] = {est_angleAxis_(0), est_angleAxis_(1), est_angleAxis_(2)};

//     // 1. 构建 GradientProblem
//     ceres::GradientProblem problem(new CMaxFunction(
//         event_undis_Bundle_.coord_3d,
//         eventBundle_.time_delta,
//         eventBundle_.polarity,
//         width_, height_,
//         K_));

//     // 2. 配置 Solver
//     ceres::GradientProblemSolver::Options options;
//     options.minimizer_progress_to_stdout = true;
//     options.line_search_direction_type = ceres::BFGS; // 推荐用于通用优化
//     options.max_num_iterations = 20;

//     ceres::GradientProblemSolver::Summary summary;

//     // 3. 求解
//     ceres::Solve(options, problem, angleAxis, &summary);

//     std::cout << summary.FullReport() << std::endl;

//     // ... 更新结果 ...
//     est_angleAxis_(0) = angleAxis[0];
//     est_angleAxis_(1) = angleAxis[1];
//     est_angleAxis_(2) = angleAxis[2];
// #ifdef DEBUG
//     std::cout << "Final estimated angular velocity: " << est_angleAxis_.transpose() << std::endl;
// #endif
//     return summary.final_cost;
//   }

//   RegistrationOutput CMaxEstimator::run(const std::vector<Vec2> &xy,
//                                         const std::vector<double> &timestamps,
//                                         const std::vector<int> &polarity,
//                                         const std::vector<double> &depth)
//   {
//     RegistrationOutput result;
//     result.timespan = {timestamps.front(), timestamps.back()};

//     if (xy.empty())
//     {
//       result.twist.setZero();
//       result.stats.loss = 0;
//       result.stats.num_iterations = 0;
//       return result;
//     }

//     // 初始化事件数据
//     eventBundle_.size = xy.size();
//     eventBundle_.coord = xy;
//     eventBundle_.polarity = polarity;
//     eventBundle_.first_tstamp = timestamps.front();
//     eventBundle_.last_tstamp = timestamps.back();

//     // 计算相对时间
//     eventBundle_.time_delta.resize(eventBundle_.size);
//     for (int i = 0; i < eventBundle_.size; ++i)
//     {
//       eventBundle_.time_delta[i] = timestamps[i] - timestamps.front();
//     }

//     // 投影到 3D
//     event_undis_Bundle_.size = eventBundle_.size;
//     projectEventsTo3D(xy, event_undis_Bundle_.coord_3d, depth);

//     std::cout << '[' << count_ << "] " << "CMax: Processing "
//               << eventBundle_.size << " events, duration: "
//               << (eventBundle_.last_tstamp - eventBundle_.first_tstamp) << "s" << std::endl;

//     // 运行优化
//     result.stats.time_taken = omp_get_wtime();
//     result.stats.loss = optimizeMotionIpopt();
//     result.stats.time_taken = omp_get_wtime() - result.stats.time_taken;
//     std::cout << "CMax: Optimization took " << result.stats.time_taken << " seconds." << std::endl;

//     // 填充结果
//     result.twist = est_angleAxis_;
//     result.stats.num_iterations = count_;

//     std::cout << '[' << count_ << "] " << "CMax: Estimated angular velocity: "
//               << est_angleAxis_.transpose()
//               << ", best loss: " << result.stats.loss << std::endl;
//     count_++;
//     return result;
//   }

// } // namespace spatiotemporal_vo