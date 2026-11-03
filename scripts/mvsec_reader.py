import h5py
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import os

args = argparse.ArgumentParser()
args.add_argument("-s", "--sequence", type=str, required=True)
args.add_argument("-i", "--ind", type=int, required=True)
args.add_argument("--input-root", required=True, help="Root directory containing the downloaded dataset")
args.add_argument("--output-root", required=True, help="Root directory for the converted TXT sequences")
args = args.parse_args()

sequence = args.sequence
ind = args.ind
input_dir = os.path.join(args.input_root, sequence)
output_dir = os.path.join(args.output_root, f"{sequence}{ind}")

def nan_close_opt(depth_image, ksize):
    """
    对深度图中的 NaN 空洞执行闭合操作（膨胀-腐蚀），填补有效深度区域中间的小空洞。
    
    Args:
        depth_image: 包含 NaN 值的深度图，形状为 (H, W)
        ksize: 形态学操作的核大小（整数或元组）
    
    Returns:
        填充后的深度图
    """
    depth_image = depth_image.copy()
    
    # 创建有效值掩码（非 NaN 为 1，NaN 为 0）
    valid_mask = (~np.isnan(depth_image)).astype(np.uint8)
    
    # 创建形态学核
    if isinstance(ksize, int):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    
    # 对有效掩码执行闭合操作（膨胀后腐蚀），填充 NaN 空洞
    closed_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_CLOSE, kernel)
    
    # 找出需要填充的区域（原来是 NaN，闭合后被认为应该有效）
    fill_mask = (closed_mask == 1) & np.isnan(depth_image)
    
    if not np.any(fill_mask):
        return depth_image
    
    # 用局部有效值的均值填充这些空洞
    # 创建一个临时图像，将 NaN 替换为 0 便于卷积
    temp_depth = np.nan_to_num(depth_image, nan=0.0)
    
    # 计算局部和与局部有效计数
    kernel_avg = np.ones(kernel.shape, dtype=np.float64)
    local_sum = cv2.filter2D(temp_depth.astype(np.float64), -1, kernel_avg, borderType=cv2.BORDER_CONSTANT)
    local_count = cv2.filter2D(valid_mask.astype(np.float64), -1, kernel_avg, borderType=cv2.BORDER_CONSTANT)
    
    # 计算局部均值，避免除以零
    local_mean = np.zeros_like(depth_image)
    nonzero_count = local_count > 0
    local_mean[nonzero_count] = local_sum[nonzero_count] / local_count[nonzero_count]
    
    # 用局部均值填充空洞
    depth_image[fill_mask] = local_mean[fill_mask]
    
    return depth_image

data = h5py.File(os.path.join(input_dir, f"{sequence}{ind}_data.hdf5"))
gt = h5py.File(os.path.join(input_dir, f"{sequence}{ind}_gt.hdf5"))

events = np.array(data["davis"]["left"]["events"])
depth_image = np.array(gt["davis"]["left"]["depth_image_raw"])
depth_image_ts = np.array(gt["davis"]["left"]["depth_image_raw_ts"])
image_ts = np.array(data["davis"]["left"]["image_raw_ts"])
images = np.array(data["davis"]["left"]["image_raw"])


events_ts = events[:, 2]
t0 = events_ts[0]
events_ts -= t0
depth_image_ts -= t0
image_ts -= t0

os.makedirs(output_dir, exist_ok=True)
events_depth_file = open(os.path.join(output_dir, "events.txt"), "a")
idx = np.arange(len(depth_image_ts))
# os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
# for i in range(len(images)):
#     cv2.imwrite(os.path.join(output_dir, "images", f"{image_ts[i]:.6f}.png"), images[i])
# 仅在 depth 图像周边时间段内取事件
for idx1, idx2 in tqdm(zip(idx[:-1], idx[1:]), total=len(depth_image_ts)-1):
    depth_image_ts1 = depth_image_ts[idx1]
    depth_image_ts2 = depth_image_ts[idx2]
    depth_image1 = depth_image[idx1]
    depth_image2 = depth_image[idx2]
    ind = (events_ts >= depth_image_ts1) & (events_ts < depth_image_ts2)
    ti = events_ts[ind]
    x = events[ind, 0].astype(int)
    y = events[ind, 1].astype(int)
    p = events[ind, 3]
    alpha = (ti - depth_image_ts1) / (depth_image_ts2 - depth_image_ts1)

    depth_image1 = nan_close_opt(depth_image1, 5)
    depth_image2 = nan_close_opt(depth_image2, 5)
    # 融合深度图
    d1 = depth_image1[y, x]
    d2 = depth_image2[y, x]
    m1 = np.isnan(d1) == False
    m2 = np.isnan(d2) == False
    # 逐个融合，只取不为 nan 的深度值
    d = np.full_like(d1, np.nan)
    ## 两边都不为 nan，则加权平均
    d[m1 & m2] = d1[m1 & m2] * (1 - alpha[m1 & m2]) + d2[m1 & m2] * alpha[m1 & m2]
    ## 仅 d1 不为 nan
    d[m1 & (~m2)] = d1[m1 & (~m2)]
    ## 仅 d2 不为 nan
    d[(~m1) & m2] = d2[(~m1) & m2]
    ## 都为 nan，则保持为 nan
    mask = ~np.isnan(d)
    # 排除深度较小的事件点
    mask &= d > 4.0
    events_depth= np.stack([ti[mask], x[mask], y[mask], p[mask], d[mask]], axis=1)
    np.savetxt(events_depth_file, events_depth, fmt=["%.6f", "%d", "%d", "%d", "%.6f"])

events_depth_file.close()
