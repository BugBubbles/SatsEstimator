import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import argparse
import os

args = argparse.ArgumentParser()
args.add_argument("-s", "--sequence", type=str, required=True)
args.add_argument("--input-root", required=True, help="Root directory containing the downloaded dataset")
args.add_argument("--output-root", required=True, help="Root directory for the converted TXT sequences")
args = args.parse_args()
sequence = args.sequence
file_dir = os.path.join(args.input_root, sequence)
save_dir = args.output_root
# 相机参数
W = 200
H = 200
distortion_coeffs = [0, 0, 0, 0, 0]
u0 = 100
v0 = 100
fx = 241
fy = 241
for file in tqdm(os.listdir(file_dir)):
    data = np.load(os.path.join(file_dir, file), allow_pickle=True)

    folder_name = file.removesuffix(".npz")
    if not os.path.exists(os.path.join(save_dir, folder_name)):
        os.makedirs(os.path.join(save_dir, folder_name))
    events = data['events']
    # 格式化事件流
    events = np.array([events['x'], events['y'], events['t']/1e6, events['p']]).T 

    timestamps = data['timestamps']
    traj = data['traj'] 
    phi_interp = interp1d(timestamps, traj[:, 6], fill_value="extrapolate")
    theta_interp = interp1d(timestamps, traj[:, 7], fill_value="extrapolate")
    range_meter = data['range_meter']
    depth_interp = interp1d(range_meter[:, 0], range_meter[:, 1], fill_value="extrapolate")

    # 针对每个事件流，分别计算其深度，使用平面模型
    phi = phi_interp(events[:, 2])
    theta = theta_interp(events[:, 2])
    depth = depth_interp(events[:, 2])
    xs = events[:, 0] - u0
    ys = events[:, 1] - v0
    d = depth * np.cos(theta) * np.cos(phi) / (-xs / fx * np.sin(theta) + ys / fy * np.cos(theta) * np.sin(phi) + np.cos(theta) * np.cos(phi))

    np.savetxt(os.path.join(save_dir, folder_name, "events.txt"), np.stack((events[:, 2], events[:, 0], events[:, 1], events[:, 3], d)).T, fmt="%.6f %d %d %d %.6f")
    with open(os.path.join(save_dir, folder_name, "calib.txt"), "w") as f:
        f.write(f"{fx} {fy} {u0} {v0} {' '.join(map(str, distortion_coeffs))}\n")
