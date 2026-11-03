import h5py
import numpy as np
import yaml
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
data = h5py.File(os.path.join(input_dir, f"{sequence}{ind}_data.hdf5"))
gt = h5py.File(os.path.join(input_dir, f"{sequence}{ind}_gt.hdf5"))
velo_gt = np.load(os.path.join(input_dir, f"{sequence}{ind}_odom.npz"))
with open(os.path.join(input_dir, f"{sequence}_calib", f"camchain-imucam-{sequence}.yaml"), "r") as f:
    calib = yaml.safe_load(f)
imu = np.array(data["davis"]["left"]["imu"])
imu_ts = np.array(data["davis"]["left"]["imu_ts"])
events = np.array(data["davis"]["left"]["events"])
depth_image_ts = np.array(gt["davis"]["left"]["depth_image_raw_ts"])
velo = velo_gt["lin_vel"]
angle = velo_gt["ang_vel"]
velo_ts = velo_gt["timestamps"]

events_ts = events[:, 2]
t0 = events_ts[0]
events_ts -= t0
depth_image_ts -= t0
imu_ts -= t0
velo_ts -= t0

os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "groundtruth.txt"), "w") as f:
    gt_out = np.concatenate([velo_ts[:, None], velo, angle], axis=1)
    np.savetxt(f, gt_out, fmt="%.6f", header="t(s) vx(m/s) vy(m/s) vz(m/s) wx(rad/s) wy(rad/s) wz(rad/s)")
