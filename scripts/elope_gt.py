import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation
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
for file in tqdm(os.listdir(file_dir)):
    data = np.load(os.path.join(file_dir, file), allow_pickle=True)

    folder_name = file.removesuffix(".npz")
    if not os.path.exists(os.path.join(save_dir, folder_name)):
        os.makedirs(os.path.join(save_dir, folder_name))
    traj = data['traj'] 
    v = traj[:, 3:6]
    phi, theta, psi = traj[:, 6:9].T
    depth = data["range_meter"]
    R = Rotation.from_euler("zyx", np.stack([psi, theta, phi], axis=1), degrees=False).as_matrix()
    # 将z轴方向调整为向上
    S = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    W = traj[:, 9:12]
    v = np.einsum("nij,nj->ni", R.transpose(0, 2, 1), v)
    t = data['timestamps']
    
    np.savetxt(os.path.join(save_dir, folder_name, "groundtruth.txt"), np.concatenate((t[:, None], v, W), axis=1), fmt="%.6f", delimiter=" ")
    np.savetxt(os.path.join(save_dir, folder_name, "depth.txt"), depth, fmt="%.6f", delimiter=" ")
