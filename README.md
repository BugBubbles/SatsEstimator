# Spatial Accumulation and Temporal Statistics for Event Alignment and Ego-Motion Estimation

## Table of Contents

- [Overview](#overview)
- [Implemented Methods](#implemented-methods)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Installing Dependencies](#installing-dependencies)
  - [Building the Project](#building-the-project)
  - [CMake Options](#cmake-options)
- [Dataset Preparation](#dataset-preparation)
  - [ECD / DAVIS 240C Downloads](#ecd--davis-240c-downloads)
  - [MVSEC: HDF5 to TXT](#mvsec-hdf5-to-txt)
  - [ELOPE: NPZ to TXT](#elope-npz-to-txt)
- [Configuration](#configuration)
  - [Configuration File Format](#configuration-file-format)
  - [Configuration Fields](#configuration-fields)
- [Usage](#usage)
  - [Running Motion Estimation](#running-motion-estimation)
  - [Executable Reference](#executable-reference)
- [Output Format](#output-format)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

This codebase implements several event-camera-based ego-motion estimation methods that exploit the spatiotemporal structure of event streams. Event cameras asynchronously record per-pixel brightness changes with microsecond resolution, providing a rich signal for motion estimation. This project provides a unified C++ framework supporting:

- **Multiple motion models**: pure rotation (3-DOF), planar translation (3-DOF), and full 6-DOF rigid-body motion.
- **Multiple contrast objectives**: including the proposed Spatial Accumulation and Temporal Statistics (SATS) method, the classical contrast maximization (CMax), spatiotemporal point-plane association (STPPP), the patch-based event motion estimation (PEME), and spatiotemporal registration (STR).
- **Incremental estimation**: several methods additionally support incremental (frame-to-frame) motion estimation pipelines.
- **Adaptive event windowing**: an optional grid-based heuristic that dynamically selects the number of events per optimization window using a statistical outlier test on the spatial event rate.

All methods share a common data loading, calibration, and logging infrastructure, allowing straightforward benchmarking and comparison.

## Implemented Methods

| Method | Description | Motion Models |
|--------|-------------|---------------|
| **SATS** | Spatial Accumulation and Temporal Statistics — uses a logarithmic contrast function on the image of warped events (IWE) for smooth, well-conditioned optimization. *(Proposed)* | Rotation, Translation, 6-DOF, 6-DOF Incremental |
| **CMax** | Classical contrast maximization — maximizes the variance of the IWE. | Rotation, Translation, 6-DOF, 6-DOF Incremental |
| **STPPP** | Spatiotemporal point-to-plane association — aligns the spatiotemporal surface of events; optionally uses Boost `digamma`/`trigamma` functions. | Rotation, Translation, 6-DOF |
| **PEME** | Patch-based event motion estimation — uses a time-surface-based Ceres optimization with overlap ratio control for dynamic batching. | Rotation, Translation, 6-DOF |
| **STR** | Spatiotemporal registration — splits events temporally and registers the two halves via ICP on the spatiotemporal point cloud. | Rotation, Translation, 6-DOF |

## Prerequisites

The project requires the following libraries and tools:

| Dependency | Minimum Version | Purpose |
|------------|--------|---------|
| **CMake** | 3.14+ | Build system |
| **C++ Compiler** | C++17 support (GCC 7+ / Clang 5+) | Language standard |
| **Eigen3** | 3.3+ | Linear algebra |
| **OpenCV** | 3.4+ / 4.x | Image processing, undistortion |
| **Ceres Solver** | 2.0+ | Nonlinear least-squares (used by PEME) |
| **PCL** (Point Cloud Library) | 1.10+ | KD-tree and point cloud registration (used by STR) |
| **OpenMP** | — | Multi-threaded parallelization |
| **Boost** *(optional)* | 1.65+ | Special math functions for STPPP (`digamma`, `trigamma`) |

## Installation

### Installing Dependencies

#### Ubuntu / Debian

```bash
# Core build tools
sudo apt-get update
sudo apt-get install -y build-essential cmake git

# Eigen3
sudo apt-get install -y libeigen3-dev

# OpenCV
sudo apt-get install -y libopencv-dev

# Ceres Solver (and its dependencies)
sudo apt-get install -y libgoogle-glog-dev libgflags-dev libatlas-base-dev libsuitesparse-dev
sudo apt-get install -y libceres-dev

# PCL
sudo apt-get install -y libpcl-dev

# OpenMP (usually bundled with GCC; verify with `gcc -fopenmp`)

# (Optional) Boost — only needed if you want native digamma/trigamma in STPPP
sudo apt-get install -y libboost-all-dev
```

> **Note:** If `libceres-dev` is not available in your distribution, you can build Ceres from source. See [ceres-solver.org/installation](http://ceres-solver.org/installation.html).

#### Building Dependencies from Source (Advanced)

If prebuilt packages are unavailable or too old, build the key dependencies from source:

<details>
<summary><b>Eigen3 from source</b></summary>

```bash
git clone https://gitlab.com/libeigen/eigen.git -b 3.4
cd eigen && mkdir build && cd build
cmake .. && sudo make install
```
</details>

<details>
<summary><b>Ceres Solver from source</b></summary>

```bash
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver && mkdir build && cd build
cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
make -j$(nproc) && sudo make install
```
</details>

<details>
<summary><b>PCL from source</b></summary>

```bash
git clone https://github.com/PointCloudLibrary/pcl.git -b pcl-1.13.1
cd pcl && mkdir build && cd build
cmake .. -DBUILD_visualization=OFF -DBUILD_examples=OFF
make -j$(nproc) && sudo make install
```
</details>

### Building the Project

```bash
cd PolyEventEstimator

# Create build directory
mkdir -p build && cd build

# Configure (default: Release mode with all executables)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)
```

All executables will be placed in the `bin/` directory at the project root.

### CMake Options

The following CMake options can be passed at configuration time via `-D<OPTION>=ON/OFF`:

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TESTS` | `ON` | Build all method executables (CMax, SATS, STPPP, PEME, STR for all motion models). Set to `OFF` to build only the static library `event_camera_vo`. |
| `SEPARATE_POLARITY` | `OFF` | Process positive and negative polarity events independently, maintaining separate warped images and gradients. This doubles memory usage but may improve estimation in scenes with strong polarity asymmetry. |
| `ANALYTIC_HESSIAN` | `OFF` | **(Deprecated)** Use analytically derived Hessian matrices instead of Gauss–Newton approximation. Significantly increases computation cost and may degrade convergence. **Not recommended.** |
| `BOOST_AVAILABLE` | `OFF` | Enable Boost math library support. When `ON`, the STPPP method will use `boost::math::digamma` and `boost::math::trigamma` instead of built-in approximations. |
| `CMAKE_BUILD_TYPE` | *(empty)* | Standard CMake build type. Use `Release` for optimized builds (recommended) or `Debug` for development (enables `-DDEBUG` macro for verbose logging). |

#### Example Build Commands

```bash
# Optimized release build (recommended)
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Debug build with verbose output
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# Build with Boost support for STPPP
cmake .. -DCMAKE_BUILD_TYPE=Release -DBOOST_AVAILABLE=ON
make -j$(nproc)

# Build with separate polarity processing
cmake .. -DCMAKE_BUILD_TYPE=Release -DSEPARATE_POLARITY=ON
make -j$(nproc)

# Build only the static library (no executables)
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF
make -j$(nproc)
```

## Dataset Preparation

The code supports standard event camera datasets. Events should be stored in a text file with one event per line:

```
timestamp x y polarity [depth]
```

where:
- `timestamp` — event timestamp in seconds (double precision)
- `x`, `y` — pixel coordinates (double precision)
- `polarity` — event polarity (integer; positive values → +1, non-positive → −1)
- `depth` *(optional)* — per-event depth (used in 6-DOF estimation with known depth)

Lines starting with `#` or `%` are treated as comments and skipped.

### Camera Calibration File

A calibration file (`calib.txt` by default) should be placed alongside the event file in the sequence directory. The format is a single data line:

```
fx fy cx cy k1 k2 p1 p2 k3
```

where `fx`, `fy`, `cx`, `cy` are the camera intrinsics and `k1, k2, p1, p2, k3` are distortion coefficients (OpenCV convention).

### Supported Datasets

The provided configuration files cover the following public datasets:

| Dataset | Motion Models | Config Directory |
|---------|---------------|------------------|
| [ECD / DAVIS 240C](https://rpg.ifi.uzh.ch/davis_data.html) (poster_rotation, boxes_rotation, shapes_rotation, dynamic_rotation) | Rotation | `configs/rotation/` |
| [MVSEC](https://daniilidis-group.github.io/mvsec/) (indoor_flying, outdoor_day) | Translation, 6-DOF | `configs/translation/`, `configs/6dof/` |
| [ELOPE](https://zenodo.org/records/15421707) | 6-DOF | `configs/6dof/elope.yaml` |

### Directory Structure

Organize each sequence as follows:

```
<data_root>/
└── <sequence_name>/
    ├── events.txt          # Event stream file
    ├── calib.txt           # Camera calibration (fx fy cx cy k1 k2 p1 p2 k3)
    └── groundtruth.txt     # (Optional) Ground truth velocities for MVSEC/ELOPE
```

### ECD / DAVIS 240C Downloads

The [Event-Camera Dataset and Simulator (ECD) official page](https://rpg.ifi.uzh.ch/davis_data.html) provides the DAVIS 240C recordings used by the rotation configurations. Download the **Text (zip)** versions:

| Sequence | Download |
|----------|----------|
| `boxes_rotation` | [boxes_rotation.zip](https://rpg.ifi.uzh.ch/datasets/davis/boxes_rotation.zip) |
| `poster_rotation` | [poster_rotation.zip](https://rpg.ifi.uzh.ch/datasets/davis/poster_rotation.zip) |
| `shapes_rotation` | [shapes_rotation.zip](https://rpg.ifi.uzh.ch/datasets/davis/shapes_rotation.zip) |
| `dynamic_rotation` | [dynamic_rotation.zip](https://rpg.ifi.uzh.ch/datasets/davis/dynamic_rotation.zip) |

Extract each sequence under `/path/to/datasets/Davis_240C/`, keeping `events.txt`, `calib.txt`, and `imu.txt` together. Events and calibration already use the TXT formats supported by this project. Rotation evaluation uses the gyroscope columns of `imu.txt`; the dataset's `groundtruth.txt` contains poses. Set `paths.data_root` to your extracted dataset root in `configs/rotation/`.

### Conversion Scripts

The four scripts in `scripts/` convert the downloaded data into the TXT layout above. They require Python 3.9+ and the Python packages `numpy`, `h5py`, `opencv-python` (`cv2`), `scipy`, `tqdm`, and `PyYAML`. Run the commands below from the repository root using an environment with these dependencies available.

All `/path/to/...` paths in this repository are anonymous placeholders. Replace them with your own directories, including `paths.data_root` in the YAML configurations. The scripts accept `--input-root` and `--output-root`; they contain no machine-specific dataset paths.

### MVSEC: HDF5 to TXT

Download the matching `<sequence><index>_data.hdf5` and `<sequence><index>_gt.hdf5` files from the [official MVSEC download page](https://daniilidis-group.github.io/mvsec/download/) ([HDF5 folder](https://drive.google.com/drive/folders/1rwyRk26wtWeRgrAx_fgPc-ubUzTFThkV)). For velocity ground truth, also download the matching `_odom.npz` from the **Ground Truth Optical Flow Generation** section of that page, and obtain the scene calibration archive from its **Calibration** links.

For example, arrange the inputs for `indoor_flying1` as follows:

```text
/path/to/raw/MVSEC/
└── indoor_flying/
    ├── indoor_flying1_data.hdf5
    ├── indoor_flying1_gt.hdf5
    ├── indoor_flying1_odom.npz
    └── indoor_flying_calib/
        └── camchain-imucam-indoor_flying.yaml
```

Convert the events and then the velocity ground truth:

```bash
python scripts/mvsec_reader.py -s indoor_flying -i 1 \
  --input-root /path/to/raw/MVSEC \
  --output-root /path/to/datasets/MVSEC
python scripts/mvsec_gt.py -s indoor_flying -i 1 \
  --input-root /path/to/raw/MVSEC \
  --output-root /path/to/datasets/MVSEC
```

Use `-s outdoor_day -i 1` for `outdoor_day1`, and change the index for other recordings. `--input-root` contains the scene directory; `--output-root` receives one directory per recording, such as `indoor_flying1/`.

- `mvsec_reader.py` reads the left DAVIS events and raw depth maps, subtracts the first event timestamp, fills small depth holes, and interpolates depth between neighboring depth frames. It keeps events with valid depth greater than 4 m and writes `events.txt` with columns `t x y polarity depth`.
- `mvsec_gt.py` reads `timestamps`, `lin_vel`, and `ang_vel` from `_odom.npz`, uses the same timestamp origin, and writes `groundtruth.txt` with columns `t vx vy vz wx wy wz` (seconds, m/s, rad/s). The script also reads the HDF5 and calibration YAML inputs shown above.

The event converter retains its original append behavior: rerunning it against an existing `events.txt` appends duplicate data. Use a fresh output directory or remove that generated file before repeating conversion.

These two scripts do **not** generate `calib.txt`. Prepare it separately in the output recording directory using the nine-value format documented above. MVSEC's original DAVIS calibration uses an [equidistant model](https://daniilidis-group.github.io/mvsec/calibration_format/), whereas the current C++ loader uses OpenCV's standard distortion model; its four fisheye coefficients cannot simply be copied into the five standard coefficient slots. Calibration and event coordinates must use a consistent model; rectified events require the corresponding rectified intrinsics and zero distortion.

Then set `paths.data_root: /path/to/datasets/MVSEC` and `paths.sequences: indoor_flying1` in the chosen configuration.

### ELOPE: NPZ to TXT

Download `elope_dataset.zip` from the [official ELOPE dataset on Zenodo](https://zenodo.org/records/15421707); the [ESA challenge page](https://live.kelvins.esa.int/elope/) provides dataset context. Extract the archive and point `--input-root` to the directory containing `train/` and `test/`:

```text
/path/to/raw/ELOPE/
├── train/
│   ├── 0000.npz
│   └── ...
└── test/
    └── ...
```

Convert all NPZ recordings in the selected split:

```bash
python scripts/elope_reader.py -s train \
  --input-root /path/to/raw/ELOPE \
  --output-root /path/to/datasets/ELOPE
python scripts/elope_gt.py -s train \
  --input-root /path/to/raw/ELOPE \
  --output-root /path/to/datasets/ELOPE
```

Here `-s` selects an input subdirectory, such as `train` or `test`, rather than an individual recording. Each NPZ basename becomes an output sequence directory (for example, `0000/`).

- `elope_reader.py` converts event timestamps from microseconds to seconds, interpolates attitude and range measurements, and computes per-event depth using the existing planar surface model. It writes `events.txt` (`t x y polarity depth`) and `calib.txt`, using the script's fixed 200 × 200 camera, `fx = fy = 241`, `cx = cy = 100`, and zero distortion.
- `elope_gt.py` rotates trajectory linear velocities into the local frame and writes `groundtruth.txt` (`t vx vy vz wx wy wz`) and `depth.txt` (`t range`). Both scripts read the `range_meter` array in the NPZ input and preserve the original conversion formulas.

The official training split contains full trajectory ground truth. The test split masks position and linear velocity with NaNs, so use training recordings for evaluation against complete velocity ground truth. ELOPE output files are overwritten when conversion is repeated.

Set `paths.data_root: /path/to/datasets/ELOPE` in `configs/6dof/elope.yaml`, and set `paths.sequences` to the converted NPZ basename you want to run (for example, `0000`).

## Configuration

### Configuration File Format

Configuration files use a simple YAML-like syntax with `section:` headers and `key: value` pairs. A complete example:

```yaml
experiment_name: outdoor_day1
seed: 0

paths:
  data_root: /path/to/datasets/MVSEC     # Root of the dataset directory
  sequences: outdoor_day1                 # Sequence subdirectory name
  output_root: ./results                  # Output directory for result logs
  suffix: 6dof_outdoor_day1_sn100k_tn     # Suffix for result filenames
  event_file: events.txt                  # Event file name (default: events.txt)
  calib_file: calib.txt                   # Calibration file name (default: calib.txt)
  gt_file: groundtruth.txt               # Ground truth file name (optional)

data:
  time_window: 0.05          # Minimum time span (seconds) per event window
  event_count: 100000        # Maximum number of events per window
  unit_count: 5000           # Number of events per unit in adaptive splitting
  adaptive_split: false      # Enable adaptive window splitting (true/false)
  max_events: 200000000000   # Maximum total events to process
  grid_size: 20              # Grid cell size for spatial rate metric

camera:
  width: 346                 # Image width in pixels
  height: 260                # Image height in pixels

solver:
  max_iters: 200             # Maximum optimizer iterations per window
  convergence_thresh: 1e-6   # Convergence threshold on parameter change
  regularization: 0          # L2 regularization weight on motion parameters
  lr: 1.0                    # Initial learning rate (step size)
  decay: 0.98                # Learning rate decay factor per iteration
  epsilon_T: 0.05            # Temporal neighborhood threshold (STR method)
  trim_ratio: 1.0            # Trimming ratio (1.0 = no trimming)
```

### Configuration Fields

#### `paths` Section

| Key | Default | Description |
|-----|---------|-------------|
| `data_root` | `.` | Root directory containing all dataset sequences |
| `sequences` | `boxes_rotation` | Name of the sequence subdirectory |
| `event_file` | `events.txt` | Event data filename |
| `calib_file` | `calib.txt` | Camera calibration filename |
| `gt_file` | `groundtruth.txt` | Ground truth filename |
| `output_root` | `./result` | Directory for writing result files |
| `suffix` | `temp` | Suffix appended to result filenames for identification |

#### `data` Section

| Key | Default | Description |
|-----|---------|-------------|
| `time_window` | `0.05` | Minimum time span (s) for each event packet. Use `-1` to disable time-based splitting (events are split by count only). |
| `event_count` | `30000` | Maximum number of events per packet |
| `unit_count` | `2000` | Number of events per sampling unit in adaptive splitting |
| `adaptive_split` | `true` | When `true`, packets are split adaptively based on a statistical test on the spatial event rate. When `false`, packets are split purely by `event_count`. |
| `max_events` | `100000` | Upper bound on total events loaded |
| `grid_size` | `20` | Grid cell size (pixels) for computing the spatial dispersion metric used in adaptive splitting |

#### `camera` Section

| Key | Default | Description |
|-----|---------|-------------|
| `width` | `346` | Sensor width (pixels) |
| `height` | `260` | Sensor height (pixels) |

> Intrinsic parameters (`fx`, `fy`, `cx`, `cy`) and distortion coefficients are loaded from the calibration file, not from the YAML config.

#### `solver` Section

| Key | Default | Description |
|-----|---------|-------------|
| `max_iters` | `10` | Maximum number of Newton/GN iterations per window |
| `lr` | `1.0` | Initial step size for the optimizer |
| `decay` | `0.98` | Multiplicative decay applied to learning rate each iteration |
| `convergence_thresh` | `1e-6` | Convergence criterion on parameter update norm |
| `regularization` | `0.0` | L2 regularization weight on motion parameters |
| `epsilon_T` | `0.05` | Temporal neighborhood threshold (used by STR) |
| `trim_ratio` | `0.9` | Fraction of events retained when trimming is active (1.0 = keep all) |

## Usage

### Running Motion Estimation

All executables accept a single argument: the path to a YAML configuration file.

```bash
./bin/<executable> <config_file.yaml>
```

If no configuration file is provided, the executable defaults to `../configs/default.yaml`.

### Executable Reference

The following executables are built when `BUILD_TESTS=ON`:

#### Proposed Method (SATS)

| Executable | Motion Model | Description |
|------------|-------------|-------------|
| `SATS_rot` | Rotation (3-DOF) | SATS contrast for pure rotation |
| `SATS_trans` | Translation (3-DOF) | SATS contrast for planar translation |
| `SATS_6dof` | 6-DOF | SATS contrast for full rigid-body motion |
| `SATS_6dof_inc` | 6-DOF Incremental | Incremental SATS contrast estimation |

#### Baseline: Contrast Maximization (CMax)

| Executable | Motion Model | Description |
|------------|-------------|-------------|
| `CMax_rot` | Rotation (3-DOF) | Variance-based contrast maximization for rotation |
| `CMax_trans` | Translation (3-DOF) | Variance-based contrast maximization for translation |
| `CMax_6dof` | 6-DOF | Variance-based contrast maximization for 6-DOF |
| `CMax_6dof_inc` | 6-DOF Incremental | Incremental CMax estimation |

#### Baseline: STPPP

| Executable | Motion Model | Description |
|------------|-------------|-------------|
| `STPPP_rot` | Rotation (3-DOF) | Spatiotemporal point-to-plane for rotation |
| `STPPP_trans` | Translation (3-DOF) | Spatiotemporal point-to-plane for translation |
| `STPPP_6dof` | 6-DOF | Spatiotemporal point-to-plane for 6-DOF |

#### Baseline: PEME

| Executable | Motion Model | Description |
|------------|-------------|-------------|
| `PEME_rot` | Rotation (3-DOF) | Patch-based estimation for rotation |
| `PEME_trans` | Translation (3-DOF) | Patch-based estimation for translation |
| `PEME_6dof` | 6-DOF | Patch-based estimation for 6-DOF |
| `PEME_rot_db` | Rotation (3-DOF) | PEME with dynamic batching for rotation |
| `PEME_trans_db` | Translation (3-DOF) | PEME with dynamic batching for translation |
| `PEME_6dof_db` | 6-DOF | PEME with dynamic batching for 6-DOF |

#### Baseline: STR (Spatiotemporal Registration)

| Executable | Motion Model | Description |
|------------|-------------|-------------|
| `STR_rot` | Rotation (3-DOF) | ICP-based spatiotemporal registration for rotation |
| `STR_trans` | Translation (3-DOF) | ICP-based spatiotemporal registration for translation |
| `STR_6dof` | 6-DOF | ICP-based spatiotemporal registration for 6-DOF |

### Example Commands

```bash
# Run SATS (proposed) for rotation on the poster_rotation sequence
./bin/SATS_rot configs/rotation/poster_rotation.yaml

# Run CMax for 6-DOF on MVSEC outdoor_day1
./bin/CMax_6dof configs/6dof/outdoor_day1.yaml

# Run STPPP for rotation on shapes_rotation
./bin/STPPP_rot configs/rotation/shapes_rotation.yaml

# Run PEME with dynamic batching for 6-DOF
./bin/PEME_6dof_db configs/6dof/outdoor_day1.yaml

# Run STR for rotation
./bin/STR_rot configs/str/poster_rotation.yaml
```

## Output Format

Results are written to the directory specified by `paths.output_root` in the configuration file. Each result file is a plain-text file with one line per event window:

```
frame_id t_start t_end runtime param_1 param_2 param_3 [param_4 param_5 param_6]
```

| Column | Description |
|--------|-------------|
| `frame_id` | Sequential frame/window index |
| `t_start` | Start timestamp of the event window (seconds) |
| `t_end` | End timestamp of the event window (seconds) |
| `runtime` | Wall-clock computation time for this window (seconds) |
| `param_1..3` | Rotation-only: angular velocity (rad/s); translation-only or 6-DOF: linear velocity (m/s) |
| `param_4..6` | *(6-DOF only)* Angular velocity (rad/s) |

Result filenames follow the pattern: `<method>_<datetime>_<suffix>.txt`

## Evaluation

The four evaluators ported from Event-ST-PPP are available in `eval/`. They read the C++ result logs directly and run without file dialogs. NumPy and SciPy are required; Matplotlib is optional for saved plots.

```bash
# Pure rotation on ECD (use imu.txt, not pose groundtruth.txt)
python eval/eval.py -t rot \
  -g /path/to/datasets/Davis_240C/boxes_rotation/imu.txt \
  -e /path/to/results/rotation.txt

# Pure translation on MVSEC
python eval/eval.py -t trans \
  -g /path/to/datasets/MVSEC/indoor_flying1/groundtruth.txt \
  -e /path/to/results/translation.txt

# Six-DoF on MVSEC
python eval/eval.py -t 6dof \
  -g /path/to/datasets/MVSEC/indoor_flying1/groundtruth.txt \
  -e /path/to/results/6dof.txt

# ELOPE (also loads depth.txt beside groundtruth.txt)
python eval/eval.py -t elope \
  -g /path/to/datasets/ELOPE/0000/groundtruth.txt \
  -e /path/to/results/elope.txt
```

Add `--output ./results/metrics.json` to save metrics and `--plot ./results/velocities.png` to save a comparison plot. The original metric formulas, ECD IMU time offset, and historical normalization conventions are preserved; see [eval/README.md](eval/README.md) for input formats, units, time-overlap handling, and the ELOPE score. The adapted evaluation files retain their source license; see [eval/NOTICE.md](eval/NOTICE.md).

## Project Structure

```
PolyEventEstimator/
├── CMakeLists.txt                    # Root CMake configuration
├── README.md                         # This file
├── LICENSE                           # MIT License
├── configs/                          # YAML configuration files
│   ├── 6dof/                         #   6-DOF motion configs
│   ├── rotation/                     #   Pure rotation configs
│   ├── str/                          #   STR-specific configs
│   └── translation/                  #   Translation configs
├── include/
│   ├── methods/
│   │   ├── estimator.hpp             # Base MotionEstimator template class
│   │   ├── peme_estimator.hpp        # PEME estimator
│   │   ├── icp_estimator.hpp         # STR (ICP-based) estimator
│   │   ├── rotation/                 # Rotation-specific: CMax, SATS, STPPP
│   │   ├── translation/             # Translation-specific: CMax, SATS, STPPP
│   │   ├── 6dof/                    # 6-DOF: CMax, SATS, STPPP
│   │   └── 6dof_inc/               # 6-DOF incremental: CMax, SATS, STPPP
│   └── spatiotemporal_vo/
│       ├── common.hpp                # Common types: EventPacket, Vec2/3, etc.
│       ├── config.hpp                # Configuration classes
│       ├── data_loader.hpp           # Event data loading and preprocessing
│       └── logger.hpp                # Result logging
├── eval/                            # Rotation, translation, six-DoF, and ELOPE evaluation
├── scripts/                         # MVSEC and ELOPE conversion scripts
├── src/
│   ├── CMakeLists.txt                # Library build configuration
│   ├── config.cpp                    # Config file parsing
│   ├── data_loader.cpp              # Event I/O, undistortion, 3D projection
│   ├── logger.cpp                   # Result file writing
│   └── methods/
│       ├── peme_estimator.cpp        # PEME implementation
│       ├── peme_warp_utils.cpp       # PEME warping utilities
│       └── icp_estimator.cpp         # STR implementation
├── test/                             # Executable entry points
│   ├── CMakeLists.txt                # Test executable build configuration
│   ├── main_SATS_rot.cpp             # SATS rotation entry point
│   ├── main_CMax_6dof.cpp            # CMax 6-DOF entry point
│   ├── ...                           # (one main_*.cpp per executable)
├── bin/                              # Compiled executables (output)
│   └── results/                      # Result log files
└── build/                            # CMake build directory
```

## Citation

If you find this code useful in your research, please cite our paper:

```bibtex
@article{SatsBofeiChen2026,
  title     = {Spatial Accumulation and Temporal Statistics for Event Alignment and Ego-Motion Estimation},
  author    = {Bofei Chen, Jie Jiang, Dongyu Xie, and Kaifan Qiao},
  journal   = {IEEE Robotics and Automation Letters},
  year      = {2026}
}
```

## Acknowledgements

This work builds upon a number of excellent open-source libraries: [Eigen](https://eigen.tuxfamily.org/), [OpenCV](https://opencv.org/), [Ceres Solver](http://ceres-solver.org/), [PCL](https://pointclouds.org/), and [Boost](https://www.boost.org/). We gratefully acknowledge the developers of these projects.

We also thank the authors of the following projects for making their code available:

- [The Spatio-Temporal Poisson Point Process: A Simple Model for the Alignment of Event Camera Data](https://github.com/pbideau/Event-ST-PPP).
- [Progressive Spatio-Temporal Alignment for Efficient Event-Based Motion Estimation](https://github.com/huangxueyan/PEME).
- [Full-DoF Egomotion Estimation for Event Cameras Using Geometric Solvers](https://github.com/jizhaox/relpose-event).
