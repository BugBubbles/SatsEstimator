# Coupled Intensity and Timestamp Cues for Event Alignment and Ego-Motion Estimation

This repository contains the official C++ implementation of the paper:

> **Coupled Intensity and Timestamp Cues for Event Alignment and Ego-Motion Estimation**
>
> *European Conference on Computer Vision (ECCV)*

<p align="center">
  <img src="docs/overview.png" alt="Method Overview" width="80%"/>
</p>

## Table of Contents

- [Overview](#overview)
- [Implemented Methods](#implemented-methods)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Installing Dependencies](#installing-dependencies)
  - [Building the Project](#building-the-project)
  - [CMake Options](#cmake-options)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
  - [Configuration File Format](#configuration-file-format)
  - [Configuration Fields](#configuration-fields)
- [Usage](#usage)
  - [Running Motion Estimation](#running-motion-estimation)
  - [Executable Reference](#executable-reference)
- [Output Format](#output-format)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Overview

This codebase implements several event-camera-based ego-motion estimation methods that exploit the spatiotemporal structure of event streams. Event cameras asynchronously record per-pixel brightness changes with microsecond resolution, providing a rich signal for motion estimation. This project provides a unified C++ framework supporting:

- **Multiple motion models**: pure rotation (3-DOF), planar translation (3-DOF), and full 6-DOF rigid-body motion.
- **Multiple contrast objectives**: including the proposed CITEnomial contrast objective (CITE), the classical contrast maximization (CMax), spatiotemporal point-plane association (STPPP), the patch-based event motion estimation (PEME), and spatiotemporal registration (STR).
- **Incremental estimation**: several methods additionally support incremental (frame-to-frame) motion estimation pipelines.
- **Adaptive event windowing**: an optional grid-based heuristic that dynamically selects the number of events per optimization window using a statistical outlier test on the spatial event rate.

All methods share a common data loading, calibration, and logging infrastructure, allowing straightforward benchmarking and comparison.

## Implemented Methods

| Method | Description | Motion Models |
|--------|-------------|---------------|
| **CITE** | CITE contrast maximization — uses a logarithmic contrast function on the image of warped events (IWE) for smooth, well-conditioned optimization. *(Proposed)* | Rotation, Translation, 6-DOF, 6-DOF Incremental |
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
cd CITEEventEstimator

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
| `BUILD_TESTS` | `ON` | Build all method executables (CMax, CITE, STPPP, PEME, STR for all motion models). Set to `OFF` to build only the static library `event_camera_vo`. |
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
| [DAVIS 240C](https://rpg.ifi.uzh.ch/davis_data.html) (poster_rotation, boxes_rotation, shapes_rotation, dynamic_rotation) | Rotation | `configs/rotation/` |
| [MVSEC](https://daniilidis-group.github.io/mvsec/) (indoor_flying, outdoor_day) | Translation, 6-DOF | `configs/translation/`, `configs/6dof/` |
| [ELOPE](https://better-flow.github.io/ELOPE/) | 6-DOF | `configs/6dof/elope.yaml` |

### Directory Structure

Organize each sequence as follows:

```
<data_root>/
└── <sequence_name>/
    ├── events.txt          # Event stream file
    ├── calib.txt           # Camera calibration (fx fy cx cy k1 k2 p1 p2 k3)
    └── groundtruth.txt     # (Optional) Ground truth poses
```

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

#### Proposed Method (CITE)

| Executable | Motion Model | Description |
|------------|-------------|-------------|
| `CITE_rot` | Rotation (3-DOF) | CITE contrast for pure rotation |
| `CITE_trans` | Translation (3-DOF) | CITE contrast for planar translation |
| `CITE_6dof` | 6-DOF | CITE contrast for full rigid-body motion |
| `CITE_6dof_inc` | 6-DOF Incremental | Incremental CITE contrast estimation |

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
# Run CITE (proposed) for rotation on the poster_rotation sequence
./bin/CITE_rot configs/rotation/poster_rotation.yaml

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
| `param_1..3` | Estimated rotation parameters (angle-axis representation) |
| `param_4..6` | *(6-DOF only)* Estimated translation parameters |

Result filenames follow the pattern: `<method>_<datetime>_<suffix>.txt`

## Project Structure

```
CITEEventEstimator/
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
│   │   ├── rotation/                 # Rotation-specific: CMax, CITE, STPPP
│   │   ├── translation/             # Translation-specific: CMax, CITE, STPPP
│   │   ├── 6dof/                    # 6-DOF: CMax, CITE, STPPP
│   │   └── 6dof_inc/               # 6-DOF incremental: CMax, CITE, STPPP
│   └── spatiotemporal_vo/
│       ├── common.hpp                # Common types: EventPacket, Vec2/3, etc.
│       ├── config.hpp                # Configuration classes
│       ├── data_loader.hpp           # Event data loading and preprocessing
│       └── logger.hpp                # Result logging
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
│   ├── main_CITE_rot.cpp             # CITE rotation entry point
│   ├── main_CMax_6dof.cpp            # CMax 6-DOF entry point
│   ├── ...                           # (one main_*.cpp per executable)
├── bin/                              # Compiled executables (output)
│   └── results/                      # Result log files
└── build/                            # CMake build directory
```

## Citation

If you find this code useful in your research, please cite our paper:

```bibtex
@inproceedings{SatsBofeiChen2026,
  title     = {Spatial Accumulation and Temporal Statistics for Event Alignment and Ego-Motion Estimation},
  author    = {Bofei Chen, Jie Jiang, Dongyu Xie, and Kaifan Qiao},
  booktitle = {IEEE Robotics and Automation Letters},
  year      = {2026}
}
```

## Acknowledgements

This work builds upon a number of excellent open-source libraries: [Eigen](https://eigen.tuxfamily.org/), [OpenCV](https://opencv.org/), [Ceres Solver](http://ceres-solver.org/), [PCL](https://pointclouds.org/), and [Boost](https://www.boost.org/). We gratefully acknowledge the developers of these projects.
