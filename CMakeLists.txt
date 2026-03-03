cmake_minimum_required(VERSION 3.14)
project(event_camera_vo_cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_definitions(-DEIGEN_MAX_ALIGN_BYTES=32)

# --- DEBUG模式 ---
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
endif()
# --- 是否构建测试 ---
option(BUILD_TESTS "Build [cmax,stppp,poly] test cases" ON)
# --- 是否将极性分开处理 ---
option(SEPARATE_POLARITY "Process positive and negative events separately" OFF)
if(SEPARATE_POLARITY)
    add_definitions(-DSEPARATE_POLARITY)
endif()
# --- 是否采用解析海森矩阵 ---
option(ANALYTIC_HESSIAN "Use analytic Hessian in ego-motion estimation" OFF)
if(ANALYTIC_HESSIAN)
# 添加警告，说明使用解析Hessian矩阵计算量较高，且收敛性会变差，已经是被deprecated的选项了
message(WARNING "Using analytic Hessian will significantly increase computational cost and may lead to worse convergence. This option is DEPRECATED and NOT recommended for use.")
    add_definitions(-DANALYTIC_HESSIAN)
endif()

# --- 是否使用Boost库 ---
option(BOOST_AVAILABLE "Boost library available for use. Optional for STPPP calculation." OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 依赖库查找 ---
# 1. PCL
find_package(PCL REQUIRED COMPONENTS common io registration kdtree search)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
# 3. Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# 4. OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 5. OpenMP
find_package(OpenMP REQUIRED)
if(BOOST_AVAILABLE)
    find_package(Boost REQUIRED)
    include_directories(${Boost_INCLUDE_DIRS})
    add_definitions(-DBOOST_AVAILABLE)
endif()
# --- 包含目录 ---
include_directories(include)

add_subdirectory(src)
if(BUILD_TESTS)
    add_subdirectory(test)
endif()
