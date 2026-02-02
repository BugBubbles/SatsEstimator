cmake_minimum_required(VERSION 3.14)
project(event_camera_vo_cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_definitions(-DEIGEN_MAX_ALIGN_BYTES=32)
if(DEBUG)
    add_definitions(-DDEBUG)
endif()
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
# --- 包含目录 ---
include_directories(include)

add_subdirectory(src)
if(BUILD_TESTS)
    add_subdirectory(test)
endif()
