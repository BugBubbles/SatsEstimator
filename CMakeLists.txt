cmake_minimum_required(VERSION 3.14)
project(spatiotemporal_vo_cpp)

add_definitions(-DDEBUG)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 依赖库查找 ---
# 1. PCL
find_package(PCL REQUIRED COMPONENTS common io registration kdtree search)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# 2. Sophus
find_package(Sophus REQUIRED)
find_package(Ceres REQUIRED)
# 3. Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# 4. OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# --- 包含目录 ---
include_directories(include)

# --- 源文件 ---
file(GLOB SOURCES "src/*.cpp")

# --- 可执行文件 ---
add_executable(${PROJECT_NAME} ${SOURCES})

# --- 链接库 ---
target_link_libraries(${PROJECT_NAME}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    Sophus::Sophus
    Ceres::ceres 
    Eigen3::Eigen
)