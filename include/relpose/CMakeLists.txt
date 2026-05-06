cmake_minimum_required(VERSION 3.10)
project(relpose LANGUAGES CXX)

# 第三方依赖
find_package(Eigen3 REQUIRED)

# 定义一个 header-only 接口库
add_library(relpose INTERFACE)

# 将当前目录设为公开头文件搜索路径
target_include_directories(relpose INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 链接 Eigen3
target_link_libraries(relpose INTERFACE
    Eigen3::Eigen
)

# 要求 C++11 或更高（用到了 std::isnan 等）
target_compile_features(relpose INTERFACE cxx_std_11)