# 收集所有源文件
file(GLOB_RECURSE SPATIOTEMPORAL_VO_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

# 创建库
add_library(event_camera_vo STATIC ${SPATIOTEMPORAL_VO_SOURCES})

# 链接依赖库
target_link_libraries(event_camera_vo
    PUBLIC
      ${PCL_LIBRARIES}
      ${OpenCV_LIBS}
      Sophus::Sophus
      ${CERES_LIBRARIES}
      Eigen3::Eigen
)

# 设置头文件包含路径
target_include_directories(event_camera_vo
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)