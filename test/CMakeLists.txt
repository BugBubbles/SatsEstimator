# 设置可执行文件输出目录为 build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# 定义一个函数来简化添加测试可执行文件
function(add_test_executable TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    target_link_libraries(${TARGET_NAME}
        PRIVATE
            event_camera_vo
            ${OpenCV_LIBS}
            ${CERES_LIBRARIES}
            Eigen3::Eigen
            ${OpenMP_CXX_LIBRARIES}
    )
    if(BOOST_AVAILABLE AND TARGET_NAME MATCHES ".*_stppp")
        target_link_libraries(${TARGET_NAME} PRIVATE ${Boost_LIBRARIES})
    endif()
    # 可执行文件会自动输出到 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} (build/bin)
endfunction()


# 自动添加所有 main_*.cpp 文件
file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main_*.cpp")
foreach(TEST_SOURCE ${TEST_SOURCES})
    # 获取文件名（不含路径和扩展名）
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    string(REPLACE "main_" "" EXEC_NAME ${TEST_NAME})

    if(NOT BUILD_STR AND TEST_NAME MATCHES "STR")
        continue()
    endif()

    # 避免重复添加已经手动添加的目标
    if(NOT TARGET ${EXEC_NAME})
        add_test_executable(${EXEC_NAME} ${TEST_SOURCE})
        message(STATUS "Added test executable: ${EXEC_NAME}")
    endif()
endforeach()