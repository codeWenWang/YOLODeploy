cmake_minimum_required(VERSION 3.15)
project(YOLODeploy)

set(CMAKE_CXX_STANDARD 17)

# 1. 配置 OpenCV
find_package(OpenCV REQUIRED)

# 2. 配置 ONNX Runtime (请核对这里的路径是否与你之前的一致)
set(ONNXRUNTIME_DIR "E:/CourseProjectC/onnxruntime-win-x64-1.24.4") 
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# ================= 新增架构配置 =================

# 3. 告诉编译器去哪里找我们自己写的头文件 (.h)
include_directories(include)

# 4. 编译可执行文件时，带上 src 目录下的相对路径
add_executable(${PROJECT_NAME} 
    src/main.cpp
    src/yolo_detector.cpp
    # 以后你在 src 里写了新的 .cpp，都要加到这里
)

# ================================================

# 5. 链接库
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} onnxruntime)