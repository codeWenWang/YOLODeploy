cmake_minimum_required(VERSION 3.15)
project(YOLODeploy)

set(CMAKE_CXX_STANDARD 17)

# ================= [核心修复] 给 CMake 指明 vcpkg 的安装路径 =================
list(APPEND CMAKE_PREFIX_PATH "E:/CourseProjectC/vcpkg/installed/x64-windows")
# =======================================================================

# [1] 新增：QT 自动化配置
# 开启这三个选项，CMake 就会自动帮你处理 QT 的特殊宏和 UI 文件
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# 寻找本地安装的 QT 库 (兼容 Qt5 和 Qt6)
find_package(QT NAMES Qt6 Qt5 COMPONENTS Widgets REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets REQUIRED)

# [2] 配置 OpenCV
find_package(OpenCV REQUIRED)

# [3] 配置 ONNX Runtime
set(ONNXRUNTIME_DIR "E:/CourseProjectC/onnxruntime-win-x64-1.24.4") 
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# 告诉编译器去哪里找头文件
include_directories(include)

# [4] 编译可执行文件：把 MainWindow 的文件加进来！
add_executable(${PROJECT_NAME} 
    src/main.cpp               
    src/yolo_detector.cpp
    src/mainwindow.cpp         
    src/mainwindow.h           
    src/mainwindow.ui          
)

# [5] 链接所有库 (注意最后加上了 Qt::Widgets)
target_link_libraries(${PROJECT_NAME} 
    ${OpenCV_LIBS} 
    onnxruntime 
    Qt${QT_VERSION_MAJOR}::Widgets
)