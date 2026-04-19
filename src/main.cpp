#include <iostream>
#include <opencv2/opencv.hpp>
#include "yolo_detector.h"

// 1. 字典：标注的类别 (加上 Unknown 防止模型输出第2个类别时越界)
const std::vector<std::string> MY_CLASSES = { "Unknown", "Shared_Bike" };

int main() {
    // 2. 加载训练的新模型
    std::string modelPath = "E:/CourseProjectC/YOLODeploy/models/best.onnx";
    std::cout << "正在加载专属共享单车检测模型..." << std::endl;
    YoloDetector detector(modelPath);

    // 3. 读取校园测试照片
    std::string imagePath = "E:/CourseProjectC/YOLODeploy/data/test_pic1.jpg";
    cv::Mat frame = cv::imread(imagePath);
    if (frame.empty()) {
        std::cerr << "图片找不到，请检查路径！" << std::endl;
        return -1;
    }

    // 4. 执行核心推理
    std::cout << "开始推理..." << std::endl;
    std::vector<Detection> results = detector.process(frame);
    std::cout << "检测到 " << results.size() << " 辆共享单车！" << std::endl;

    // 5. 遍历结果并画框
    // ==========================================================
    // === 高清渲染重构：先高质量缩放图片，再画框！ ===
    // ==========================================================

    // (1) 计算缩放比例 (基于屏幕安全尺寸 1280x720)
    int maxWindowWidth = 1280;
    int maxWindowHeight = 720;
    double scaleX = (double)maxWindowWidth / frame.cols;
    double scaleY = (double)maxWindowHeight / frame.rows;
    double scale = std::min(scaleX, scaleY);
    if (scale > 1.0) scale = 1.0;

    // (2) 物理缩放原图 (使用 INTER_AREA 算法，这是图片缩小不模糊的关键！)
    cv::Mat displayFrame;
    int finalWidth = (int)(frame.cols * scale);
    int finalHeight = (int)(frame.rows * scale);
    cv::resize(frame, displayFrame, cv::Size(finalWidth, finalHeight), 0, 0, cv::INTER_AREA);

    // (3) 遍历结果，在【缩小后的高清图】上画框
    for (const auto& det : results) {
        std::string className = (det.class_id >= 0 && det.class_id < MY_CLASSES.size())
            ? MY_CLASSES[det.class_id] : "Unknown";
        std::string label = cv::format("%s: %.2f", className.c_str(), det.confidence);

        // --- 核心：把框的坐标也等比例缩小 ---
        int boxX = std::round(det.box.x * scale);
        int boxY = std::round(det.box.y * scale);
        int boxW = std::round(det.box.width * scale);
        int boxH = std::round(det.box.height * scale);
        cv::Rect scaledBox(boxX, boxY, boxW, boxH);

        // 字体大小现在可以固定一个舒适的值了，因为画布已经标准化了
        double fontScale = 0.30;
        int thickness = 1;

        // 画绿色目标框
        cv::rectangle(displayFrame, scaledBox, cv::Scalar(0, 255, 0), 2);

        // 绘制文字背景板和文字
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseLine);
        int top = std::max(scaledBox.y, labelSize.height + 10);

        cv::rectangle(displayFrame,
            cv::Point(scaledBox.x, top - labelSize.height - 5),
            cv::Point(scaledBox.x + labelSize.width + 5, top + baseLine + 5),
            cv::Scalar(0, 255, 0), cv::FILLED);

        cv::putText(displayFrame, label, cv::Point(scaledBox.x + 2, top),
            cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), thickness);
    }

    // (4) 显示最终的高清结果
    // 注意：不再使用 WINDOW_NORMAL，因为图片已经是完美尺寸了，直接用 AUTOSIZE 最清晰
    cv::namedWindow("Campus Smart Parking Detection", cv::WINDOW_AUTOSIZE);
    cv::imshow("Campus Smart Parking Detection", displayFrame);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}