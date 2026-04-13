#include <iostream>
#include "yolo_detector.h"

int main() {
    // 1. 初始化检测器 
    // 【注意】使用绝对路径，并且一定要用正斜杠 "/"！
    std::string modelPath = "E:/CourseProjectC/YOLODeploy/models/yolov8n.onnx";

    std::cout << "正在加载模型: " << modelPath << std::endl;
    YoloDetector detector(modelPath);

    // 2. 模拟一张图片进行测试
    cv::Mat testFrame(1080, 1920, CV_8UC3, cv::Scalar(0, 255, 0));
    std::cout << "原始画面尺寸: " << testFrame.cols << "x" << testFrame.rows << std::endl;

    // 3. 将画面送入检测器
    detector.process(testFrame);

    return 0;
}