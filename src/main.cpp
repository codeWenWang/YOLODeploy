#include <iostream>
#include "yolo_detector.h"

int main() {
    // 1. 初始化检测器
    // 【注意】：请把 yolov8n.onnx 放在和 YOLODeploy.exe 同一个文件夹里！
    YoloDetector detector("yolov8n.onnx");

    // 2. 用 OpenCV 生成一张 1920x1080 的绿色测试图像模拟摄像头输入
    cv::Mat testFrame(1080, 1920, CV_8UC3, cv::Scalar(0, 255, 0));
    std::cout << "原始画面尺寸: " << testFrame.cols << "x" << testFrame.rows << std::endl;

    // 3. 将画面送入检测器
    detector.process(testFrame);

    return 0;
}