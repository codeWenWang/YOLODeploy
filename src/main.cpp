#include <iostream>
#include <opencv2/opencv.hpp>
#include "yolo_detector.h"

const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};


int main() {
    // 1. 加载模型
    std::string modelPath = "E:/CourseProjectC/YOLODeploy/models/yolov8n.onnx";
    std::cout << "正在初始化 YOLOv8 检测器..." << std::endl;
    YoloDetector detector(modelPath);

    // 2. 读取一张真实的测试图片
    std::string imagePath = "E:/CourseProjectC/YOLODeploy/data/test2.jpg";
    cv::Mat frame = cv::imread(imagePath);

    if (frame.empty()) {
        std::cerr << "错误: 找不到图片，请检查路径 " << imagePath << std::endl;
        return -1;
    }

    std::cout << "图片读取成功! 尺寸: " << frame.cols << "x" << frame.rows << std::endl;

    // 3. 执行核心推理，获取检测结果
    std::vector<Detection> results = detector.process(frame);
    std::cout << "共检测到 " << results.size() << " 个目标。" << std::endl;

    // ==========================================
    // 4. 可视化渲染 (把框画到原图上)
    // ==========================================
    for (const auto& det : results) {
        // 画出矩形框 (绿色)
        cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);

        // 去字典里查这个 ID 对应的单词！
        std::string className = COCO_CLASSES[det.class_id];

        // 准备标签文字 (例如: person: 85%)
        std::string label = className + ": " + std::to_string((int)(det.confidence * 100)) + "%";

        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top = std::max(det.box.y, labelSize.height);

        cv::rectangle(frame, cv::Point(det.box.x, top - labelSize.height),
            cv::Point(det.box.x + labelSize.width, top + baseLine),
            cv::Scalar(0, 0, 0), cv::FILLED);

        cv::putText(frame, label, cv::Point(det.box.x, top),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    // 5. 显示最终结果图
    cv::imshow("YOLOv8 C++ Deployment Result", frame);

    // 等待用户按任意键后关闭窗口
    std::cout << "按任意键退出..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}