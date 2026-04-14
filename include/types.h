// 全局数据结构定义(如 BoundingBox 结构体)
#pragma once
#include <opencv2/opencv.hpp>

// 定义一个结构体，用来保存最终合格的检测结果
struct Detection {
    int class_id;       // 类别 (如 0代表"人")
    float confidence;   // 置信度 (如 0.85)
    cv::Rect box;       // 边界框的坐标 (X, Y, 宽, 高)
};

