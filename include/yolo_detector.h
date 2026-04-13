// 核心推理类
#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

class YoloDetector {
public:
    // 构造函数：负责加载 ONNX 模型
    YoloDetector(const std::string& modelPath, bool isGPU = false);

    // 核心处理函数（我们之后会在 main 里面循环调用它）
    void process(cv::Mat& frame);

private:
    // --- 核心三步曲 ---
    // 1. 预处理：把图像缩放并转为模型需要的张量
    cv::Mat preprocess(const cv::Mat& inputImg);

    // 2. 推理：执行前向传播 (之后写)
    // 3. 后处理：解析结果并画框 (之后写)

    // --- 辅助工具函数 ---
    // Letterbox 缩放：保持比例缩放并填充黑边（这是目标检测最关键的预处理！）
    cv::Mat letterbox(const cv::Mat& source, const cv::Size& targetSize);

    // --- ONNX Runtime 相关的环境变量 ---
    Ort::Env env;
    Ort::Session* session;
    Ort::MemoryInfo memoryInfo;

    // 模型需要的输入尺寸 (YOLOv8 默认通常是 640x640)
    cv::Size inputSize = cv::Size(640, 640);
};