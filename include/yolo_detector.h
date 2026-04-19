#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "types.h" // 引入数据结构

class YoloDetector {
public:
    YoloDetector(const std::string& modelPath, bool isGPU = false);

    // 把 process 的返回值改为 std::vector<Detection>
    std::vector<Detection> process(cv::Mat& frame);

private:
    cv::Mat preprocess(const cv::Mat& inputImg);
    cv::Mat letterbox(const cv::Mat& source, const cv::Size& targetSize);

    // ======== 新增的后处理函数 ========
    // 输入：模型吐出的 float 数组； 
    // originalImageSize: 原图尺寸 (用来把坐标从 640x640 还原回去)
    std::vector<Detection> postprocess(float* outputData, const cv::Size& originalImageSize);

    Ort::Env env;
    Ort::Session* session;
    Ort::MemoryInfo memoryInfo;

    cv::Size inputSize = cv::Size(640, 640);

    // 两个核心阈值
    float confThreshold = 0.4f; // 调整置信度用于测试
    float nmsThreshold = 0.20f;  // 重叠度大于nmsThreshold的框判定为重复
};