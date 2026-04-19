// 推理逻辑实现
#include "yolo_detector.h"

// ---------------------------------------------------------
// 1. 构造函数：加载模型
// ---------------------------------------------------------
YoloDetector::YoloDetector(const std::string& modelPath, bool isGPU)
    : env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8"),
    memoryInfo(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU))
{
    Ort::SessionOptions sessionOptions;
    // 设置使用多少个 CPU 线程来跑，设置 4 个通常比较合适
    sessionOptions.SetIntraOpNumThreads(4);

    // 在 Windows 下，ONNX 路径需要是宽字符 (wstring)
    std::wstring widestr = std::wstring(modelPath.begin(), modelPath.end());

    // 初始化 Session
    session = new Ort::Session(env, widestr.c_str(), sessionOptions);
    std::cout << "YOLO 模型加载成功!" << std::endl;
}

// ---------------------------------------------------------
// 2. Letterbox 算法：带黑边的等比例缩放
// ---------------------------------------------------------
cv::Mat YoloDetector::letterbox(const cv::Mat& source, const cv::Size& targetSize) {
    cv::Mat output;
    // 计算缩放比例，取宽和高中较小的那个缩放比，保证图片不变形
    float scale = std::min((float)targetSize.width / source.cols, (float)targetSize.height / source.rows);

    // 缩放后的真实尺寸
    int newWidth = std::round(source.cols * scale);
    int newHeight = std::round(source.rows * scale);

    // 先用 OpenCV 进行常规缩放
    cv::resize(source, output, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);

    // 计算需要填充的黑边大小
    int top = (targetSize.height - newHeight) / 2;
    int bottom = targetSize.height - newHeight - top;
    int left = (targetSize.width - newWidth) / 2;
    int right = targetSize.width - newWidth - left;

    // 使用 OpenCV 的 copyMakeBorder 填充黑边 (灰度值 114 是 YOLO 的默认背景色)
    cv::copyMakeBorder(output, output, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    return output;
}

// ---------------------------------------------------------
// 3. 预处理主函数
// ---------------------------------------------------------
cv::Mat YoloDetector::preprocess(const cv::Mat& inputImg) {
    // 步骤 A: 保持比例缩放并填充黑边到 640x640
    cv::Mat resizedImg = letterbox(inputImg, inputSize);

    // 步骤 B: 色彩空间转换 & 归一化 & 维度重排 (HWC -> CHW)
    // OpenCV 读取的图片是 BGR 格式，而 YOLO 模型需要 RGB 格式
    // 且需要把像素值从 0~255 除以 255 变成 0.0~1.0
    cv::Mat blob;
    cv::dnn::blobFromImage(resizedImg, blob, 1.0 / 255.0, inputSize, cv::Scalar(), true, false);

    return blob;
}

// ---------------------------------------------------------
// 4. 后处理主函数 (大浪淘沙)
// ---------------------------------------------------------
std::vector<Detection> YoloDetector::postprocess(float* outputData, const cv::Size& originalImageSize) {
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // === [1] 模型结构定义 ===
    int classes = 2;              // 类别数量
    int dimensions = 6;           // 张量维度
    int rows = 8400;              // YOLOv8 默认输出 8400 个预测框

    // === [2] 计算 Letterbox 逆向缩放系数与黑边偏移量 ===
    float rx = (float)inputSize.width / originalImageSize.width;
    float ry = (float)inputSize.height / originalImageSize.height;
    float r = std::min(rx, ry); // 真实的缩放比例 (保持长宽比)

    // 计算预处理时添加的黑边大小 (除以2是因为两边对称补黑边)
    float pad_w = (inputSize.width - originalImageSize.width * r) / 2.0f;
    float pad_h = (inputSize.height - originalImageSize.height * r) / 2.0f;

    // [调试专用]：用于记录全场最高分
    float absoluteMaxConf = 0.0f;

    // === [3] 遍历所有 8400 个预测结果 ===
    for (int i = 0; i < rows; ++i) {
        float maxClassConf = 0.0f;
        int maxClassId = 0;

        // 寻找当前框中概率最高的类别
        for (int c = 0; c < classes; ++c) {
            float conf = outputData[(c + 4) * rows + i];
            if (conf > maxClassConf) {
                maxClassConf = conf;
                maxClassId = c;
            }
        }

        // [调试专用]：更新全图最高置信度
        if (maxClassConf > absoluteMaxConf) {
            absoluteMaxConf = maxClassConf;
        }

        // 只有当预测概率大于我们在 .h 里设置的及格线时，才保留这个框
        if (maxClassConf > confThreshold) {
            float cx = outputData[0 * rows + i];
            float cy = outputData[1 * rows + i];
            float w = outputData[2 * rows + i];
            float h = outputData[3 * rows + i];

            // === [4] 核心坐标还原逻辑：扣除黑边，除以真实缩放比例 ===
            cx = (cx - pad_w) / r;
            cy = (cy - pad_h) / r;
            w = w / r;
            h = h / r;

            int left = std::round(cx - 0.5 * w);
            int top = std::round(cy - 0.5 * h);
            int width = std::round(w);
            int height = std::round(h);

            // === [5] 边界保护：防止框画到图片外面导致程序崩溃 ===
            left = std::max(0, std::min(left, originalImageSize.width - 1));
            top = std::max(0, std::min(top, originalImageSize.height - 1));
            width = std::max(1, std::min(width, originalImageSize.width - left));
            height = std::max(1, std::min(height, originalImageSize.height - top));

            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(maxClassConf);
            classIds.push_back(maxClassId);
        }
    }

    // === [6] 非极大值抑制 (NMS)，过滤重叠框 ===
    std::vector<int> nmsResult;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, nmsResult);

    std::vector<Detection> finalDetections;
    for (int idx : nmsResult) {
        Detection d;
        d.box = boxes[idx];
        d.confidence = confidences[idx];
        d.class_id = classIds[idx];
        finalDetections.push_back(d);
    }

    return finalDetections;
}


// ---------------------------------------------------------
// 5. 更新后的完整 Process 函数
// ---------------------------------------------------------
std::vector<Detection> YoloDetector::process(cv::Mat& frame) {
    // 1. 预处理
    cv::Mat inputBlob = preprocess(frame);

    // 2. 推理
    std::vector<int64_t> inputDims = { 1, 3, 640, 640 };
    size_t inputTensorSize = 1 * 3 * 640 * 640;
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, (float*)inputBlob.data, inputTensorSize, inputDims.data(), inputDims.size()
    );

    const char* inputNames[] = { "images" };
    const char* outputNames[] = { "output0" };

    std::vector<Ort::Value> outputTensors = session->Run(
        Ort::RunOptions{ nullptr }, inputNames, &inputTensor, 1, outputNames, 1
    );

    // ?????? 新增：获取并打印 ONNX 输出张量的真实形状
    Ort::TensorTypeAndShapeInfo shapeInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputShape = shapeInfo.GetShape();
    std::cout << "【形状侦测】模型输出的张量维度为: ";
    for (size_t i = 0; i < outputShape.size(); i++) {
        std::cout << outputShape[i] << " ";
    }
    std::cout << std::endl;
    // ?????? 结束

    float* outputData = outputTensors[0].GetTensorMutableData<float>();

    // 3. 后处理（提取结果并返回）
    cv::Size originalImageSize(frame.cols, frame.rows);
    return postprocess(outputData, originalImageSize);
}