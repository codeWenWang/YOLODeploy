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

    // YOLOv8 的输出形状是 [1, 84, 8400]
    // 在内存中，它按通道存储：先是 8400 个 x，然后 8400 个 y...
    int dimensions = 84;
    int rows = 8400;

    // 为了计算还原比例，我们要知道原图是怎么缩放到 640 的
    float x_factor = (float)originalImageSize.width / inputSize.width;
    float y_factor = (float)originalImageSize.height / inputSize.height;
    // 真实的缩放比例是宽高中的较小值（Letterbox 规则）
    float scale = std::min(x_factor, y_factor);

    // 第一重过滤：遍历 8400 个网格点
    for (int i = 0; i < rows; ++i) {
        // 找这个框的 80 个类别中，概率最高的那一个
        float maxClassConf = 0.0f;
        int maxClassId = 0;

        for (int c = 0; c < 80; ++c) {
            // 定位到对应的内存地址 (通道 c+4, 偏移 i)
            float conf = outputData[(c + 4) * rows + i];
            if (conf > maxClassConf) {
                maxClassConf = conf;
                maxClassId = c;
            }
        }

        // 如果最高概率都达不到阈值，直接跳过（扔掉）
        if (maxClassConf > confThreshold) {
            // 提取该框的坐标 (前面 4 个通道)
            float cx = outputData[0 * rows + i];
            float cy = outputData[1 * rows + i];
            float w = outputData[2 * rows + i];
            float h = outputData[3 * rows + i];

            // 把中心点宽高，转成 OpenCV 喜欢的左上角宽高，并反推回原始像素尺寸
            int left = std::round((cx - 0.5 * w) * scale);
            int top = std::round((cy - 0.5 * h) * scale);
            int width = std::round(w * scale);
            int height = std::round(h * scale);

            // 保存入围者的数据
            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(maxClassConf);
            classIds.push_back(maxClassId);
        }
    }

    // 第二重过滤：执行 NMS (非极大值抑制)，剔除重叠的重复框
    std::vector<int> nmsResult;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, nmsResult);

    // 把最终幸存下来的框打包输出
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

    float* outputData = outputTensors[0].GetTensorMutableData<float>();

    // 3. 后处理（提取结果并返回）
    cv::Size originalImageSize(frame.cols, frame.rows);
    return postprocess(outputData, originalImageSize);
}