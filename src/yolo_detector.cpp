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
// Process 函数
// ---------------------------------------------------------
void YoloDetector::process(cv::Mat& frame) {
    // ==========================================
    // 第一步：预处理 (Pre-processing)
    // ==========================================
    cv::Mat inputBlob = preprocess(frame);

    // ==========================================
    // 第二步：模型推理 (Inference)
    // ==========================================
    // 1. 定义输入的维度和数据大小
    std::vector<int64_t> inputDims = { 1, 3, 640, 640 };
    size_t inputTensorSize = 1 * 3 * 640 * 640;

    // 2. 将 OpenCV 的 Mat 数据“绑定”为 ONNX Runtime 认识的张量 (Tensor)
    // 注意：这里没有复制数据，只是用指针指了过去，所以速度极快！
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        (float*)inputBlob.data,
        inputTensorSize,
        inputDims.data(),
        inputDims.size()
    );

    // 3. 定义 YOLOv8 标准的输入输出节点名称
    const char* inputNames[] = { "images" };
    const char* outputNames[] = { "output0" };

    // 4. 执行核心运算！(这行代码运行期间，你的 CPU 会疯狂做矩阵乘法)
    std::vector<Ort::Value> outputTensors = session->Run(
        Ort::RunOptions{ nullptr },
        inputNames,
        &inputTensor,
        1,  // 我们有 1 个输入
        outputNames,
        1   // 我们期待 1 个输出
    );

    // 5. 获取输出结果的数据指针和维度信息
    float* outputData = outputTensors[0].GetTensorMutableData<float>();
    auto outputInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputDims = outputInfo.GetShape();

    // 打印测试一下
    std::cout << "推理完成! 模型输出的神秘矩阵尺寸: ";
    for (int i = 0; i < outputDims.size(); ++i) {
        std::cout << outputDims[i] << " ";
    }
    std::cout << std::endl;

    // (第三步后处理我们稍后再写...)
}