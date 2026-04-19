#include "mainwindow.h"
#include "ui_mainwindow.h"

// QT 相关头文件
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QGraphicsPixmapItem>
#include <QGraphicsRectItem>
#include <QGraphicsTextItem>
#include <QFont>
#include <QColor>

// OpenCV 相关头文件
#include <opencv2/opencv.hpp>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , currentImagePath("")
{
    ui->setupUi(this);

    // 1. 初始化场景，并将其绑定到 QGraphicsView 控件上
    scene = new QGraphicsScene(this);
    ui->graphicsView->setScene(scene);

    // 2. 初始化你的专属 YOLO 检测器
    // 注意：请将模型路径替换为你实际的绝对路径
    std::string modelPath = "E:/CourseProjectC/YOLODeploy/models/best.onnx";
    try {
        detector = new YoloDetector(modelPath);
    }
    catch (...) {
        QMessageBox::critical(this, "错误", "模型加载失败，请检查路径！");
        // 实际开发中应更优雅地处理
    }

    // 3. 连接按钮的点击信号到对应的槽函数
    // 如果你在 .ui 文件中正确使用了 QPushButton 并设置了 objectName (如 btnUpload)，
    // QT 的 on_objectName_signalName 机制会自动连接，这里可以省略显式 connect。
}

MainWindow::~MainWindow()
{
    delete ui;
    delete scene;
    delete detector; // 别忘了释放检测器内存
}

void MainWindow::on_btnUpload_clicked()
{
    // 1. 弹出文件选择对话框
    QString path = QFileDialog::getOpenFileName(this, "选择测试图片", "", "Images (*.png *.jpg *.jpeg *.bmp)");

    if (path.isEmpty()) return;

    // 2. 保存路径，并显示原图
    currentImagePath = path;
    lastResults.clear(); // 清除上一次的结果
    displayImage(currentImagePath);

    // 3. 清空信息面板
    ui->textEditInfo->clear();
}

void MainWindow::displayImage(const QString& path)
{
    // 1. 清空场景中的所有元素（原图、旧框等）
    scene->clear();

    // 2. 加载图片并添加到场景中
    QPixmap pixmap(path);
    if (pixmap.isNull()) {
        QMessageBox::warning(this, "警告", "图片加载失败！");
        return;
    }
    QGraphicsPixmapItem* pixmapItem = scene->addPixmap(pixmap);

    // 3. 调整场景大小以适应图片，并让 GraphicsView 居中显示
    scene->setSceneRect(pixmap.rect());
    ui->graphicsView->fitInView(pixmapItem, Qt::KeepAspectRatio);
}

void MainWindow::on_btnDetect_clicked()
{
    if (currentImagePath.isEmpty() || !detector) {
        QMessageBox::warning(this, "警告", "请先上传图片，并确保模型已加载！");
        return;
    }

    // 1. 使用 OpenCV 读取图片 (BGR 格式，用于推理)
    // 新代码 (使用 toLocal8Bit 转换编码)兼容中文路径
    cv::Mat frame = cv::imread(currentImagePath.toLocal8Bit().constData());
    if (frame.empty()) {
        QMessageBox::warning(this, "警告", "OpenCV 读取图片失败！");
        return;
    }

    // 2. 执行核心推理，获取结果
    // 这里的 process 函数就是你昨天完善的、包含 1 6 8400 解析逻辑的那个！
    lastResults = detector->process(frame);

    // 3. 更新界面：在图上画框，并更新信息面板
    drawDetections();
    updateInfo(lastResults);
}

void MainWindow::drawDetections()
{
    // 1. 重新显示原图，覆盖旧框
    displayImage(currentImagePath);

    // 2. 遍历检测结果，利用 QT 的图形项动态画框和写字
    // 这里的坐标已经是 detector->process 返回的、基于原图分辨率的完美坐标了！
    for (const auto& det : lastResults) {
        // 定义颜色和字体（可以使用 QColor 和 QFont）
        QColor boxColor(0, 255, 0); // 绿色
        QFont labelFont("Arial", 12, QFont::Bold);

        // a. 画绿色目标框
        QGraphicsRectItem* rectItem = scene->addRect(
            det.box.x, det.box.y, det.box.width, det.box.height,
            QPen(boxColor, 3)); // 线宽为3

        // b. 准备标签文本
        std::string labelStr = cv::format("Shared_Bike: %.2f", det.confidence);

        // c. 画带有底板的文字标签
        QGraphicsTextItem* textItem = scene->addText(QString::fromStdString(labelStr), labelFont);
        textItem->setDefaultTextColor(Qt::black); // 黑色文字

        // 计算文字底板的位置
        textItem->setPos(det.box.x, det.box.y - textItem->boundingRect().height() - 5);

        // d. 为文字添加一个绿色的底板Item
        QGraphicsRectItem* labelBgItem = scene->addRect(textItem->boundingRect(), QPen(Qt::NoPen), QBrush(boxColor));
        labelBgItem->setPos(textItem->pos());
        // 调整层级，确保底板在文字下面，但在图片上面
        labelBgItem->setZValue(textItem->zValue() - 1);
    }
}

void MainWindow::updateInfo(const std::vector<Detection>& results)
{
    ui->textEditInfo->clear();

    // 1. 基本信息：识别到的数量
    QString info = QString("<h3>识别结果概要：</h3><p>检测到 <b>%1</b> 辆共享单车。</p>").arg(results.size());

    // 2. 详细信息：遍历每个目标，展示置信度
    info += "<h4>详细信息 (置信度)：</h4><ul>";
    for (size_t i = 0; i < results.size(); ++i) {
        info += QString("<li>目标 %1: %2%</li>").arg(i + 1).arg(results[i].confidence * 100, 0, 'f', 1);
    }
    info += "</ul>";

    // 3. 将富文本更新到信息面板
    ui->textEditInfo->setHtml(info);
}