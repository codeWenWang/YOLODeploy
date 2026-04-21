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

#pragma execution_character_set("utf-8")

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , currentImagePath("")
    , detector(nullptr) // 安全修复：初始化检测器为空指针
{
    ui->setupUi(this);

    // ================= [UI 美化核心代码] =================
    QString proDarkStyle = R"(
        /* 1. 主窗口：深邃的工业灰底色 */
        QMainWindow { background-color: #1E1E2E; }

        /* 2. 侧边栏文本和滑块 */
        QLabel {
            color: #CDD6F4;
            font-family: "Microsoft YaHei";
            font-size: 13px;
            font-weight: bold;
        }
        QSlider::groove:horizontal {
            border-radius: 4px;
            height: 8px;
            background: #313244;
        }
        QSlider::handle:horizontal {
            background: #89B4FA;
            width: 16px;
            margin: -4px 0;
            border-radius: 8px;
        }

        /* 3. 图片展示区：纯黑背景，取消边框，沉浸式看图 */
        QGraphicsView {
            background-color: #11111B;
            border: 1px solid #313244;
            border-radius: 8px;
        }

        /* 4. 专业级按钮操作区：科技蓝 */
        QPushButton {
            background-color: #89B4FA;
            color: #11111B;
            border: none;
            border-radius: 6px;
            padding: 10px 15px;
            font-family: "Microsoft YaHei";
            font-size: 14px;
            font-weight: bold;
            min-height: 25px;
        }
        QPushButton:hover { background-color: #B4BEFE; }
        QPushButton:pressed {
            background-color: #74C7EC;
            padding-top: 12px;
        }

        /* 5. 识别日志区：控制台风格代码框 */
        QTextEdit {
            background-color: #181825;
            color: #A6E3A1; /* 极客绿文字 */
            border: 1px solid #313244;
            border-radius: 6px;
            padding: 10px;
            font-family: "Consolas", "Microsoft YaHei"; 
            font-size: 13px;
        }
    )";
    this->setStyleSheet(proDarkStyle);

    ui->btnUpload->setCursor(Qt::PointingHandCursor);
    ui->btnDetect->setCursor(Qt::PointingHandCursor);
    ui->sliderConf->setCursor(Qt::PointingHandCursor);

    // 新增：初始化滑块范围 1~100，默认值 50 (代表 0.5 置信度)
    ui->sliderConf->setRange(1, 100);
    ui->sliderConf->setValue(50);
    // =====================================================

    // 1. 初始化场景，并将其绑定到 QGraphicsView 控件上
    scene = new QGraphicsScene(this);
    ui->graphicsView->setScene(scene);

    // 2. 初始化专属 YOLO 检测器
    std::string modelPath = "E:/CourseProjectC/YOLODeploy/models/best.onnx";
    try {
        detector = new YoloDetector(modelPath);
    }
    catch (...) {
        QMessageBox::critical(this, "错误", "模型加载失败，请检查路径！");
    }

    // 实时响应滑块拖动：只要滑块值改变，立马重新画框并更新日志！
    connect(ui->sliderConf, &QSlider::valueChanged, this, [=]() {
        if (!currentImagePath.isEmpty() && !lastResults.empty()) {
            drawDetections();
            updateInfo(lastResults);
        }
        });
}

MainWindow::~MainWindow()
{
    delete ui;
    delete scene;
    delete detector;
}

void MainWindow::on_btnUpload_clicked()
{
    QString path = QFileDialog::getOpenFileName(this, "选择测试图片", "", "Images (*.png *.jpg *.jpeg *.bmp)");
    if (path.isEmpty()) return;

    currentImagePath = path;
    lastResults.clear();
    displayImage(currentImagePath);
    ui->textEditInfo->clear();
}

void MainWindow::displayImage(const QString& path)
{
    scene->clear();
    QPixmap pixmap(path);
    if (pixmap.isNull()) {
        QMessageBox::warning(this, "警告", "图片加载失败！");
        return;
    }
    QGraphicsPixmapItem* pixmapItem = scene->addPixmap(pixmap);
    scene->setSceneRect(pixmap.rect());
    ui->graphicsView->fitInView(pixmapItem, Qt::KeepAspectRatio);
}

void MainWindow::on_btnDetect_clicked()
{
    if (currentImagePath.isEmpty() || !detector) {
        QMessageBox::warning(this, "警告", "请先上传图片，并确保模型已加载！");
        return;
    }

    cv::Mat frame = cv::imread(currentImagePath.toLocal8Bit().constData());
    if (frame.empty()) {
        QMessageBox::warning(this, "警告", "OpenCV 读取图片失败！");
        return;
    }

    lastResults = detector->process(frame);

    drawDetections();
    updateInfo(lastResults);
}

void MainWindow::drawDetections()
{
    // 1. 重新显示原图，覆盖旧框
    displayImage(currentImagePath);

    // 核心逻辑：获取当前滑块设定的阈值 (0.01 ~ 1.0)
    float threshold = ui->sliderConf->value() / 100.0f;

    for (const auto& det : lastResults) {
        // 过滤拦截：置信度低于滑块值的，直接跳过不画！
        if (det.confidence < threshold) {
            continue;
        }

        // ================== [视觉强化升级] ==================
        // 1. 颜色改为极限对比度的“亮青色/荧光青” (Cyan)
        QColor boxColor(0, 255, 255);

        // 2. 字体加倍, 从 11 改为 24，确保在高清图上清晰可见
        QFont labelFont("Consolas", 24, QFont::Bold);

        // 3. 边框加粗！从 3 改为 6
        QGraphicsRectItem* rectItem = scene->addRect(
            det.box.x, det.box.y, det.box.width, det.box.height,
            QPen(boxColor, 6));

        // 修复 Z 层级：确保框永远在图片最上方 (Z值设为 1)
        rectItem->setZValue(1);

        // ================== [标签渲染升级] ==================
        std::string labelStr = cv::format("Bike: %.2f", det.confidence);
        QGraphicsTextItem* textItem = scene->addText(QString::fromStdString(labelStr), labelFont);

        // 底板上的文字设为纯黑，对比度最高
        textItem->setDefaultTextColor(QColor(0, 0, 0));
        textItem->setPos(det.box.x, det.box.y - textItem->boundingRect().height() - 5);

        // 修复 Z 层级：确保文字在最顶层 (Z值设为 3)
        textItem->setZValue(3);

        // 画文字的底板背景
        QGraphicsRectItem* labelBgItem = scene->addRect(textItem->boundingRect(), QPen(Qt::NoPen), QBrush(boxColor));
        labelBgItem->setPos(textItem->pos());

        // 修复 Z 层级：确保底板在图片之上，文字之下 (Z值设为 2)
        labelBgItem->setZValue(2);
    }
}

void MainWindow::updateInfo(const std::vector<Detection>& results)
{
    ui->textEditInfo->clear();

    // 同步获取阈值
    float threshold = ui->sliderConf->value() / 100.0f;
    int validCount = 0; // 记录真正过线的单车数量

    QString details = "<h4>详细信息 (置信度)：</h4><ul>";
    for (size_t i = 0; i < results.size(); ++i) {
        // 日志里也只显示过线的单车
        if (results[i].confidence < threshold) continue;

        validCount++;
        details += QString("<li>目标 %1: %2%</li>").arg(validCount).arg(results[i].confidence * 100, 0, 'f', 1);
    }
    details += "</ul>";

    // 动态显示当前滑块阈值和最终数量
    QString info = QString("<h3>识别结果概要：</h3><p>当前过滤阈值: <b>%1%</b><br>检测到 <b>%2</b> 辆共享单车。</p>")
        .arg(ui->sliderConf->value())
        .arg(validCount);

    ui->textEditInfo->setHtml(info + details);
}