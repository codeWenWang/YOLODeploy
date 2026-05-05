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
#include <QFileInfo>
#include <QPainter>
#include <QProxyStyle>
#include <QStyleOptionSlider>

// OpenCV 相关头文件
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>

#pragma execution_character_set("utf-8")

namespace {
const QString kTargetName = QStringLiteral("美团单车");
const QString kTaskName = QStringLiteral("美团单车检测");

class RoundSliderStyle : public QProxyStyle {
public:
    using QProxyStyle::QProxyStyle;

    int pixelMetric(PixelMetric metric, const QStyleOption* option = nullptr, const QWidget* widget = nullptr) const override
    {
        if (metric == PM_SliderThickness || metric == PM_SliderLength) {
            return 16;
        }
        return QProxyStyle::pixelMetric(metric, option, widget);
    }

    QRect subControlRect(ComplexControl control, const QStyleOptionComplex* option, SubControl subControl, const QWidget* widget = nullptr) const override
    {
        if (control == CC_Slider) {
            const QStyleOptionSlider* slider = qstyleoption_cast<const QStyleOptionSlider*>(option);
            if (slider && slider->orientation == Qt::Horizontal) {
                const int handleSize = 16;
                const int y = slider->rect.center().y() - handleSize / 2;

                if (subControl == SC_SliderGroove) {
                    return QRect(slider->rect.left(), slider->rect.center().y() - 2, slider->rect.width(), 4);
                }

                if (subControl == SC_SliderHandle) {
                    const int available = slider->rect.width() - handleSize;
                    const int x = slider->rect.left() + sliderPositionFromValue(
                        slider->minimum,
                        slider->maximum,
                        slider->sliderPosition,
                        available,
                        slider->upsideDown);
                    return QRect(x, y, handleSize, handleSize);
                }
            }
        }

        return QProxyStyle::subControlRect(control, option, subControl, widget);
    }

    void drawComplexControl(ComplexControl control, const QStyleOptionComplex* option, QPainter* painter, const QWidget* widget = nullptr) const override
    {
        if (control != CC_Slider) {
            QProxyStyle::drawComplexControl(control, option, painter, widget);
            return;
        }

        const QStyleOptionSlider* slider = qstyleoption_cast<const QStyleOptionSlider*>(option);
        if (!slider || slider->orientation != Qt::Horizontal) {
            QProxyStyle::drawComplexControl(control, option, painter, widget);
            return;
        }

        painter->save();
        painter->setRenderHint(QPainter::Antialiasing, true);

        const QRect groove = subControlRect(control, option, SC_SliderGroove, widget);
        const QRect handle = subControlRect(control, option, SC_SliderHandle, widget);
        const int centerX = handle.center().x();

        painter->setPen(Qt::NoPen);
        painter->setBrush(QColor(52, 58, 70));
        painter->drawRoundedRect(groove, 2, 2);

        QRect activeGroove = groove;
        activeGroove.setRight(centerX);
        painter->setBrush(QColor(255, 209, 102));
        painter->drawRoundedRect(activeGroove, 2, 2);

        const bool hovered = slider->state & State_MouseOver;
        painter->setBrush(hovered ? QColor(255, 224, 138) : QColor(255, 209, 102));
        painter->setPen(QPen(QColor(255, 243, 208), 2));
        painter->drawEllipse(handle.adjusted(1, 1, -1, -1));

        painter->restore();
    }
};
}

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , currentImagePath("")
    , detector(nullptr) // 安全修复：初始化检测器为空指针
{
    ui->setupUi(this);

    // ================= [UI 美化核心代码] =================
    QString proDarkStyle = R"(
        QMainWindow, QWidget#centralwidget {
            background-color: #101114;
            font-family: "Microsoft YaHei UI", "Microsoft YaHei";
        }

        QWidget#leftPanel {
            background-color: #17191F;
            border: 1px solid #2A2E37;
            border-radius: 14px;
        }

        QWidget#imagePanel {
            background-color: #14161B;
            border: 1px solid #292D36;
            border-radius: 16px;
        }

        QLabel {
            color: #DCE1EA;
            font-size: 13px;
            font-weight: 500;
        }

        QLabel#labelAppTitle {
            color: #FFFFFF;
            font-size: 26px;
            font-weight: 700;
            letter-spacing: 0px;
        }

        QLabel#labelSubtitle {
            color: #8F96A3;
            font-size: 13px;
            font-weight: 500;
        }

        QLabel#labelTask, QLabel#labelConf {
            color: #AEB6C4;
            font-size: 12px;
            font-weight: 500;
        }

        QLabel#labelStatus {
            color: #9CA5B4;
            background-color: #1F232B;
            border: 1px solid #303642;
            border-radius: 8px;
            padding: 9px 10px;
            font-size: 13px;
            font-weight: 500;
        }

        QLabel#labelViewportTitle {
            color: #F3F6FA;
            font-size: 16px;
            font-weight: 650;
        }

        QLabel#labelModelBadge {
            color: #FFE9D1;
            background-color: #3A251D;
            border: 1px solid #725039;
            border-radius: 10px;
            padding: 5px 12px;
            font-size: 12px;
            font-weight: 600;
        }

        QComboBox {
            background-color: #20242D;
            color: #F4F7FA;
            border: 1px solid #383F4C;
            border-radius: 10px;
            padding: 10px 12px;
            min-height: 24px;
            font-family: "Microsoft YaHei UI";
            font-size: 15px;
            font-weight: 500;
        }
        QComboBox:hover {
            border-color: #FFD166;
        }
        QComboBox::drop-down {
            border: none;
            width: 28px;
        }
        QComboBox QAbstractItemView {
            background-color: #20242D;
            color: #F4F7FA;
            selection-background-color: #313847;
            border: 1px solid #383F4C;
            outline: 0;
        }

        QGraphicsView {
            background-color: #090A0D;
            border: 1px solid #2B303A;
            border-radius: 12px;
            padding: 4px;
        }

        QPushButton {
            background-color: #FFD166;
            color: #181A1F;
            border: none;
            border-radius: 10px;
            padding: 10px 12px;
            font-family: "Microsoft YaHei UI";
            font-size: 15px;
            font-weight: 600;
            min-height: 30px;
        }
        QPushButton:hover { background-color: #FFE08A; }
        QPushButton:pressed {
            background-color: #E9B84F;
            padding-top: 11px;
            padding-bottom: 9px;
        }

        QTextEdit {
            background-color: #111318;
            color: #DCE1EA;
            border: 1px solid #303642;
            border-radius: 10px;
            padding: 12px;
            font-family: "Microsoft YaHei UI", "Microsoft YaHei";
            font-size: 13px;
            font-weight: 400;
        }
    )";
    this->setStyleSheet(proDarkStyle);
    this->setWindowTitle(QStringLiteral("Meituan Vision - 多目标检测工作台"));

    ui->sliderConf->setStyle(new RoundSliderStyle(ui->sliderConf->style()));
    ui->sliderConf->setMinimumHeight(24);
    ui->sliderConf->setMouseTracking(true);
    ui->btnUpload->setCursor(Qt::PointingHandCursor);
    ui->btnDetect->setCursor(Qt::PointingHandCursor);
    ui->sliderConf->setCursor(Qt::PointingHandCursor);
    ui->comboTarget->setCursor(Qt::PointingHandCursor);

    // 新增：初始化滑块范围 1~100，默认值 50 (代表 0.5 置信度)
    ui->sliderConf->setRange(1, 100);
    ui->sliderConf->setValue(50);
    ui->textEditInfo->setHtml(QStringLiteral(
        "<h3 style='color:#FFFFFF; margin-top:0;'>等待检测</h3>"
        "<p style='color:#AEB6C4;'>当前任务：<b>美团单车检测</b></p>"
        "<p style='color:#7F8794;'>请选择图片后开始检测。</p>"));
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
        ui->labelStatus->setText(QStringLiteral("状态：模型加载失败"));
    }

    // 实时响应滑块拖动：只要滑块值改变，立马重新画框并更新日志！
    connect(ui->sliderConf, &QSlider::valueChanged, this, [=]() {
        if (!currentImagePath.isEmpty() && !lastResults.empty()) {
            drawDetections();
            updateInfo(lastResults);
        }
        });

    connect(ui->comboTarget, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [=](int index) {
        if (index != 0) {
            ui->comboTarget->setCurrentIndex(0);
            QMessageBox::information(this, QStringLiteral("任务预留"), QStringLiteral("当前版本已接入美团单车检测，其他目标类型会在后续模型接入后开放。"));
        }
        ui->labelStatus->setText(QStringLiteral("状态：当前任务 - %1").arg(kTaskName));
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
    ui->labelStatus->setText(QStringLiteral("状态：已载入 %1").arg(QFileInfo(path).fileName()));
    ui->textEditInfo->setHtml(QStringLiteral(
        "<h3 style='color:#FFFFFF; margin-top:0;'>图片已就绪</h3>"
        "<p style='color:#AEB6C4;'>任务：<b>%1</b></p>"
        "<p style='color:#7F8794;'>点击“开始检测”运行模型。</p>").arg(kTaskName));
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

    ui->labelStatus->setText(QStringLiteral("状态：正在检测美团单车..."));

    cv::Mat frame = cv::imread(currentImagePath.toLocal8Bit().constData());
    if (frame.empty()) {
        QMessageBox::warning(this, "警告", "OpenCV 读取图片失败！");
        ui->labelStatus->setText(QStringLiteral("状态：图片读取失败"));
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
        // 使用美团黄作为目标色，让检测结果与任务语义保持一致。
        QColor boxColor(255, 209, 102);

        const double viewScale = std::max(0.01, std::abs(ui->graphicsView->transform().m11()));
        const double displayedBoxWidth = det.box.width * viewScale;
        const double displayedBoxHeight = det.box.height * viewScale;

        int dynamicFontSize = static_cast<int>(std::round(std::min(displayedBoxHeight * 0.15, displayedBoxWidth * 0.085)));
        dynamicFontSize = std::max(12, std::min(dynamicFontSize, 20));

        int dynamicLineWidth = static_cast<int>(std::round(std::min(displayedBoxWidth, displayedBoxHeight) * 0.008));
        dynamicLineWidth = std::max(1, std::min(dynamicLineWidth, 3));

        QFont labelFont("Microsoft YaHei UI", QFont::Bold);
        labelFont.setPixelSize(dynamicFontSize);
        labelFont.setWeight(QFont::Black);

        QPen boxPen(boxColor, dynamicLineWidth);
        boxPen.setCosmetic(true);

        QGraphicsRectItem* rectItem = scene->addRect(
            det.box.x, det.box.y, det.box.width, det.box.height,
            boxPen);

        // 修复 Z 层级：确保框永远在图片最上方 (Z值设为 1)
        rectItem->setZValue(1);

        // ================== [标签渲染升级] ==================
        QString labelStr = QStringLiteral("%1  %2%").arg(kTargetName).arg(det.confidence * 100, 0, 'f', 0);
        QGraphicsTextItem* textItem = scene->addText(labelStr, labelFont);
        textItem->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);

        textItem->setDefaultTextColor(QColor(18, 20, 24));

        // 修复 Z 层级：确保文字在最顶层 (Z值设为 3)
        textItem->setZValue(3);

        // 画文字的底板背景
        const int horizontalPadding = std::max(6, dynamicFontSize / 2);
        const int verticalPadding = std::max(3, dynamicFontSize / 4);
        const QRectF textRect = textItem->boundingRect();
        const QRectF labelRect(0, 0, textRect.width() + horizontalPadding * 2, textRect.height() + verticalPadding * 2);
        const double labelHeightInScene = labelRect.height() / viewScale;
        const double labelGapInScene = 1.0 / viewScale;
        const bool hasRoomAbove = det.box.y > labelHeightInScene + labelGapInScene;
        const double labelX = det.box.x;
        const double labelY = hasRoomAbove
            ? det.box.y - labelHeightInScene - labelGapInScene
            : det.box.y + labelGapInScene;

        textItem->setPos(labelX + horizontalPadding / viewScale, labelY + verticalPadding / viewScale);

        QGraphicsRectItem* labelBgItem = scene->addRect(labelRect, QPen(Qt::NoPen), QBrush(boxColor));
        labelBgItem->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
        labelBgItem->setPos(labelX, labelY);

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

    QString details = QStringLiteral(
        "<div style='margin-top:18px;'>"
        "<div style='color:#F4F7FA; font-size:16px; font-weight:700; margin-bottom:8px;'>检测明细</div>"
        "<ul style='color:#D7DEE9; font-size:14px; line-height:145%; margin-top:0;'>");
    for (size_t i = 0; i < results.size(); ++i) {
        // 日志里也只显示过线的单车
        if (results[i].confidence < threshold) continue;

        validCount++;
        details += QStringLiteral("<li>%1 %2：%3%</li>")
            .arg(kTargetName)
            .arg(validCount)
            .arg(results[i].confidence * 100, 0, 'f', 1);
    }
    details += "</ul></div>";

    // 动态显示当前滑块阈值和最终数量
    QString info = QStringLiteral(
        "<div style='font-family:\"Microsoft YaHei UI\", \"Microsoft YaHei\";'>"
        "<div style='color:#F4F7FA; font-size:22px; font-weight:800; margin-bottom:14px;'>识别结果概要</div>"
        "<div style='color:#AEB6C4; font-size:14px; line-height:155%;'>"
        "当前任务：<b style='color:#FFFFFF;'>%1</b><br>"
        "过滤阈值：<b style='color:#FFFFFF;'>%2%</b><br>"
        "检测到 <b style='color:#FFD166; font-size:18px;'>%3</b> 辆美团单车。"
        "</div>")
        .arg(kTaskName)
        .arg(ui->sliderConf->value())
        .arg(validCount);

    ui->labelStatus->setText(QStringLiteral("状态：完成检测，发现 %1 辆美团单车").arg(validCount));
    ui->textEditInfo->setHtml(info + details + QStringLiteral("</div>"));
}
