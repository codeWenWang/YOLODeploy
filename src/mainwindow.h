#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <QFileInfo>
#include <vector>
#include <string>

// 包含你之前打磨完美的 YOLO 检测器头文件
#include "yolo_detector.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

private slots:
    // 按钮点击事件的槽函数
    void on_btnUpload_clicked();   // 上传图片
    void on_btnDetect_clicked();   // 开始识别

private:
    Ui::MainWindow* ui;

    // 核心组件
    QGraphicsScene* scene;         // 用于管理和显示图片、框的场景
    YoloDetector* detector;         // 你的专属 YOLO 检测器实例

    // 状态变量
    QString currentImagePath;      // 当前加载的图片路径
    std::vector<Detection> lastResults; // 最近一次的检测结果

    // 内部辅助函数
    void displayImage(const QString& path); // 显示原图
    void drawDetections();          // 在图上画框和标签
    void updateInfo(const std::vector<Detection>& results); // 更新信息面板
};

#endif // MAINWINDOW_H