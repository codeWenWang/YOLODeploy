#include "mainwindow.h"
#include <QApplication>

int main(int argc, char* argv[]) {
    // 1. 初始化 QT 应用程序环境
    QApplication a(argc, argv);

    // 2. 实例化主窗口
    MainWindow w;

    // 3. 显示窗口
    w.show();

    // 4. 进入 QT 的事件循环（程序会停在这里，等待用户点击按钮）
    return a.exec();
}