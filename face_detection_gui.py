# face_detection_gui.py - 人脸识别图像检测系统
# 高颜值设计，适合学术/课程作业风格

import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QGroupBox, QScrollArea,
    QFileDialog, QMessageBox, QApplication, QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QColor, QPainter, QPen, QBrush

# 颜色配置
COLORS = {
    'bg_main': '#f8f9fa',           # 主背景
    'bg_light': '#ffffff',          # 卡片背景
    'border': '#86a8e7',            # 上传框边框
    'face_box': '#22c55e',          # 人脸框颜色
    'text_dark': '#333333',          # 深色文字
    'text_light': '#666666',         # 浅色文字
    'btn_primary': '#86a8e7',       # 主按钮色
    'btn_hover': '#6b8fd4',         # 按钮悬停
    'success': '#22c55e',           # 成功色
    'warning': '#f59e0b',           # 警告色
    'danger': '#ef4444',            # 危险色
    'shadow': 'rgba(0, 0, 0, 0.1)', # 阴影色
}

class RoundedButton(QPushButton):
    """圆角按钮"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {
                background-color: %s;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                font-family: 'Microsoft YaHei';
            }
            QPushButton:hover {
                background-color: %s;
            }
            QPushButton:pressed {
                background-color: #5a7fc4;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """ % (COLORS['btn_primary'], COLORS['btn_hover']))

class ImageDisplayLabel(QLabel):
    """图片显示标签"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(500, 400)
        self.setStyleSheet("""
            QLabel {
                background-color: %s;
                border: 3px dashed %s;
                border-radius: 10px;
                color: %s;
                font-size: 16px;
                font-family: 'Microsoft YaHei';
            }
        """ % (COLORS['bg_light'], COLORS['border'], COLORS['text_light']))
        self.setText("📷\n\n请选择待识别人像图片\n\n点击下方【选择图片】按钮上传")

    def setImage(self, pixmap):
        """设置图片"""
        self.setPixmap(pixmap)
        self.setStyleSheet("""
            QLabel {
                border: 3px solid %s;
                border-radius: 10px;
                background-color: %s;
            }
        """ % (COLORS['border'], COLORS['bg_light']))

    def clear(self):
        """清除图片"""
        self.setImage(None)
        self.setStyleSheet("""
            QLabel {
                background-color: %s;
                border: 3px dashed %s;
                border-radius: 10px;
                color: %s;
                font-size: 16px;
                font-family: 'Microsoft YaHei';
            }
        """ % (COLORS['bg_light'], COLORS['border'], COLORS['text_light']))
        self.setText("📷\n\n请选择待识别人像图片\n\n点击下方【选择图片】按钮上传")

class InfoCard(QGroupBox):
    """信息卡片"""
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                font-family: 'Microsoft YaHei';
                color: %s;
                border: 2px solid #e5e7eb;
                border-radius: 10px;
                margin-top: 10px;
                padding: 15px;
                background-color: %s;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                color: %s;
            }
        """ % (COLORS['text_dark'], COLORS['bg_light'], COLORS['btn_primary']))
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 20, 10, 10)
        self.setLayout(layout)

class DetectionThread(QThread):
    """人脸检测线程"""
    finished = pyqtSignal(list, list, float)  # 人脸列表, 坐标列表, 平均置信度

    def __init__(self, image_path, face_cascade):
        super().__init__()
        self.image_path = image_path
        self.face_cascade = face_cascade

    def run(self):
        image = cv2.imread(self.image_path)
        if image is None:
            self.finished.emit([], [], 0.0)
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale3(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            outputRejectLevels=True
        )

        face_rects = faces[0]
        confidents = faces[1]
        reject_levels = faces[2]

        # 计算平均置信度
        avg_confidence = 0.0
        if len(confidents) > 0:
            avg_confidence = np.mean(confidents)

        self.finished.emit(face_rects.tolist(), confidents.tolist(), float(avg_confidence))

class FaceDetectionGUI(QMainWindow):
    """人脸识别主窗口"""

    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.detected_faces = []
        self.face_cascade = None
        self.initUI()
        self.loadFaceDetector()

    def initUI(self):
        """初始化界面"""
        self.setWindowTitle("🤖 人脸识别图像检测系统")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("QMainWindow { background-color: %s; }" % COLORS['bg_main'])

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # 标题区域
        self.createHeader(main_layout)

        # 内容区域
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # 左侧：图片展示区
        self.createImageSection(content_layout)

        # 右侧：信息面板
        self.createInfoSection(content_layout)

        main_layout.addLayout(content_layout)

        # 底部按钮区域
        self.createButtonSection(main_layout)

        # 状态栏
        self.statusBar().showMessage("就绪")
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: %s;
                color: %s;
                font-family: 'Microsoft YaHei';
                font-size: 13px;
                padding: 5px;
            }
        """ % (COLORS['bg_light'], COLORS['text_dark']))

    def createHeader(self, parent_layout):
        """创建标题区域"""
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 %s,
                    stop: 1 %s
                );
                border-radius: 15px;
                padding: 25px;
            }
        """ % (COLORS['btn_primary'], COLORS['btn_hover']))

        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 15, 20, 15)

        # 主标题
        title_label = QLabel("👤 人脸识别图像检测系统")
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 28px;
                font-weight: bold;
                font-family: 'Microsoft YaHei';
            }
        """)
        title_label.setAlignment(Qt.AlignCenter)

        # 副标题
        subtitle_label = QLabel("📷 图片人脸定位 · 特征比对 · 精准识别")
        subtitle_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 0.9);
                font-size: 14px;
                font-family: 'Microsoft YaHei';
                margin-top: 5px;
            }
        """)
        subtitle_label.setAlignment(Qt.AlignCenter)

        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)

        parent_layout.addWidget(header_frame)

    def createImageSection(self, parent_layout):
        """创建图片展示区域"""
        image_container = QFrame()
        image_container.setStyleSheet("""
            QFrame {
                background-color: %s;
                border-radius: 15px;
                padding: 15px;
            }
        """ % (COLORS['bg_light']))

        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 5)
        image_container.setGraphicsEffect(shadow)

        layout = QVBoxLayout(image_container)
        layout.setContentsMargins(15, 15, 15, 15)

        # 图片显示标签
        self.image_label = ImageDisplayLabel()

        # 加载示例图片按钮提示
        self.load_hint = QLabel("支持 JPG/PNG/BMP 格式图片")
        self.load_hint.setStyleSheet("""
            QLabel {
                color: %s;
                font-size: 12px;
                font-family: 'Microsoft YaHei';
                margin-top: 10px;
            }
        """ % COLORS['text_light'])
        self.load_hint.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.load_hint)

        parent_layout.addWidget(image_container, 3)

    def createInfoSection(self, parent_layout):
        """创建信息面板区域"""
        info_container = QFrame()
        info_container.setStyleSheet("""
            QFrame {
                background-color: %s;
                border-radius: 15px;
                padding: 15px;
            }
        """ % (COLORS['bg_light']))

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 5)
        info_container.setGraphicsEffect(shadow)

        layout = QVBoxLayout(info_container)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # 基础信息卡片
        self.basic_info_card = InfoCard("📋 基础信息")
        self.basic_info_card.layout().addWidget(self.createInfoLabel("图片尺寸：", "---"))
        self.basic_info_card.layout().addWidget(self.createInfoLabel("文件大小：", "---"))
        self.basic_info_card.layout().addWidget(self.createInfoLabel("图片格式：", "---"))
        layout.addWidget(self.basic_info_card)

        # 识别结果卡片
        self.result_card = InfoCard("🎯 识别结果")
        self.face_count_label = self.createInfoLabel("人脸数量：", "0")
        self.avg_confidence_label = self.createInfoLabel("平均置信度：", "0%")
        self.result_card.layout().addWidget(self.face_count_label)
        self.result_card.layout().addWidget(self.avg_confidence_label)
        layout.addWidget(self.result_card)

        # 判定结果卡片
        self.status_card = InfoCard("✅ 判定结果")
        self.detection_status_label = QLabel("⏳ 等待检测")
        self.detection_status_label.setStyleSheet("""
            QLabel {
                color: %s;
                font-size: 16px;
                font-weight: bold;
                font-family: 'Microsoft YaHei';
                padding: 10px;
                background-color: %s;
                border-radius: 8px;
            }
        """ % (COLORS['text_light'], COLORS['bg_main']))
        self.detection_status_label.setAlignment(Qt.AlignCenter)
        self.status_card.layout().addWidget(self.detection_status_label)
        layout.addWidget(self.status_card)

        layout.addStretch()
        parent_layout.addWidget(info_container, 1)

    def createInfoLabel(self, title, value):
        """创建信息标签"""
        label = QLabel(f"{title} <span style='color: {COLORS['btn_primary']}; font-weight: bold;'>{value}</span>")
        label.setStyleSheet("""
            QLabel {
                font-size: 13px;
                font-family: 'Microsoft YaHei';
                color: %s;
                padding: 5px 0;
            }
        """ % COLORS['text_dark'])
        return label

    def createButtonSection(self, parent_layout):
        """创建按钮区域"""
        button_frame = QFrame()
        button_frame.setStyleSheet("""
            QFrame {
                background-color: %s;
                border-radius: 15px;
                padding: 15px;
            }
        """ % (COLORS['bg_light']))

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 5)
        button_frame.setGraphicsEffect(shadow)

        layout = QHBoxLayout(button_frame)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(15)

        # 选择图片按钮
        self.select_btn = RoundedButton("📁 选择图片")
        self.select_btn.setMinimumHeight(45)
        self.select_btn.clicked.connect(self.selectImage)

        # 开始识别按钮
        self.detect_btn = RoundedButton("🔍 开始识别")
        self.detect_btn.setMinimumHeight(45)
        self.detect_btn.setEnabled(False)
        self.detect_btn.clicked.connect(self.startDetection)

        # 清空图片按钮
        self.clear_btn = RoundedButton("🗑️ 清空图片")
        self.clear_btn.setMinimumHeight(45)
        self.clear_btn.setEnabled(False)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: %s;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                font-family: 'Microsoft YaHei';
            }
            QPushButton:hover {
                background-color: %s;
            }
            QPushButton:pressed {
                background-color: #d42e2e;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """ % (COLORS['danger'], '#c0392b'))
        self.clear_btn.clicked.connect(self.clearAll)

        # 保存结果按钮
        self.save_btn = RoundedButton("💾 保存结果图")
        self.save_btn.setMinimumHeight(45)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: %s;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                font-family: 'Microsoft YaHei';
            }
            QPushButton:hover {
                background-color: #1d9a4a;
            }
            QPushButton:pressed {
                background-color: #1a8641;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """ % (COLORS['success']))
        self.save_btn.clicked.connect(self.saveResultImage)

        layout.addWidget(self.select_btn)
        layout.addWidget(self.detect_btn)
        layout.addWidget(self.clear_btn)
        layout.addWidget(self.save_btn)

        parent_layout.addWidget(button_frame)

    def loadFaceDetector(self):
        """加载人脸检测器"""
        # 使用OpenCV内置的Haar级联分类器
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            QMessageBox.warning(self, "警告", "无法加载人脸检测器！")
            self.statusBar().showMessage("人脸检测器加载失败")

    def selectImage(self):
        """选择图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )

        if file_path:
            self.current_image_path = file_path
            self.loadImage(file_path)
            self.detect_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)
            self.save_btn.setEnabled(False)

    def loadImage(self, file_path):
        """加载图片"""
        # 读取图片获取信息
        img = cv2.imread(file_path)
        if img is None:
            QMessageBox.warning(self, "错误", "无法读取图片！")
            return

        height, width, channels = img.shape
        file_size = os.path.getsize(file_path) / 1024  # KB

        # 更新基础信息
        self.basic_info_card.layout().itemAt(0).widget().setText(
            f"图片尺寸：<span style='color: {COLORS['btn_primary']}; font-weight: bold;'>{width} × {height}</span>"
        )
        self.basic_info_card.layout().itemAt(1).widget().setText(
            f"文件大小：<span style='color: {COLORS['btn_primary']}; font-weight: bold;'>{file_size:.1f} KB</span>"
        )

        # 获取文件格式
        ext = os.path.splitext(file_path)[1].upper()[1:]
        self.basic_info_card.layout().itemAt(2).widget().setText(
            f"图片格式：<span style='color: {COLORS['btn_primary']}; font-weight: bold;'>{ext}</span>"
        )

        # 显示图片
        pixmap = QPixmap(file_path)
        scaled_pixmap = pixmap.scaled(
            600, 450,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setImage(scaled_pixmap)

        # 重置状态
        self.face_count_label.setText(
            f"人脸数量：<span style='color: {COLORS['btn_primary']}; font-weight: bold;'>0</span>"
        )
        self.avg_confidence_label.setText(
            f"平均置信度：<span style='color: {COLORS['btn_primary']}; font-weight: bold;'>0%</span>"
        )
        self.detection_status_label.setText("⏳ 等待检测")
        self.detection_status_label.setStyleSheet("""
            QLabel {
                color: %s;
                font-size: 16px;
                font-weight: bold;
                font-family: 'Microsoft YaHei';
                padding: 10px;
                background-color: %s;
                border-radius: 8px;
            }
        """ % (COLORS['text_light'], COLORS['bg_main']))

        self.statusBar().showMessage("图片已加载，点击【开始识别】进行人脸检测")
        self.detected_faces = []

    def startDetection(self):
        """开始人脸检测"""
        if not self.current_image_path:
            return

        self.statusBar().showMessage("🔍 正在检测人脸...")
        self.detect_btn.setEnabled(False)
        self.detection_status_label.setText("🔄 检测中...")

        # 创建检测线程
        self.detection_thread = DetectionThread(self.current_image_path, self.face_cascade)
        self.detection_thread.finished.connect(self.onDetectionComplete)
        self.detection_thread.start()

    def onDetectionComplete(self, face_rects, confidents, avg_confidence):
        """检测完成回调"""
        self.detected_faces = face_rects
        num_faces = len(face_rects)

        # 更新识别结果
        self.face_count_label.setText(
            f"人脸数量：<span style='color: {COLORS['btn_primary']}; font-weight: bold;'>{num_faces}</span>"
        )
        self.avg_confidence_label.setText(
            f"平均置信度：<span style='color: {COLORS['btn_primary']}; font-weight: bold;'>{avg_confidence:.2f}%</span>"
        )

        # 更新检测状态
        if num_faces > 0:
            self.detection_status_label.setText(f"✅ 检测完成：发现 {num_faces} 个人脸")
            self.detection_status_label.setStyleSheet("""
                QLabel {
                    color: %s;
                    font-size: 16px;
                    font-weight: bold;
                    font-family: 'Microsoft YaHei';
                    padding: 10px;
                    background-color: rgba(34, 197, 94, 0.2);
                    border-radius: 8px;
                }
            """ % COLORS['success'])
            self.save_btn.setEnabled(True)
        else:
            self.detection_status_label.setText("❌ 未检测到人脸")
            self.detection_status_label.setStyleSheet("""
                QLabel {
                    color: %s;
                    font-size: 16px;
                    font-weight: bold;
                    font-family: 'Microsoft YaHei';
                    padding: 10px;
                    background-color: rgba(239, 68, 68, 0.2);
                    border-radius: 8px;
                }
            """ % COLORS['danger'])

        # 在图片上绘制人脸框
        self.drawFaceBoxes()

        self.detect_btn.setEnabled(True)
        self.statusBar().showMessage(f"检测完成：发现 {num_faces} 个人脸")

    def drawFaceBoxes(self):
        """在人脸图片上绘制人脸框"""
        if not self.current_image_path or not self.detected_faces:
            return

        # 读取原图
        image = cv2.imread(self.current_image_path)
        if image is None:
            return

        # 绘制每个人脸框
        for (x, y, w, h) in self.detected_faces:
            # 绘制绿色矩形框
            cv2.rectangle(image, (x, y), (x + w, y + h), (34, 197, 94), 3)

            # 添加人脸区域标签
            label = f"Face"
            cv2.putText(
                image, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (34, 197, 94), 2
            )

        # 保存带人脸框的图片到临时文件
        temp_path = os.path.join(os.path.dirname(self.current_image_path), "temp_face_detected.jpg")
        cv2.imwrite(temp_path, image)

        # 显示带人脸框的图片
        pixmap = QPixmap(temp_path)
        scaled_pixmap = pixmap.scaled(
            600, 450,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setImage(scaled_pixmap)

    def clearAll(self):
        """清空所有"""
        self.current_image_path = None
        self.detected_faces = []

        # 重置图片显示
        self.image_label.clear()

        # 重置基础信息
        self.basic_info_card.layout().itemAt(0).widget().setText(
            f"图片尺寸：<span style='color: {COLORS['btn_primary']}; font-weight: bold;'>---</span>"
        )
        self.basic_info_card.layout().itemAt(1).widget().setText(
            f"文件大小：<span style='color: {COLORS['btn_primary']}; font-weight: bold;'>---</span>"
        )
        self.basic_info_card.layout().itemAt(2).widget().setText(
            f"图片格式：<span style='color: {COLORS['btn_primary']}; font-weight: bold;'>---</span>"
        )

        # 重置识别结果
        self.face_count_label.setText(
            f"人脸数量：<span style='color: {COLORS['btn_primary']}; font-weight: bold;'>0</span>"
        )
        self.avg_confidence_label.setText(
            f"平均置信度：<span style='color: {COLORS['btn_primary']}; font-weight: bold;'>0%</span>"
        )

        # 重置检测状态
        self.detection_status_label.setText("⏳ 等待检测")
        self.detection_status_label.setStyleSheet("""
            QLabel {
                color: %s;
                font-size: 16px;
                font-weight: bold;
                font-family: 'Microsoft YaHei';
                padding: 10px;
                background-color: %s;
                border-radius: 8px;
            }
        """ % (COLORS['text_light'], COLORS['bg_main']))

        # 禁用按钮
        self.detect_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

        self.statusBar().showMessage("已清空")

    def saveResultImage(self):
        """保存结果图"""
        if not self.current_image_path or not self.detected_faces:
            return

        # 读取原图并绘制人脸框
        image = cv2.imread(self.current_image_path)
        for (x, y, w, h) in self.detected_faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (34, 197, 94), 3)
            cv2.putText(image, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (34, 197, 94), 2)

        # 获取保存路径
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存结果图",
            os.path.splitext(self.current_image_path)[0] + "_detected.jpg",
            "图片文件 (*.jpg *.png);;所有文件 (*)"
        )

        if save_path:
            cv2.imwrite(save_path, image)
            QMessageBox.information(self, "成功", f"结果图已保存到：\n{save_path}")
            self.statusBar().showMessage(f"已保存：{save_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    # 设置应用样式
    app.setStyle('Fusion')

    window = FaceDetectionGUI()
    window.show()

    sys.exit(app.exec_())