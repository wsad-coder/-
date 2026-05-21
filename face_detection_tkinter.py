# face_detection_tkinter.py - 人脸识别图像检测系统（tkinter版本）
# 高颜值设计，适合学术/课程作业风格

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import os
import numpy as np
from threading import Thread

# 颜色配置
COLORS = {
    'bg_main': '#f8f9fa',
    'bg_light': '#ffffff',
    'border': '#86a8e7',
    'face_box': '#22c55e',
    'text_dark': '#333333',
    'text_light': '#666666',
    'btn_primary': '#86a8e7',
    'btn_hover': '#6b8fd4',
    'success': '#22c55e',
    'warning': '#f59e0b',
    'danger': '#ef4444',
}

class ModernButton(tk.Button):
    def __init__(self, master, text, command, bg=COLORS['btn_primary'], hover_bg=COLORS['btn_hover'], **kwargs):
        super().__init__(
            master, text=text, command=command,
            bg=bg, fg='white', activebackground=hover_bg,
            activeforeground='white', relief=tk.RAISED, bd=2,
            font=('Microsoft YaHei', 11, 'bold'), cursor='hand2',
            padx=20, pady=10
        )
        self.hover_bg = hover_bg
        self.default_bg = bg
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)

    def on_enter(self, event):
        self.config(bg=self.hover_bg, relief=tk.SUNKEN)

    def on_leave(self, event):
        self.config(bg=self.default_bg, relief=tk.RAISED)

class InfoCard(tk.Frame):
    def __init__(self, parent, title, **kwargs):
        super().__init__(parent, bg=COLORS['bg_light'], bd=0, relief=tk.RAISED, **kwargs)

        # 标题
        title_label = tk.Label(
            self,
            text=title,
            font=('Microsoft YaHei', 12, 'bold'),
            bg=COLORS['bg_light'],
            fg=COLORS['btn_primary'],
            anchor='w'
        )
        title_label.pack(fill=tk.X, padx=15, pady=(10, 5))

        # 分隔线
        separator = tk.Frame(self, bg=COLORS['border'], height=2)
        separator.pack(fill=tk.X, padx=15)

        # 内容区域
        self.content_frame = tk.Frame(self, bg=COLORS['bg_light'])
        self.content_frame.pack(fill=tk.X, padx=15, pady=10)

    def add_info(self, label_text, value_text="---"):
        row = tk.Frame(self.content_frame, bg=COLORS['bg_light'])
        row.pack(fill=tk.X, pady=3)

        label = tk.Label(
            row,
            text=label_text,
            font=('Microsoft YaHei', 10),
            bg=COLORS['bg_light'],
            fg=COLORS['text_dark'],
            anchor='w'
        )
        label.pack(side=tk.LEFT)

        value = tk.Label(
            row,
            text=value_text,
            font=('Microsoft YaHei', 10, 'bold'),
            bg=COLORS['bg_light'],
            fg=COLORS['btn_primary'],
            anchor='e'
        )
        value.pack(side=tk.RIGHT)

        return value

class FaceDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 人脸识别图像检测系统")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        self.root.configure(bg=COLORS['bg_main'])

        self.current_image_path = None
        self.detected_faces = []
        self.face_cascade = None
        self.processing = False

        self.load_face_detector()
        self.create_ui()

    def load_face_detector(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            messagebox.showwarning("警告", "无法加载人脸检测器！")

    def create_ui(self):
        # 标题栏
        self.create_header()

        # 主内容区
        content = tk.Frame(self.root, bg=COLORS['bg_main'])
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        # 左侧：图片展示区
        left_frame = tk.Frame(content, bg=COLORS['bg_light'], bd=0)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.create_image_section(left_frame)
        self.create_buttons(left_frame)

        # 右侧：信息面板
        right_frame = tk.Frame(content, bg=COLORS['bg_light'], bd=0, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)

        self.create_info_section(right_frame)

        # 状态栏
        self.create_footer()

    def create_header(self):
        header = tk.Frame(self.root, bg=COLORS['btn_primary'], height=100)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        # 标题
        title = tk.Label(
            header,
            text="👤 人脸识别图像检测系统",
            font=('Microsoft YaHei', 24, 'bold'),
            bg=COLORS['btn_primary'],
            fg='white'
        )
        title.pack(pady=(20, 5))

        # 副标题
        subtitle = tk.Label(
            header,
            text="📷 图片人脸定位 · 特征比对 · 精准识别",
            font=('Microsoft YaHei', 11),
            bg=COLORS['btn_primary'],
            fg='rgba(255,255,255,0.9)'
        )
        subtitle.pack(pady=(0, 15))

    def create_image_section(self, parent):
        # 图片容器
        container = tk.Frame(parent, bg=COLORS['bg_light'], bd=0)
        container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # 图片显示标签
        self.image_label = tk.Label(
            container,
            text="📷\n\n请选择待识别人像图片\n\n点击下方【选择图片】按钮上传",
            font=('Microsoft YaHei', 14),
            bg=COLORS['bg_light'],
            fg=COLORS['text_light'],
            width=60,
            height=20,
            anchor='center',
            justify='center',
            relief=tk.RAISED,
            bd=3
        )
        self.image_label.pack(pady=(0, 10))

        # 提示文字
        hint = tk.Label(
            container,
            text="支持 JPG/PNG/BMP 格式图片 | 检测到的人脸将用绿色框标注",
            font=('Microsoft YaHei', 9),
            bg=COLORS['bg_light'],
            fg=COLORS['text_light']
        )
        hint.pack()

    def create_buttons(self, parent):
        btn_frame = tk.Frame(parent, bg=COLORS['bg_light'])
        btn_frame.pack(pady=15)

        self.select_btn = ModernButton(
            btn_frame, "📁 选择图片",
            command=self.select_image,
            bg='#4A90E2', hover_bg='#357ABD'
        )
        self.select_btn.grid(row=0, column=0, padx=10, ipady=8)

        self.detect_btn = ModernButton(
            btn_frame, "🔍 开始识别",
            command=self.start_detection,
            bg=COLORS['success'], hover_bg='#1d9a4a'
        )
        self.detect_btn.grid(row=0, column=1, padx=10, ipady=8)
        self.detect_btn.config(state=tk.DISABLED)

        self.clear_btn = ModernButton(
            btn_frame, "🗑️ 清空图片",
            command=self.clear_all,
            bg=COLORS['danger'], hover_bg='#c0392b'
        )
        self.clear_btn.grid(row=0, column=2, padx=10, ipady=8)
        self.clear_btn.config(state=tk.DISABLED)

        self.save_btn = ModernButton(
            btn_frame, "💾 保存结果",
            command=self.save_result,
            bg='#2ECC71', hover_bg='#1d9a4a'
        )
        self.save_btn.grid(row=0, column=3, padx=10, ipady=8)
        self.save_btn.config(state=tk.DISABLED)

    def create_info_section(self, parent):
        # 基础信息卡片
        self.basic_card = InfoCard(parent, "📋 基础信息")
        self.basic_card.pack(fill=tk.X, padx=10, pady=(15, 10))

        self.size_label = self.basic_card.add_info("图片尺寸：")
        self.size_label.config(text="---")

        self.size_label = self.basic_card.add_info("文件大小：")
        self.size_label.config(text="---")

        self.size_label = self.basic_card.add_info("图片格式：")
        self.size_label.config(text="---")

        # 识别结果卡片
        self.result_card = InfoCard(parent, "🎯 识别结果")
        self.result_card.pack(fill=tk.X, padx=10, pady=10)

        self.face_count_label = self.result_card.add_info("人脸数量：")
        self.face_count_label.config(text="0")

        self.confidence_label = self.result_card.add_info("平均置信度：")
        self.confidence_label.config(text="0%")

        # 判定结果卡片
        self.status_card = InfoCard(parent, "✅ 判定结果")
        self.status_card.pack(fill=tk.X, padx=10, pady=(10, 15))

        self.status_label = tk.Label(
            self.status_card.content_frame,
            text="⏳ 等待检测",
            font=('Microsoft YaHei', 14, 'bold'),
            bg=COLORS['bg_light'],
            fg=COLORS['text_light'],
            pady=15
        )
        self.status_label.pack(fill=tk.X)

    def create_footer(self):
        footer = tk.Frame(self.root, bg=COLORS['bg_light'], height=30)
        footer.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_bar = tk.Label(
            footer,
            text="就绪",
            font=('Microsoft YaHei', 9),
            bg=COLORS['bg_light'],
            fg=COLORS['text_light'],
            anchor='w'
        )
        self.status_bar.pack(side=tk.LEFT, padx=20)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            self.current_image_path = file_path
            self.load_image(file_path)
            self.detect_btn.config(state=tk.NORMAL)
            self.clear_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.DISABLED)

    def load_image(self, file_path):
        # 获取图片信息
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("错误", "无法读取图片！")
            return

        height, width = img.shape[:2]
        file_size = os.path.getsize(file_path) / 1024
        ext = os.path.splitext(file_path)[1].upper()[1:]

        # 更新基础信息
        for widget in self.basic_card.content_frame.winfo_children():
            text = widget.winfo_children()[0].cget("text")
            if "图片尺寸" in text:
                widget.winfo_children()[1].config(text=f"{width} × {height}")
            elif "文件大小" in text:
                widget.winfo_children()[1].config(text=f"{file_size:.1f} KB")
            elif "图片格式" in text:
                widget.winfo_children()[1].config(text=ext)

        # 重置识别结果
        self.face_count_label.config(text="0")
        self.confidence_label.config(text="0%")
        self.status_label.config(text="⏳ 等待检测", fg=COLORS['text_light'])
        self.detected_faces = []

        # 显示图片
        pixmap = Image.open(file_path)
        pixmap = self.resize_image(pixmap, 600, 450)

        self.photo = ImageTk.PhotoImage(pixmap)
        self.image_label.config(image=self.photo, text="", width=60, height=20)

        self.status_bar.config(text=f"已加载：{os.path.basename(file_path)}")

    def resize_image(self, image, max_width, max_height):
        w, h = image.size
        ratio = min(max_width / w, max_height / h)

        if ratio < 1:
            new_size = (int(w * ratio), int(h * ratio))
            image = image.resize(new_size, Image.LANCZOS)

        return image

    def start_detection(self):
        if not self.current_image_path or self.processing:
            return

        self.processing = True
        self.status_bar.config(text="🔍 正在检测人脸...")
        self.status_label.config(text="🔄 检测中...", fg=COLORS['warning'])
        self.detect_btn.config(state=tk.DISABLED)

        # 在新线程中执行检测
        thread = Thread(target=self.detect_faces)
        thread.daemon = True
        thread.start()

    def detect_faces(self):
        image = cv2.imread(self.current_image_path)
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

        avg_confidence = 0.0
        if len(confidents) > 0:
            avg_confidence = np.mean(confidents)

        self.detected_faces = face_rects.tolist()

        # 更新UI（需要在线程中使用after方法）
        self.root.after(0, self.update_detection_result, face_rects, confidents, avg_confidence)

    def update_detection_result(self, face_rects, confidents, avg_confidence):
        num_faces = len(face_rects)

        self.face_count_label.config(text=str(num_faces))
        self.confidence_label.config(text=f"{avg_confidence:.2f}%")

        if num_faces > 0:
            self.status_label.config(
                text=f"✅ 检测完成：发现 {num_faces} 个人脸",
                fg=COLORS['success']
            )
            self.save_btn.config(state=tk.NORMAL)
            self.draw_face_boxes()
        else:
            self.status_label.config(
                text="❌ 未检测到人脸",
                fg=COLORS['danger']
            )

        self.processing = False
        self.detect_btn.config(state=tk.NORMAL)
        self.status_bar.config(text=f"检测完成：发现 {num_faces} 个人脸")

    def draw_face_boxes(self):
        if not self.current_image_path or not self.detected_faces:
            return

        image = cv2.imread(self.current_image_path)

        for (x, y, w, h) in self.detected_faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (34, 197, 94), 3)
            cv2.putText(
                image, "Face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (34, 197, 94), 2
            )

        # 保存临时结果图
        temp_path = os.path.join(os.path.dirname(self.current_image_path), "temp_detected.jpg")
        cv2.imwrite(temp_path, image)

        # 显示带人脸框的图片
        pixmap = Image.open(temp_path)
        pixmap = self.resize_image(pixmap, 600, 450)

        self.photo = ImageTk.PhotoImage(pixmap)
        self.image_label.config(image=self.photo)

        # 删除临时文件
        try:
            os.remove(temp_path)
        except:
            pass

    def clear_all(self):
        self.current_image_path = None
        self.detected_faces = []

        self.image_label.config(image="", text="📷\n\n请选择待识别人像图片\n\n点击下方【选择图片】按钮上传", width=60, height=20)
        self.image_label.image = None

        # 重置基础信息
        for widget in self.basic_card.content_frame.winfo_children():
            widget.winfo_children()[1].config(text="---")

        # 重置识别结果
        self.face_count_label.config(text="0")
        self.confidence_label.config(text="0%")
        self.status_label.config(text="⏳ 等待检测", fg=COLORS['text_light'])

        # 禁用按钮
        self.detect_btn.config(state=tk.DISABLED)
        self.clear_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)

        self.status_bar.config(text="已清空")

    def save_result(self):
        if not self.current_image_path or not self.detected_faces:
            return

        save_path = filedialog.asksaveasfilename(
            title="保存结果图",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG 图片", "*.jpg"),
                ("PNG 图片", "*.png"),
                ("所有文件", "*.*")
            ],
            initialfile=os.path.splitext(os.path.basename(self.current_image_path))[0] + "_detected"
        )

        if save_path:
            image = cv2.imread(self.current_image_path)

            for (x, y, w, h) in self.detected_faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (34, 197, 94), 3)
                cv2.putText(
                    image, "Face",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (34, 197, 94), 2
                )

            cv2.imwrite(save_path, image)
            messagebox.showinfo("成功", f"结果图已保存到：\n{save_path}")
            self.status_bar.config(text=f"已保存：{os.path.basename(save_path)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionGUI(root)
    root.mainloop()