# -*- coding: utf-8 -*-
"""
Realsense D435 计算机视觉实验集 (最终修正版 - 智能用户管理)
本文件集成了基于Realsense D435深度摄像头的多个OpenCV计算机视觉实验。

[!!] 重要环境配置 [!!]
此脚本中的人脸识别(Face Recognition)和ArUco标记功能需要OpenCV的扩展模块。
请务必通过以下命令安装正确的库:
pip uninstall opencv-python
pip install opencv-contrib-python
或
conda install -c conda-forge opencv
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import math
import json # 引入JSON库用于读写用户信息

class RealsenseCamera:
    """
    一个封装了Realsense摄像头基本操作的类，用于简化代码。
    """
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.profile = None
        self.depth_scale = 0.0

    def start(self):
        """启动摄像头并获取参数"""
        print("正在启动Realsense摄像头...")
        try:
            self.profile = self.pipeline.start(self.config)
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print(f"摄像头已启动。深度缩放因子: {self.depth_scale}")
            # 等待自动曝光/增益稳定
            for _ in range(30):
                self.pipeline.wait_for_frames()
            return True
        except Exception as e:
            print(f"[错误] 无法启动Realsense摄像头: {e}")
            print("请检查: 1.摄像头是否连接正常; 2.Intel RealSense SDK 2.0是否已安装。")
            return False

    def get_frames(self):
        """获取对齐的彩色帧和深度帧"""
        try:
            frames = self.pipeline.wait_for_frames(5000) # 5秒超时
            align = rs.align(rs.stream.color)
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                return None, None, None, None

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            return depth_frame, color_frame, depth_image, color_image
        except Exception as e:
            print(f"[错误] 获取帧失败: {e}")
            return None, None, None, None


    def get_intrinsics(self):
        """获取相机内参"""
        if self.profile is None:
            print("错误: 摄像头未启动，无法获取内参。")
            return None, None
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        cam_matrix = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ])
        dist_coeffs = np.array(intr.coeffs)
        return cam_matrix, dist_coeffs

    def stop(self):
        """停止摄像头"""
        print("正在关闭摄像头...")
        self.pipeline.stop()
        cv2.destroyAllWindows()

# --- Helper function to display error for contrib modules ---
def handle_contrib_error():
    """Prints a detailed error message for missing OpenCV contrib modules."""
    print("\n" + "#"*50)
    print(" [致命错误] OpenCV扩展模块缺失")
    print(" " + "-"*47)
    print(" 错误原因: 您安装的OpenCV版本不包含 'face' 模块。")
    print("          人脸识别等功能属于扩展模块(contrib)。")
    print("\n [解决方案]")
    print(" 1. 打开终端或命令提示符。")
    print(" 2. 卸载当前版本: pip uninstall opencv-python opencv-contrib-python")
    print(" 3. 安装完整版本: pip install opencv-contrib-python")
    print("    或使用Conda: conda install -c conda-forge opencv")
    print("#"*50 + "\n")

# --- 实验函数定义 ---

def run_environment_validation():
    """
    实验0: 环境验证
    """
    print("\n--- 正在运行实验0: 环境验证 ---")
    cam = RealsenseCamera()
    if not cam.start():
        return
    print("环境正常。按 'q' 键退出。")
    try:
        while True:
            _, _, depth_image, color_image = cam.get_frames()
            if color_image is None:
                continue
            
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow('RealSense Color Stream', color_image)
            cv2.imshow('RealSense Depth Stream', depth_colormap)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.stop()

def run_face_detection():
    """
    实验2.1: 人脸检测 (Face Detection)
    """
    print("\n--- 正在运行实验2.1: 人脸检测 ---")
    cascade_path = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        print(f"错误: 未找到 '{cascade_path}' 文件。")
        print("请从OpenCV官方仓库下载该文件并放置于脚本同级目录。")
        return

    face_cascade = cv2.CascadeClassifier(cascade_path)
    cam = RealsenseCamera()
    if not cam.start():
        return
    print("人脸检测已启动。按 'q' 键退出。")

    try:
        while True:
            _, _, _, color_image = cam.get_frames()
            if color_image is None:
                continue

            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(color_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            cv2.imshow('Face Detection', color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.stop()

def collect_face_data():
    """
    实验2.2 - 步骤1: 收集人脸数据
    """
    print("\n--- 正在运行实验2.2 - 步骤1: 收集人脸数据 ---")
    cascade_path = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        print(f"错误: '{cascade_path}' 不存在。")
        return
        
    face_detector = cv2.CascadeClassifier(cascade_path)
    
    face_id = input("请输入您的用户ID (纯数字，例如: 1): ")
    user_name = input("请输入您的名字 (拼音或英文，例如: Tom): ")
    print(f"用户 {user_name}(ID:{face_id})，请正对摄像头，程序将自动采集30张面部样本...")
    
    # --- 新增: 自动保存用户ID和姓名映射 ---
    user_map_file = 'user_map.json'
    user_map = {}
    if os.path.exists(user_map_file):
        with open(user_map_file, 'r', encoding='utf-8') as f:
            user_map = json.load(f)
    
    # JSON的键必须是字符串
    user_map[str(face_id)] = user_name
    
    with open(user_map_file, 'w', encoding='utf-8') as f:
        json.dump(user_map, f, indent=4, ensure_ascii=False)
    print(f"用户信息已保存至 {user_map_file}")
    # --- 新增结束 ---

    dataset_path = 'dataset'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    cam = RealsenseCamera()
    if not cam.start():
        return
    count = 0

    try:
        while count < 30:
            _, _, _, color_image = cam.get_frames()
            if color_image is None:
                continue

            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(color_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                count += 1
                file_path = os.path.join(dataset_path, f"User.{face_id}.{count}.jpg")
                cv2.imwrite(file_path, gray[y:y+h, x:x+w])
                print(f"已采集样本: {count}/30")
                cv2.putText(color_image, f"Samples: {count}/30", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow('Collecting Data', color_image)
            if cv2.waitKey(100) & 0xFF == ord('q'): # Add a small delay
                break
    finally:
        print("数据采集完成。")
        cam.stop()

def train_face_model():
    """
    实验2.2 - 步骤2: 训练人脸识别模型
    """
    print("\n--- 正在运行实验2.2 - 步骤2: 训练模型 ---")
    path = 'dataset'
    if not os.path.exists(path) or not os.listdir(path):
        print("错误: 'dataset' 文件夹为空或不存在，请先运行数据采集程序。")
        return

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        handle_contrib_error()
        return
        
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def get_images_and_labels(path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []
        for image_path in image_paths:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            face_samples.append(img)
            id_str = os.path.split(image_path)[-1].split(".")[1]
            ids.append(int(id_str))
        return face_samples, ids

    print("正在从数据集中读取图像并训练模型... 这可能需要一些时间。")
    faces, ids = get_images_and_labels(path)
    if not faces:
        print("数据集中未找到有效图像，训练中止。")
        return
        
    recognizer.train(faces, np.array(ids))

    trainer_dir = 'trainer'
    if not os.path.exists(trainer_dir):
        os.makedirs(trainer_dir)
    recognizer.write(os.path.join(trainer_dir, 'trainer.yml'))

    print(f"模型训练完成。已发现 {len(np.unique(ids))} 位用户。模型已保存到 trainer/trainer.yml")

def recognize_faces():
    """
    实验2.2 - 步骤3: 实时人脸识别
    """
    print("\n--- 正在运行实验2.2 - 步骤3: 实时人脸识别 ---")
    trainer_path = 'trainer/trainer.yml'
    cascade_path = 'haarcascade_frontalface_default.xml'
    user_map_file = 'user_map.json'

    if not os.path.exists(trainer_path) or not os.path.exists(cascade_path):
        print("错误: 缺少 'trainer/trainer.yml' 或 'haarcascade_frontalface_default.xml'。")
        print("请先运行数据采集和模型训练程序。")
        return
    
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        handle_contrib_error()
        return

    recognizer.read(trainer_path)
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # --- 修改: 自动从文件加载用户姓名 ---
    names = {}
    if os.path.exists(user_map_file):
        with open(user_map_file, 'r', encoding='utf-8') as f:
            names = json.load(f)
        print("已成功从 'user_map.json' 加载用户姓名映射。")
    else:
        print("警告: 未找到 'user_map.json' 文件。将只显示用户ID。")
    # --- 修改结束 ---

    cam = RealsenseCamera()
    if not cam.start():
        return
    print("人脸识别已启动。按 'q' 键退出。")

    try:
        while True:
            _, _, _, color_image = cam.get_frames()
            if color_image is None: continue

            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))

            for (x, y, w, h) in faces:
                cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                id_val, confidence = recognizer.predict(gray[y:y+h, x:x+w])

                if confidence < 100:
                    # 使用ID的字符串形式在字典中查找姓名
                    name = names.get(str(id_val), f"ID:{id_val}")
                    confidence_text = f"  {round(100 - confidence)}%"
                else:
                    name = "Unknown"
                    confidence_text = ""

                cv2.putText(color_image, name + confidence_text, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Face Recognition', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.stop()

def run_human_body_detection():
    print("\n--- 正在运行实验2.3: 人体检测 ---")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cam = RealsenseCamera()
    if not cam.start(): return
    print("人体检测已启动。按 'q' 键退出。")
    try:
        while True:
            _, _, _, color_image = cam.get_frames()
            if color_image is None: continue
            rects, weights = hog.detectMultiScale(color_image, winStride=(4, 4), padding=(8, 8), scale=1.05)
            for (x, y, w, h) in rects:
                cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.imshow('Human Body Detection', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally: cam.stop()

def run_gesture_recognition():
    print("\n--- 正在运行实验2.4: 手势识别 (手指计数) ---")
    cam = RealsenseCamera()
    if not cam.start(): return
    print("手势识别已启动。将手放在离摄像头约0.01-0.27米处。按 'q' 键退出。")
    try:
        while True:
            _, _, depth_image, color_image = cam.get_frames()
            if color_image is None: continue
            depth_mask = cv2.inRange(depth_image, 10, 270)
            kernel = np.ones((5, 5), np.uint8)
            depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            depth_mask = cv2.GaussianBlur(depth_mask, (7, 7), 0)
            contours, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            finger_count = 0
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                if cv2.contourArea(cnt) > 2000:
                    cv2.drawContours(color_image, [cnt], -1, (0, 255, 0), 2)
                    hull = cv2.convexHull(cnt, returnPoints=False)
                    if hull is not None and len(hull) > 3 and len(cnt) > 3:
                        defects = cv2.convexityDefects(cnt, hull)
                        count_defects = 0
                        if defects is not None:
                            for i in range(defects.shape[0]):
                                s, e, f, d = defects[i, 0]
                                start = tuple(cnt[s][0]); end = tuple(cnt[e][0]); far = tuple(cnt[f][0])
                                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                                angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 57
                                if angle <= 90:
                                    count_defects += 1
                                    cv2.circle(color_image, far, 5, [0, 0, 255], -1)
                        finger_count = count_defects + 1
            cv2.putText(color_image, f"Fingers: {finger_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Gesture Recognition', color_image)
            cv2.imshow('Depth Mask', depth_mask)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally: cam.stop()

def run_hsv_tool():
    print("\n--- 正在运行实验3.1: HSV颜色查找工具 ---")
    def nothing(x): pass
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("H_min", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("S_min", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("V_min", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("H_max", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("S_max", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("V_max", "Trackbars", 255, 255, nothing)
    cam = RealsenseCamera()
    if not cam.start(): return
    print("请调整Trackbars窗口中的滑块来选择颜色。按 'q' 键退出。")
    try:
        while True:
            _, _, _, color_image = cam.get_frames()
            if color_image is None: continue
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            h_min = cv2.getTrackbarPos("H_min", "Trackbars"); s_min = cv2.getTrackbarPos("S_min", "Trackbars"); v_min = cv2.getTrackbarPos("V_min", "Trackbars")
            h_max = cv2.getTrackbarPos("H_max", "Trackbars"); s_max = cv2.getTrackbarPos("S_max", "Trackbars"); v_max = cv2.getTrackbarPos("V_max", "Trackbars")
            lower = np.array([h_min, s_min, v_min]); upper = np.array([h_max, s_max, v_max])
            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(color_image, color_image, mask=mask)
            cv2.imshow("Original", color_image); cv2.imshow("Mask", mask); cv2.imshow("Result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"\n记录下的HSV值:\nLower: H={h_min}, S={s_min}, V={v_min}\nUpper: H={h_max}, S={s_max}, V={v_max}")
                break
    finally: cam.stop()

import numpy as np
import cv2
# 假设 RealsenseCamera 类在其他地方定义或导入
# from your_camera_module import RealsenseCamera 

def run_color_tracking():
    """
    运行颜色跟踪。
    
    参数:
    color_lower (np.array): 颜色的HSV下限值 [H, S, V]
    color_upper (np.array): 颜色的HSV上限值 [H, S, V]
    """
    print("\n--- 正在运行实验3.2: 颜色跟踪 ---")
        # --- 在这里自定义你要追踪的颜色 ---
    # 你需要在这里设置HSV的下限和上限
    
    # 示例 1: 追踪绿色 (你的原始值)
    color_lower = np.array([35, 80, 80])
    color_upper = np.array([85, 255, 255])
    
    # 示例 2: 追踪蓝色 (取消下面两行的注释来使用)
    #color_lower = np.array([100, 80, 80])
    #color_upper = np.array([130, 255, 255])
    
    # 示例 3: 追踪红色 (红色比较特殊，H值在0附近，可能需要两个范围)
    # 这是一个简单的红色范围
    # color_lower = np.array([170, 80, 80])
    # color_upper = np.array([180, 255, 255])
    print(f"追踪的HSV范围: Lower={color_lower}, Upper={color_upper}")

    cam = RealsenseCamera()
    if not cam.start(): 
        print("相机启动失败")
        return

    print("正在跟踪指定颜色的物体。按 'q' 键退出。")
    try:
        while True:
            _, _, _, color_image = cam.get_frames()
            if color_image is None: 
                continue

            blurred = cv2.GaussianBlur(color_image, (11, 11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            
            # 1. 使用传入的参数来创建掩码
            mask = cv2.inRange(hsv, color_lower, color_upper)
            
            # 2. 形态学操作
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            # 3. 查找轮廓
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            center = None # 初始化center
            
            if len(contours) > 0:
                # 找到最大的轮廓
                c = max(contours, key=cv2.contourArea)
                # 计算最小外接圆
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                # 计算轮廓的矩
                M = cv2.moments(c)
                
                # 确保m00不为0以避免除零错误
                if M["m00"] > 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                
                # 仅在半径大于某个值时才绘制（避免噪点）
                if radius > 10:
                    # 绘制外接圆
                    cv2.circle(color_image, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    # 绘制中心点
                    if center:
                        cv2.circle(color_image, center, 5, (0, 0, 255), -1)
                        
            cv2.imshow("Color Tracking", color_image)
            # cv2.imshow("Mask", mask) # 你也可以取消这行注释来查看掩码
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
    finally:
        print("停止跟踪...")
        cam.stop()

    


def run_edge_detection():
    print("\n--- 正在运行实验4.1: Canny边缘检测 ---")
    cam = RealsenseCamera()
    if not cam.start(): return
    print("边缘检测已启动。按 'q' 键退出。")
    try:
        while True:
            _, _, _, color_image = cam.get_frames()
            if color_image is None: continue
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            cv2.imshow("Original", color_image); cv2.imshow("Edges", edges)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally: cam.stop()

def run_qr_code_recognition():
    print("\n--- 正在运行实验4.2: QR二维码识别 ---")
    qr_detector = cv2.QRCodeDetector()
    cam = RealsenseCamera()
    if not cam.start(): return
    print("QR码识别已启动。按 'q' 键退出。")
    try:
        while True:
            _, _, _, color_image = cam.get_frames()
            if color_image is None: continue
            decoded_info, points, _ = qr_detector.detectAndDecode(color_image)
            if points is not None:
                points = np.int32(points.reshape(-1, 2))
                cv2.polylines(color_image, [points], True, (0, 255, 0), 3)
                if decoded_info:
                    print(f"Decoded data: {decoded_info}")
                    cv2.putText(color_image, decoded_info, (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("QR Code Recognition", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally: cam.stop()

def run_aruco_pose_estimation():
    print("\n--- 正在运行实验4.3 (深度修正版): ArUco标记检测与姿态估计 ---")
    cam = RealsenseCamera()
    if not cam.start(): return
    cam_matrix, dist_coeffs = cam.get_intrinsics()
    if cam_matrix is None: cam.stop(); return
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        print("Aruco检测器创建成功。")
    except AttributeError:
        handle_contrib_error()
        cam.stop()
        return
    marker_length_meters = 0.05 
    obj_points = np.array([
        [-marker_length_meters / 2, marker_length_meters / 2, 0],
        [marker_length_meters / 2, marker_length_meters / 2, 0],
        [marker_length_meters / 2, -marker_length_meters / 2, 0],
        [-marker_length_meters / 2, -marker_length_meters / 2, 0]
    ], dtype=np.float32)
    print("ArUco姿态估计已启动。按 'q' 键退出。")
    try:
        while True:
            _, _, _, color_image = cam.get_frames()
            if color_image is None: continue
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
                for i in range(len(ids)):
                    success, rvec, tvec = cv2.solvePnP(obj_points, corners[i], cam_matrix, dist_coeffs)
                    if success:
                        cv2.drawFrameAxes(color_image, cam_matrix, dist_coeffs, rvec, tvec, marker_length_meters / 2)
                        distance = np.linalg.norm(tvec)
                        cv2.putText(color_image, f"ID: {ids[i][0]} Dist: {distance:.2f}m", tuple(corners[i][0][0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            else:
                cv2.putText(color_image, "No markers detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("ArUco Pose Estimation (Fixed)", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally: cam.stop()

# --- 主程序入口 ---
def main():
    """
    主菜单函数，用于选择并运行不同的实验。
    """
    while True:
        print("\n" + "="*40)
        print("  基于Realsense D435的计算机视觉实验菜单")
        print("-"*40)
        print(" 0. 环境验证")
        print("\n--- 物体识别类 ---")
        print(" 1. 人脸检测 (Haar)")
        print(" 2. 人体检测 (HOG)")
        print(" 3. 手势识别 (手指计数)")
        print("\n--- 颜色识别类 ---")
        print(" 4. HSV颜色查找工具")
        print(" 5. 颜色跟踪")
        print("\n--- 基础检测 ---")
        print(" 6. Canny边缘检测")
        print(" 7. QR二维码识别")
        print(" 8. ArUco标记姿态估计")
        print("\n--- 人脸识别套件 (自动用户管理) ---")
        print(" 9. [步骤1] 收集人脸数据")
        print("10. [步骤2] 训练人脸模型")
        print("11. [步骤3] 实时人脸识别")
        print("\n q. 退出程序")
        print("="*40)
        
        choice = input("请输入您想运行的实验编号: ")

        menu = {
            '0': run_environment_validation,
            '1': run_face_detection,
            '2': run_human_body_detection,
            '3': run_gesture_recognition,
            '4': run_hsv_tool,
            '5': run_color_tracking,
            '6': run_edge_detection,
            '7': run_qr_code_recognition,
            '8': run_aruco_pose_estimation,
            '9': collect_face_data,
            '10': train_face_model,
            '11': recognize_faces
        }

        if choice.lower() == 'q':
            print("程序已退出。")
            break
        
        action = menu.get(choice)
        if action:
            action()
        else:
            print("无效输入，请输入列表中的编号。")

if __name__ == "__main__":
    main()

