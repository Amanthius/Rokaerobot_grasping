# -*- coding: utf-8 -*-
"""
ROKAE 机械臂 + Realsense D435 眼在手 (Eye-in-Hand) 标定程序
优化整理版
"""

import cv2
import numpy as np
import time
import math
import os
import sys

# --- 模拟驱动 (如果没有实际硬件库，这段代码允许脚本运行演示逻辑) ---
# 在实际部署时，请确保 PYTHONPATH 能找到真实的 drivers
try:
    from drivers.camera_realsense import RealsenseCamera
    from drivers.robot_arm_lib import RobotController
    print(">>> 成功加载硬件驱动库。")
except ImportError:
    print(">>> [模拟模式] 未找到硬件驱动，使用 Mock 类代替。")
    class RobotController:
        def __init__(self, ip): self.ip = ip
        def connect(self): print(f"Mock Robot connected to {self.ip}")
        def disconnect(self): print("Mock Robot disconnected")
        def get_current_pose(self): return [0.3, 0, 0.4, 0, 0, 0] # 模拟 x,y,z,rx,ry,rz (rad)
        def move_l(self, pose, speed): print(f"Mock Move to {pose[:3]}...")

    class RealsenseCamera:
        def start(self): return True
        def stop(self): pass
        def get_intrinsics(self): 
            # 模拟 D435 内参
            K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
            D = np.zeros(5)
            return K, D
        def get_frames(self):
            # 返回黑色图像用于测试流程
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            return img, img, None, None

# ================= 配置区 =================
ROBOT_IP = "192.168.0.160"

# 标定板参数
BOARD_SIZE = (9, 6)     # 内角点数量 (行, 列)
SQUARE_SIZE = 0.022     # 方格边长 (米)

# 自动标定参数
AUTO_MOVE_SPEED = 20    
MOVE_WAIT_TIME = 1.5    # 增加一点延时以防抖动
OFFSET_POS = 0.04       # 平移偏移量 (米)
OFFSET_ROT = 15.0       # 旋转偏移量 (度)

DATA_DIR = "calibration_data"
RESULT_FILE = os.path.join("config", "hand_eye_result.txt")
# ==========================================

class HandEyeCalibration:
    def __init__(self):
        print(f">>> 正在连接机器人 ({ROBOT_IP})...")
        self.bot = RobotController(ROBOT_IP)
        self.bot.connect()
        
        print(">>> 正在启动 Realsense...")
        self.cam = RealsenseCamera()
        if not self.cam.start():
            print("!!! 相机启动失败")
            sys.exit(1)
        
        self.cam_matrix, self.dist_coeffs = self.cam.get_intrinsics()
        
        # 数据容器
        self.R_gripper2base = [] # 机械臂末端姿态 (旋转矩阵)
        self.t_gripper2base = [] # 机械臂末端位置 (平移向量)
        self.R_target2cam = []   # 标定板相对相机姿态
        self.t_target2cam = []   # 标定板相对相机位置

        # 准备 3D 世界坐标点 (Z=0)
        self.objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
        self.objp *= SQUARE_SIZE

        # 创建目录
        if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
        config_dir = os.path.dirname(RESULT_FILE)
        if config_dir and not os.path.exists(config_dir): os.makedirs(config_dir)

    def euler_to_matrix(self, r, p, y):
        """
        将欧拉角转换为旋转矩阵。
        假设输入单位为弧度 (Radians)。
        假设顺序为 ROKAE 常见的 Intrinsic ZYX (Rz * Ry * Rx)。
        """
        Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
        Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
        Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        return np.dot(Rz, np.dot(Ry, Rx))

    def generate_auto_trajectory(self, center_pose):
        """
        生成以当前点为中心的采样轨迹。
        注意：ROKAE get_current_pose 返回的单位如果是度，需要在此处转换，
        或者确保 OFFSET_ROT 与之单位一致。这里假设输入 center_pose 的旋转分量是弧度。
        """
        x, y, z, rx, ry, rz = center_pose
        d_p = OFFSET_POS
        d_r = math.radians(OFFSET_ROT) # 将度转换为弧度用于偏移计算
        
        # 生成 14 个采样点：单纯 XYZ 偏移 + 单纯 RPY 偏移 + 组合偏移
        offsets = [
            [0, 0, 0, 0, 0, 0],
            [d_p, 0, 0, 0, 0, 0], [-d_p, 0, 0, 0, 0, 0],
            [0, d_p, 0, 0, 0, 0], [0, -d_p, 0, 0, 0, 0],
            [0, 0, d_p, 0, 0, 0], [0, 0, -d_p, 0, 0, 0],
            [0, 0, 0, d_r, 0, 0], [0, 0, 0, -d_r, 0, 0],
            [0, 0, 0, 0, d_r, 0], [0, 0, 0, 0, -d_r, 0],
            [0, 0, 0, 0, 0, d_r], [0, 0, 0, 0, 0, -d_r],
            [d_p/2, d_p/2, 0, d_r, 0, 0], [-d_p/2, -d_p/2, 0, 0, d_r, 0],
        ]
        
        targets = []
        for off in offsets:
            targets.append([x+off[0], y+off[1], z+off[2], rx+off[3], ry+off[4], rz+off[5]])
        return targets

    def run_auto_sequence(self):
        print("\n>>> [自动模式] 启动...")
        start_pose = self.bot.get_current_pose()
        # TODO: 确认 start_pose 的角度单位。如果是度，需先转换为弧度再传入 generate
        # 假设驱动层已经处理为弧度
        trajectory = self.generate_auto_trajectory(start_pose)
        
        success_count = 0
        for i, target_pose in enumerate(trajectory):
            print(f"--- 移动到第 {i+1}/{len(trajectory)} 点 ---")
            self.bot.move_l(target_pose, speed=AUTO_MOVE_SPEED)
            
            # 等待机械臂稳定
            time.sleep(MOVE_WAIT_TIME)
            
            # 清空相机缓存（Realsense 有时会有帧缓冲）
            for _ in range(5): self.cam.get_frames()
            
            if self.process_frame(auto_mode=True):
                print(f">>> 第 {i+1} 点采集成功")
                success_count += 1
            else:
                print(f">>> [警告] 第 {i+1} 点角点检测失败")

        print(f"\n>>> 自动采集结束。成功: {success_count}/{len(trajectory)}")
        print(">>> 机械臂归位中...")
        self.bot.move_l(start_pose, speed=AUTO_MOVE_SPEED)

    def process_frame(self, auto_mode=False):
        """
        处理单帧：查找角点 -> PnP 解算 -> 记录数据
        """
        f1, f2, _, _ = self.cam.get_frames()
        
        # 兼容性处理：Realsense SDK 返回的可能是对象或 numpy 数组
        if hasattr(f1, 'get_data'): f1 = np.asanyarray(f1.get_data())
        if hasattr(f2, 'get_data'): f2 = np.asanyarray(f2.get_data())
        
        # 优先使用 RGB (f1)，如果 f1 为空则尝试 f2
        img = f1 if (isinstance(f1, np.ndarray) and f1.size > 0) else f2
        if img is None: return False

        # 确保是 BGR 格式
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)
        
        if auto_mode and not ret: return False

        if ret:
            # 1. 亚像素优化
            corners_sub = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), 
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            
            if auto_mode:
                # 2. PnP 解算 (IPPE 算法)
                # 计算 标定板(Target) 相对于 相机(Cam) 的位姿
                success, rvec, tvec = cv2.solvePnP(
                    self.objp, corners_sub, self.cam_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE
                )
                
                if success:
                    # 获取当前机械臂位姿 (End 相对于 Base)
                    # 假设返回的是 [x, y, z, rx, ry, rz(rad)]
                    robot_pose = self.bot.get_current_pose()
                    
                    R_g2b = self.euler_to_matrix(robot_pose[3], robot_pose[4], robot_pose[5])
                    t_g2b = np.array(robot_pose[:3]).reshape(3, 1)
                    
                    R_t2c, _ = cv2.Rodrigues(rvec)
                    
                    self.R_gripper2base.append(R_g2b)
                    self.t_gripper2base.append(t_g2b)
                    self.R_target2cam.append(R_t2c)
                    self.t_target2cam.append(tvec)
                    
                    # 保存采样图片用于后续 debug
                    idx = len(self.R_gripper2base)
                    cv2.imwrite(os.path.join(DATA_DIR, f"sample_{idx}.jpg"), img)
                    return True
        return ret

    def run_calibration_process(self):
        print("\n" + "="*60 + "\n ROKAE 手眼标定向导 (可视化验证版)\n" + "-"*60)
        print(" [重要] 坐标轴含义：")
        print("   - 红色 (X): 沿标定板长边")
        print("   - 绿色 (Y): 沿标定板短边")
        print("   - 蓝色 (Z): 垂直板面，正向应指向相机(朝向屏幕)")
        print("="*60)
        print(" 按 'a' -> 自动采集多组数据 (Auto)")
        print(" 按 's' -> 手动采集当前帧 (Single)")
        print(" 按 'c' -> 计算结果 (Calculate)")
        print(" 按 'q' -> 退出 (Quit)")

        while True:
            f1, f2, _, _ = self.cam.get_frames()
            # 模拟环境下的空数据处理
            if f1 is None and f2 is None:
                time.sleep(0.1)
                continue

            if hasattr(f1, 'get_data'): f1 = np.asanyarray(f1.get_data())
            
            # 这里简化处理，实际请保留原有的流判断逻辑
            img = f1
            if img is None: continue

            if len(img.shape) == 2: display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else: display_img = img.copy()
            
            gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)
            
            # --- 可视化绘制 ---
            if ret:
                corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                              (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                cv2.drawChessboardCorners(display_img, BOARD_SIZE, corners_sub, ret)
                
                # 实时解算 PnP 以绘制坐标轴
                success, rvec, tvec = cv2.solvePnP(self.objp, corners_sub, self.cam_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
                
                if success:
                    axis_len = SQUARE_SIZE * 3
                    # 绘制坐标轴
                    # OpenCV drawFrameAxes (需要 OpenCV 4.x+) 替代手动 line 绘制，更简单
                    try:
                        cv2.drawFrameAxes(display_img, self.cam_matrix, self.dist_coeffs, rvec, tvec, axis_len)
                    except AttributeError:
                        # 兼容旧版 OpenCV
                        points = np.float32([[axis_len,0,0], [0,axis_len,0], [0,0,axis_len], [0,0,0]]).reshape(-1,3)
                        imgpts, _ = cv2.projectPoints(points, rvec, tvec, self.cam_matrix, self.dist_coeffs)
                        imgpts = np.int32(imgpts).reshape(-1,2)
                        origin = tuple(imgpts[3])
                        cv2.line(display_img, origin, tuple(imgpts[0]), (0,0,255), 3) # X Red
                        cv2.line(display_img, origin, tuple(imgpts[1]), (0,255,0), 3) # Y Green
                        cv2.line(display_img, origin, tuple(imgpts[2]), (255,0,0), 3) # Z Blue
                    
                    # 显示 Z 轴距离 (深度)
                    cv2.putText(display_img, f"Dist: {tvec[2][0]:.3f}m", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            # ----------------

            cv2.putText(display_img, f"Captured: {len(self.R_gripper2base)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Hand-Eye Calibration", display_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('a'):
                if ret: 
                    cv2.destroyAllWindows()
                    self.run_auto_sequence()
            elif key == ord('s'):
                if self.process_frame(auto_mode=True): 
                    print(f">>> 手动采集成功: {len(self.R_gripper2base)}")
            elif key == ord('c'):
                self.calculate_result()
                break
            elif key == ord('q'):
                break

        if hasattr(self.bot, 'disconnect'): self.bot.disconnect()
        if hasattr(self.cam, 'stop'): self.cam.stop()
        cv2.destroyAllWindows()

    def calculate_result(self):
        print("\n>>> 开始计算手眼变换矩阵...")
        if len(self.R_gripper2base) < 5:
            print(">>> [警告] 数据点过少 (<5)，建议采集 10-15 组不同姿态的数据以保证精度。")
            
        try:
            # 核心函数: AX = XB
            # R_gripper2base: 机械臂基座到末端
            # R_target2cam:   相机到标定板
            # 输出: Cam 到 Gripper (End) 的变换
            R_c2g, t_c2g = cv2.calibrateHandEye(
                self.R_gripper2base, self.t_gripper2base,
                self.R_target2cam, self.t_target2cam,
                method=cv2.CALIB_HAND_EYE_DANIILIDIS
            )
            
            H_cam2end = np.eye(4)
            H_cam2end[:3, :3] = R_c2g
            H_cam2end[:3, 3] = t_c2g.ravel()

            print("\n" + "#"*40 + "\n [标定结果 (T_End_Cam)] \n" + "#"*40)
            print("格式: 齐次变换矩阵 4x4")
            print(np.array2string(H_cam2end, separator=', ', suppress_small=True))
            
            # 计算平移距离（方便人眼验证）
            dist = np.linalg.norm(t_c2g)
            print(f"\n[验证] 相机光心距离法兰中心约: {dist*1000:.2f} mm")
            
            np.savetxt(RESULT_FILE, H_cam2end, fmt='%.6f')
            print(f"结果已保存至: {RESULT_FILE}")
            
        except Exception as e:
            print(f"!!! 计算失败: {e}")

if __name__ == "__main__":
    app = HandEyeCalibration()
    app.run_calibration_process()