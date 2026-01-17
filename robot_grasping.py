# -*- coding: utf-8 -*-
"""
ROKAE 机械臂 AI 视觉抓取系统 V2 (坐标变换修正完整版)
功能：
1. 集成了 Eye-in-Hand 坐标变换的核心修正算法 (矩阵链乘法 + 轴向对齐)。
2. 包含完整的视觉感知流程 (YOLO + GraspNet + Open3D)。
3. 包含 LLM 任务规划接口。
"""

import cv2
import numpy as np
import open3d as o3d
import time
import sys
import os
import textwrap
import json
import requests
import base64
from scipy.spatial.transform import Rotation as R
from openai import OpenAI

# --- 路径适配 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# --- 导入驱动 (如果缺少环境会自动进入模拟模式) ---
try:
    from drivers.robot_arm_lib import RobotController, PGIA140
    from drivers.camera_realsense import RealsenseCamera
    from graspnet.graspnet import GraspBaseline 
    MOCK_MODE = False
except ImportError as e:
    print(f"!!! 驱动导入失败 ({e})，进入模拟模式 (Mock Mode) !!!")
    MOCK_MODE = True

# ================= 配置区 =================
ROBOT_IP = "192.168.0.160"
GRIPPER_PORT = "/dev/ttyUSB0"
YOLO_SERVER_URL = "http://127.0.0.1:5000/detect"

HAND_EYE_FILE = os.path.join("config", "hand_eye_result.txt")
API_KEY_FILE = os.path.join("config", "keys.txt")

# [关键参数] TCP Z轴长度 (法兰面到夹爪指尖的垂直距离，单位: 米)
# 请务必用游标卡尺准确测量！
TCP_Z_OFFSET = 0.20  

# [调试开关] 如果发现机械臂总是往反方向移动，将此设为 True
INVERT_HAND_EYE = False 

# 放置点与 Home 点 [x, y, z, rx, ry, rz]
DROP_POSE_XYZ = [0.3, -0.3, 0.2, 3.14, 0, 0] 

# LLM 配置
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-coder"
# ==========================================

# ---------------------------------------------------------
# 类 1: 坐标变换器 (修复定位问题的核心)
# ---------------------------------------------------------
class EyeInHandTransformer:
    def __init__(self, hand_eye_matrix_path, tcp_z_offset=0.20):
        self.tcp_offset = tcp_z_offset
        self.T_flange_cam = self._load_matrix(hand_eye_matrix_path)
        
        # [关键修正] GraspNet 坐标系 -> 机械臂 TCP 坐标系修正矩阵
        # GraspNet: X轴=接近方向, Y轴=夹爪开合方向
        # 机械臂TCP: Z轴=接近方向, Y轴=夹爪开合方向
        # 变换: 绕 Y 轴旋转 -90 度，将 X 轴转到 Z 轴位置
        self.R_grasp2robot = np.array([
            [ 0,  0, -1],
            [ 0,  1,  0],
            [ 1,  0,  0]
        ])

    def _load_matrix(self, path):
        try:
            if os.path.exists(path):
                mat = np.loadtxt(path)
                print(f">>> [Transform] 加载手眼矩阵:\n{mat}")
                return mat
            else:
                print(f"!!! [Transform] 未找到文件 {path}，使用单位矩阵")
                return np.eye(4)
        except Exception as e:
            print(f"!!! [Transform] 加载失败: {e}")
            return np.eye(4)

    def calculate_target_pose(self, T_base_flange_capture, grasp_pose_cam, invert_hand_eye=False):
        """
        计算机械臂目标法兰位姿 (Base Frame)
        :param T_base_flange_capture: 拍照瞬间的机械臂基座位姿 (4x4)
        :param grasp_pose_cam: GraspNet 输出的位姿 (RotationMatrix + Translation)
        """
        # 1. 构造 GraspNet 原始位姿矩阵 (T_cam_grasp)
        T_cam_grasp = np.eye(4)
        if hasattr(grasp_pose_cam, 'rotation_matrix'):
            T_cam_grasp[:3, :3] = grasp_pose_cam.rotation_matrix
            T_cam_grasp[:3, 3] = grasp_pose_cam.translation
        else:
            # 兼容模拟对象
            if hasattr(grasp_pose_cam, 'translation'):
                T_cam_grasp[:3, 3] = grasp_pose_cam.translation
                if hasattr(grasp_pose_cam, 'rotation_matrix'):
                     T_cam_grasp[:3, :3] = grasp_pose_cam.rotation_matrix

        # 2. 处理手眼矩阵方向
        if invert_hand_eye:
            T_hand_eye = np.linalg.inv(self.T_flange_cam)
        else:
            T_hand_eye = self.T_flange_cam

        # 3. [核心公式] 链式变换
        # Base -> Flange(Capture) -> Camera -> Grasp(Object)
        T_base_object_raw = T_base_flange_capture @ T_hand_eye @ T_cam_grasp
        
        print(f">>> [Debug] 物体在基座系下的原始坐标: {T_base_object_raw[:3, 3]}")

        # 4. [轴向对齐] GraspNet Frame -> Robot TCP Frame
        # 保持位置不变，旋转姿态对齐
        T_grasp_aligned = T_base_object_raw.copy()
        T_grasp_aligned[:3, :3] = T_base_object_raw[:3, :3] @ self.R_grasp2robot

        # 5. [约束优化] 强制 Z 轴垂直向下 (防止侧倾抓取)
        T_grasp_aligned = self._enforce_vertical_attack(T_grasp_aligned)

        # 6. [TCP 偏移补偿] 从指尖反推回法兰中心
        # T_flange_goal @ T_tcp = T_grasp_aligned
        T_flange_tcp = np.eye(4)
        T_flange_tcp[2, 3] = self.tcp_offset # TCP 长度 Z+
        
        T_base_flange_goal = T_grasp_aligned @ np.linalg.inv(T_flange_tcp)

        return T_base_flange_goal

    def _enforce_vertical_attack(self, T_matrix):
        """优化姿态：如果接近垂直向下，则强制垂直，并处理对称翻转"""
        R_curr = T_matrix[:3, :3]
        z_curr = R_curr[:, 2] # 当前接近方向
        
        # 阈值：如果 Z 分量 < -0.8 (说明大致朝下)
        if z_curr[2] < -0.8:
            z_new = np.array([0.0, 0.0, -1.0]) # 强制垂直向下
            y_curr = R_curr[:, 1] # 保持夹爪开合方向
            
            # 重建坐标系
            x_new = np.cross(y_curr, z_new)
            if np.linalg.norm(x_new) < 1e-6: return T_matrix # 奇异值保护
            x_new /= np.linalg.norm(x_new)
            y_new = np.cross(z_new, x_new)
            
            T_out = T_matrix.copy()
            T_out[:3, :3] = np.column_stack((x_new, y_new, z_new))
            
            # [对称翻转] 检查 Yaw 角，防止线缆缠绕
            # 如果机械臂手腕转到了后面 (>90度)，绕 Z 转 180 度回来
            r_temp = R.from_matrix(T_out[:3, :3])
            euler = r_temp.as_euler('zyx', degrees=True) # [rz, ry, rx]
            if abs(euler[0]) > 90:
                 rot_180 = R.from_euler('z', 180, degrees=True).as_matrix()
                 T_out[:3, :3] = T_out[:3, :3] @ rot_180
            return T_out
            
        return T_matrix

    def matrix_to_rokae_pose(self, matrix):
        """矩阵转 ROKAE 欧拉角 [x, y, z, rx, ry, rz]"""
        xyz = matrix[:3, 3]
        r = R.from_matrix(matrix[:3, :3])
        # 假设 ROKAE 使用标准的 ZYX 欧拉角
        euler = r.as_euler('zyx', degrees=False) 
        return list(xyz) + [euler[2], euler[1], euler[0]]

# ---------------------------------------------------------
# 类 2: 主控制系统
# ---------------------------------------------------------
class GraspSystemV2:
    def __init__(self):
        self.mock = MOCK_MODE
        
        # 1. 硬件连接
        self._init_hardware()
        
        # 2. 坐标转换器
        self.transformer = EyeInHandTransformer(HAND_EYE_FILE, TCP_Z_OFFSET)
        
        # 3. AI 初始化
        self._init_llm()
        
        # 4. 数据容器
        self.current_color = None
        self.current_depth = None
        self.T_capture = np.eye(4) # 拍照时的机械臂位姿

    def _init_hardware(self):
        if not self.mock:
            try:
                print(">>> 连接机械臂...")
                self.bot = RobotController(ROBOT_IP)
                self.bot.connect()
                self.home_pose = self.bot.get_current_pose()
                print(f"    Home Pose: {self.home_pose}")
                
                print(">>> 连接夹爪...")
                self.gripper = PGIA140(port=GRIPPER_PORT)
                self.gripper.connect()
                self.gripper.initialize()
                self.gripper.open_gripper()
                
                print(">>> 启动相机...")
                self.cam = RealsenseCamera()
                self.cam.start()
                
                print(">>> 加载 GraspNet...")
                self.grasp_net = GraspBaseline()
            except Exception as e:
                print(f"!!! 硬件初始化失败: {e}")
                self.mock = True

    def _init_llm(self):
        try:
            if os.path.exists(API_KEY_FILE):
                with open(API_KEY_FILE, 'r') as f: key = f.read().strip()
                self.llm = OpenAI(api_key=key, base_url=LLM_BASE_URL)
            else:
                self.llm = None
        except:
            self.llm = None

    def get_current_T_base_flange(self):
        """获取当前机械臂的变换矩阵"""
        if self.mock: 
            return np.array([[-1,0,0,0.3], [0,1,0,0.0], [0,0,-1,0.4], [0,0,0,1]])
            
        pose = self.bot.get_current_pose() # [x, y, z, rx, ry, rz]
        # ROKAE ZYX 欧拉角转换
        r = R.from_euler('zyx', [pose[5], pose[4], pose[3]], degrees=False)
        T = np.eye(4)
        T[:3, :3] = r.as_matrix()
        T[:3, 3] = pose[:3]
        return T

    def capture_scene_data(self):
        """采集 RGB-D 并生成点云"""
        if self.mock: return o3d.geometry.PointCloud()

        # [关键] 必须在读图前/后瞬间记录位姿
        self.T_capture = self.get_current_T_base_flange()
        
        f1, f2, _, _ = self.cam.get_frames()
        if hasattr(f1, 'get_data'): f1 = np.asanyarray(f1.get_data())
        if hasattr(f2, 'get_data'): f2 = np.asanyarray(f2.get_data())

        # 简单判断哪个是深度图 (uint16)
        if f1.dtype == np.uint8:
            color, depth = f1, f2
        else:
            color, depth = f2, f1
            
        depth = depth.astype(np.uint16)
        color = color.astype(np.uint8)
        
        # 格式统一
        if len(color.shape) == 2: color = cv2.cvtColor(color, cv2.COLOR_GRAY2RGB)
        elif color.shape[2] == 4: color = cv2.cvtColor(color, cv2.COLOR_BGRA2RGB)
        
        h, w = depth.shape[:2]
        if color.shape[:2] != (h, w): color = cv2.resize(color, (w, h))
        
        self.current_color = color
        self.current_depth = depth

        # 生成点云
        o3d_color = o3d.geometry.Image(color)
        o3d_depth = o3d.geometry.Image(depth)
        K, _ = self.cam.get_intrinsics()
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, K[0,0], K[1,1], K[0,2], K[1,2])
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, depth_scale=1000.0, depth_trunc=2.0, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
        pcd = pcd.voxel_down_sample(voxel_size=0.005)
        return pcd

    def detect_yolo_objects(self, classes):
        """调用 YOLO"""
        if self.mock or self.current_color is None: return []
        
        img_bgr = cv2.cvtColor(self.current_color, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', img_bgr)
        b64_str = base64.b64encode(buf).decode()
        
        try:
            res = requests.post(YOLO_SERVER_URL, json={'image': b64_str, 'classes': classes}, timeout=5)
            if res.status_code == 200:
                dets = res.json().get('detections', [])
                if dets: print(f">>> [YOLO] 检测到: {dets}")
                return dets
        except Exception as e:
            print(f"YOLO Error: {e}")
        return []

    def select_best_grasp(self, target_name):
        """视觉流程整合"""
        print(f"--- 寻找物体: {target_name} ---")
        pcd = self.capture_scene_data() # 此处会更新 self.T_capture
        if pcd is None: return None

        if self.mock:
            # 模拟返回
            class MockGrasp:
                rotation_matrix = np.eye(3)
                translation = np.array([0.0, 0.0, 0.4])
                score = 0.95
            return MockGrasp()

        # 1. YOLO 检测
        dets = self.detect_yolo_objects([target_name])
        if not dets:
            print(">>> 未检测到物体")
            return None
        
        # 取置信度最高的框
        box = max(dets, key=lambda x: x[4])
        x1, y1, x2, y2 = map(int, box[:4])
        
        # 2. GraspNet 推理
        try:
            gg = self.grasp_net.run(pcd, vis=False)
            gg.nms()
            gg.sort_by_score()
        except Exception as e:
            print(f"GraspNet Error: {e}")
            return None

        # 3. 筛选 (必须在 YOLO 框内)
        best_g = None
        K, _ = self.cam.get_intrinsics()
        margin = 20
        
        for g in gg:
            # 投影到像素坐标
            pt = g.translation
            px = (K @ pt.reshape(3,1)).flatten()
            u, v = int(px[0]/px[2]), int(px[1]/px[2])
            
            if (x1-margin <= u <= x2+margin) and (y1-margin <= v <= y2+margin):
                best_g = g
                break # 取分数最高的即可
                
        if best_g: print(f">>> 找到最佳抓取点 (Score: {best_g.score:.2f})")
        else: print(">>> 区域内无有效抓取点")
        
        return best_g

    def execute_grasp(self, grasp_in_cam):
        if grasp_in_cam is None: return

        print("\n>>> [执行] 开始计算坐标变换...")
        
        # 1. 计算目标位姿 (使用新的 Transformer)
        # 传入: 拍照时的机械臂位姿 + 相机下的抓取点
        T_target_flange = self.transformer.calculate_target_pose(
            self.T_capture, 
            grasp_in_cam, 
            invert_hand_eye=INVERT_HAND_EYE
        )
        
        # 2. 转为指令格式
        target_pose = self.transformer.matrix_to_rokae_pose(T_target_flange)
        
        print(f"--------------------------------------------------")
        print(f"目标法兰坐标 (Base系): {['%.3f'%x for x in target_pose[:3]]}")
        print(f"--------------------------------------------------")

        # 3. 安全检查
        if target_pose[2] < 0.03:
            print("!!! [警告] 目标过低 (<3cm)，存在撞击风险，已取消！")
            return

        # 4. 执行动作
        pre_pose = list(target_pose)
        pre_pose[2] += 0.12 # 抬高 12cm 作为预备点

        print("1. 移动到预备点...")
        if not self.mock: self.bot.move_l(pre_pose, speed=40)
        
        print("2. 下探抓取...")
        if not self.mock:
            self.gripper.open_gripper()
            self.bot.move_l(target_pose, speed=15) # 慢速
            self.gripper.close_gripper(force=60)
            time.sleep(1.0)
        
        print("3. 抬起...")
        if not self.mock: self.bot.move_l(pre_pose, speed=40)
        
        print("4. 放置...")
        if not self.mock:
            self.bot.move_l(DROP_POSE_XYZ, speed=50)
            self.gripper.open_gripper()
            time.sleep(0.5)
            
        print("5. 归位...")
        if not self.mock: self.bot.move_l(self.home_pose, speed=50)
        print(">>> 任务完成")

    def plan_from_llm(self, prompt):
        """LLM 规划 (生成调用代码)"""
        if not self.llm: return ""
        print(f"\n[AI] 思考中: {prompt}")
        tpl = """
        你是一个机械臂助手。可用函数:
        1. grasp_obj = get_grasp_candidates("物体名") 
        2. do_grasp(grasp_obj)
        请输出一个完整的 def plan(): 函数代码。不要Markdown。
        用户指令: "{task}"
        """
        try:
            res = self.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": tpl.replace("{task}", prompt)}]
            )
            code = res.choices[0].message.content.replace("```python","").replace("```","").strip()
            # 简单的代码提取逻辑
            if "def plan():" in code:
                return textwrap.dedent(code.split("def plan():")[1].split("\n", 1)[1]).strip()
            return code
        except Exception as e:
            print(f"LLM Error: {e}")
            return ""

# ================= 主程序入口 =================
if __name__ == "__main__":
    app = GraspSystemV2()
    
    # 封装给 LLM 的全局函数
    def get_grasp_candidates(name): return app.select_best_grasp(name)
    def do_grasp(obj): app.execute_grasp(obj)
    
    print("\n=== ROKAE AI 抓取系统 V2 ===")
    print("输入 'q' 退出，或直接输入指令 (例如: 把苹果抓起来)")
    
    while True:
        try:
            cmd = input("\n指令 > ").strip()
            if cmd in ['q', 'exit']: break
            if not cmd: continue
            
            # 1. 简单指令直接匹配
            if "抓" in cmd and not app.llm:
                # 无 LLM 时的降级处理
                target = "orange" # 默认测试物体
                print(f">>> [离线模式] 尝试抓取默认物体: {target}")
                g = app.select_best_grasp(target)
                app.execute_grasp(g)
                continue

            # 2. LLM 规划
            plan_code = app.plan_from_llm(cmd)
            if plan_code:
                print(f"--- Plan ---\n{plan_code}\n------------")
                try:
                    # 动态执行生成的代码
                    exec_scope = {'get_grasp_candidates': get_grasp_candidates, 'do_grasp': do_grasp, 'print': print}
                    exec(plan_code, exec_scope)
                except Exception as e:
                    print(f"执行出错: {e}")
            else:
                print("无法生成计划")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"系统异常: {e}")
            
    if not app.mock:
        app.bot.disconnect()
        app.cam.stop()