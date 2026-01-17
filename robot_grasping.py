# -*- coding: utf-8 -*-
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

# --- 环境依赖检查 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    # 尝试加载机器人驱动、相机SDK及算法模型
    from drivers.robot_arm_lib import RobotController, PGIA140
    from drivers.camera_realsense import RealsenseCamera
    from graspnet.graspnet import GraspBaseline 
    MOCK_MODE = False
except ImportError as e:
    # 环境缺失时自动切换至模拟模式，便于离线调试逻辑
    print(f"!!! 硬件驱动或模型库加载失败 ({e})，当前运行在：模拟模式 (Mock Mode) !!!")
    MOCK_MODE = True

# ================= 核心硬件与算法配置 =================
ROBOT_IP = "192.168.0.160"          # 珞石机器人控制箱IP
GRIPPER_PORT = "/dev/ttyUSB0"       # 夹爪串行通信接口
YOLO_SERVER_URL = "http://127.0.0.1:5000/detect" # 视觉检测服务器地址

# 配置文件路径
HAND_EYE_FILE = os.path.join("config", "hand_eye_result.txt") # 手眼标定矩阵文件
API_KEY_FILE = os.path.join("config", "keys.txt")              # LLM 密钥文件

# 关键物理参数：工具中心点 (TCP) Z轴偏移
# 定义：从机器人法兰盘中心到夹爪指尖末端的垂直距离
TCP_Z_OFFSET = 0.20  

# 坐标系镜像修正开关
INVERT_HAND_EYE = False 

# 预定义的放置点坐标 [x, y, z, rx, ry, rz] (单位：米/弧度)
DROP_POSE_XYZ = [0.3, -0.3, 0.2, 3.14, 0, 0] 

# 大语言模型(LLM)接口配置
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-coder"
# ===================================================

class EyeInHandTransformer:
    """
    眼在手上 (Eye-in-Hand) 空间坐标变换模块
    负责将相机坐标系下的视觉抓取位姿映射到机器人基座坐标系
    """
    def __init__(self, hand_eye_matrix_path, tcp_z_offset=0.20):
        self.tcp_offset = tcp_z_offset
        # 加载 4x4 手眼标定矩阵 (T_flange_cam)
        self.T_flange_cam = self._load_matrix(hand_eye_matrix_path)
        
        # 算法坐标系与机器人坐标系对齐矩阵
        # GraspNet输出：X轴为接近方向；机器人TCP：Z轴为接近方向
        # 需绕Y轴旋转-90度，完成轴向一致性转换
        self.R_grasp2robot = np.array([
            [ 0,  0, -1],
            [ 0,  1,  0],
            [ 1,  0,  0]
        ])

    def _load_matrix(self, path):
        """从文件读取标定矩阵，若失败则返回单位阵"""
        try:
            if os.path.exists(path):
                mat = np.loadtxt(path)
                return mat
            return np.eye(4)
        except Exception as e:
            print(f"矩阵加载异常: {e}")
            return np.eye(4)

    def calculate_target_pose(self, T_base_flange_capture, grasp_pose_cam, invert_hand_eye=False):
        """
        核心变换公式：
        T_{base\_goal} = T_{base\_flange(capture)} * T_{flange\_cam} * T_{cam\_grasp} * T_{grasp\_align} * T_{tcp\_offset}^{-1}
        """
        # 1. 构建相机坐标系下的目标抓取位姿矩阵
        T_cam_grasp = np.eye(4)
        if hasattr(grasp_pose_cam, 'rotation_matrix'):
            T_cam_grasp[:3, :3] = grasp_pose_cam.rotation_matrix
            T_cam_grasp[:3, 3] = grasp_pose_cam.translation
        else:
            if hasattr(grasp_pose_cam, 'translation'):
                T_cam_grasp[:3, 3] = grasp_pose_cam.translation

        # 2. 确定手眼变换方向
        T_hand_eye = np.linalg.inv(self.T_flange_cam) if invert_hand_eye else self.T_flange_cam

        # 3. 计算物体在机器人基座坐标系下的原始位姿
        T_base_object_raw = T_base_flange_capture @ T_hand_eye @ T_cam_grasp
        
        # 4. 执行坐标轴对齐，确保机械臂末端姿态符合工程习惯
        T_grasp_aligned = T_base_object_raw.copy()
        T_grasp_aligned[:3, :3] = T_base_object_raw[:3, :3] @ self.R_grasp2robot

        # 5. 姿态约束：强制垂直向下抓取并防止关节限位
        T_grasp_aligned = self._enforce_vertical_attack(T_grasp_aligned)

        # 6. 工具偏移补偿：计算法兰盘中心应到达的位置
        T_flange_tcp = np.eye(4)
        T_flange_tcp[2, 3] = self.tcp_offset 
        
        T_base_flange_goal = T_grasp_aligned @ np.linalg.inv(T_flange_tcp)

        return T_base_flange_goal

    def _enforce_vertical_attack(self, T_matrix):
        """
        姿态优化逻辑：
        若识别到的抓取方向大致朝下，则将其强制修正为垂直向下，并处理Yaw角以防线缆缠绕
        """
        R_curr = T_matrix[:3, :3]
        z_curr = R_curr[:, 2] # 当前Z轴向量
        
        # 判断是否接近垂直向下 (阈值 -0.8)
        if z_curr[2] < -0.8:
            z_new = np.array([0.0, 0.0, -1.0]) 
            y_curr = R_curr[:, 1] 
            
            # 正交化重建坐标系
            x_new = np.cross(y_curr, z_new)
            if np.linalg.norm(x_new) < 1e-6: return T_matrix 
            x_new /= np.linalg.norm(x_new)
            y_new = np.cross(z_new, x_new)
            
            T_out = T_matrix.copy()
            T_out[:3, :3] = np.column_stack((x_new, y_new, z_new))
            
            # 旋转镜像处理：若手腕旋转超过90度，自动翻转180度以优化路径
            r_temp = R.from_matrix(T_out[:3, :3])
            euler = r_temp.as_euler('zyx', degrees=True) 
            if abs(euler[0]) > 90:
                 rot_180 = R.from_euler('z', 180, degrees=True).as_matrix()
                 T_out[:3, :3] = T_out[:3, :3] @ rot_180
            return T_out
            
        return T_matrix

    def matrix_to_rokae_pose(self, matrix):
        """将 4x4 变换矩阵转换为 ROKAE 控制器所需的 [x, y, z, rx, ry, rz] 格式"""
        xyz = matrix[:3, 3]
        r = R.from_matrix(matrix[:3, :3])
        euler = r.as_euler('zyx', degrees=False) # ROKAE 通常采用 ZYX 欧拉角顺规
        return list(xyz) + [euler[2], euler[1], euler[0]]

class RoboticVisionSystem:
    """
    机器人视觉抓取集成系统类
    封装了从感知、决策到执行的全流程
    """
    def __init__(self):
        self.mock = MOCK_MODE
        self._init_hardware()
        self.transformer = EyeInHandTransformer(HAND_EYE_FILE, TCP_Z_OFFSET)
        self._init_llm()
        
        # 缓存当前场景的视觉与位姿数据
        self.current_color = None
        self.current_depth = None
        self.T_capture = np.eye(4) 

    def _init_hardware(self):
        """初始化机械臂、夹爪与相机硬件"""
        if not self.mock:
            try:
                self.bot = RobotController(ROBOT_IP)
                self.bot.connect()
                self.home_pose = self.bot.get_current_pose()
                
                self.gripper = PGIA140(port=GRIPPER_PORT)
                self.gripper.connect()
                self.gripper.initialize()
                self.gripper.open_gripper()
                
                self.cam = RealsenseCamera()
                self.cam.start()
                
                self.grasp_net = GraspBaseline()
                print(">>> 硬件系统初始化完毕")
            except Exception as e:
                print(f"!!! 硬件初始化失败: {e}，自动降级至模拟模式")
                self.mock = True

    def _init_llm(self):
        """初始化大模型 API 客户端"""
        try:
            if os.path.exists(API_KEY_FILE):
                with open(API_KEY_FILE, 'r') as f: key = f.read().strip()
                self.llm = OpenAI(api_key=key, base_url=LLM_BASE_URL)
            else:
                self.llm = None
        except:
            self.llm = None

    def get_current_T_base_flange(self):
        """实时获取并转换当前机械臂末端的变换矩阵"""
        if self.mock: 
            return np.array([[-1,0,0,0.3], [0,1,0,0.0], [0,0,-1,0.4], [0,0,0,1]])
            
        pose = self.bot.get_current_pose()
        r = R.from_euler('zyx', [pose[5], pose[4], pose[3]], degrees=False)
        T = np.eye(4)
        T[:3, :3] = r.as_matrix()
        T[:3, 3] = pose[:3]
        return T

    def capture_scene_data(self):
        """同步采集 RGB-D 图像并生成 3D 点云"""
        if self.mock: return o3d.geometry.PointCloud()

        # 核心：必须在图像采集的同一时刻记录机械臂位姿，确保坐标变换同步
        self.T_capture = self.get_current_T_base_flange()
        
        f1, f2, _, _ = self.cam.get_frames()
        # 处理不同的相机返回格式
        color = np.asanyarray(f1.get_data()) if hasattr(f1, 'get_data') else f1
        depth = np.asanyarray(f2.get_data()) if hasattr(f2, 'get_data') else f2

        # 自动对齐深度图与彩色图数据格式
        if color.dtype != np.uint8: color, depth = depth, color
        depth = depth.astype(np.uint16)
        color = color.astype(np.uint8)
        
        h, w = depth.shape[:2]
        if color.shape[:2] != (h, w): color = cv2.resize(color, (w, h))
        
        self.current_color, self.current_depth = color, depth

        # 利用 Open3D 生成点云
        o3d_color = o3d.geometry.Image(color)
        o3d_depth = o3d.geometry.Image(depth)
        K, _ = self.cam.get_intrinsics()
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, K[0,0], K[1,1], K[0,2], K[1,2])
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, depth_scale=1000.0, depth_trunc=2.0, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
        return pcd.voxel_down_sample(voxel_size=0.005)

    def detect_yolo_objects(self, classes):
        """通过外部 YOLO 服务获取 2D 检测框"""
        if self.mock or self.current_color is None: return []
        
        img_bgr = cv2.cvtColor(self.current_color, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', img_bgr)
        b64_str = base64.b64encode(buf).decode()
        
        try:
            res = requests.post(YOLO_SERVER_URL, json={'image': b64_str, 'classes': classes}, timeout=5)
            return res.json().get('detections', []) if res.status_code == 200 else []
        except:
            return []

    def select_best_grasp(self, target_name):
        """视觉决策链：YOLO 锁定区域 -> GraspNet 提取候选位姿 -> 投影匹配筛选最佳点"""
        print(f">>> 正在识别目标物体: {target_name}...")
        pcd = self.capture_scene_data()
        if pcd is None: return None

        if self.mock:
            class MockGrasp:
                rotation_matrix = np.eye(3)
                translation = np.array([0.0, 0.0, 0.4])
                score = 0.95
            return MockGrasp()

        # 1. 2D 目标检测过滤
        dets = self.detect_yolo_objects([target_name])
        if not dets: return None
        box = max(dets, key=lambda x: x[4]) # 取置信度最高的目标
        x1, y1, x2, y2 = map(int, box[:4])
        
        # 2. 3D 抓取位姿推理
        try:
            gg = self.grasp_net.run(pcd, vis=False)
            gg.nms()
            gg.sort_by_score()
        except:
            return None

        # 3. 2D-3D 空间一致性筛选
        K, _ = self.cam.get_intrinsics()
        for g in gg:
            pt = g.translation
            px = (K @ pt.reshape(3,1)).flatten()
            u, v = int(px[0]/px[2]), int(px[1]/px[2])
            
            # 检查抓取点投影是否落在 YOLO 检测框内
            if (x1-20 <= u <= x2+20) and (y1-20 <= v <= y2+20):
                print(f">>> 成功锁定最佳抓取位姿，置信度: {g.score:.2f}")
                return g
        return None

    def execute_grasp(self, grasp_in_cam):
        """执行物理抓取动作序列"""
        if grasp_in_cam is None: return

        # 1. 位姿计算与安全校验
        T_target_flange = self.transformer.calculate_target_pose(
            self.T_capture, grasp_in_cam, invert_hand_eye=INVERT_HAND_EYE
        )
        target_pose = self.transformer.matrix_to_rokae_pose(T_target_flange)

        if target_pose[2] < 0.03: # 安全底线：防止撞击桌面
            print("!!! 安全拦截：目标位置过低，操作已取消")
            return

        # 2. 执行标准化抓取流程 (预备-下探-抓取-抬起-放置)
        pre_pose = list(target_pose); pre_pose[2] += 0.12 

        if not self.mock:
            print(">>> 正在移动至预备点...")
            self.bot.move_l(pre_pose, speed=40)
            
            self.gripper.open_gripper()
            print(">>> 正在执行下探抓取...")
            self.bot.move_l(target_pose, speed=15)
            self.gripper.close_gripper(force=60)
            time.sleep(1.0)
            
            self.bot.move_l(pre_pose, speed=40)
            print(">>> 正在移动至放置点...")
            self.bot.move_l(DROP_POSE_XYZ, speed=50)
            self.gripper.open_gripper()
            time.sleep(0.5)
            self.bot.move_l(self.home_pose, speed=50)
        
        print(">>> 抓取任务圆满完成")

    def plan_from_llm(self, prompt):
        """大模型任务规划：将模糊的自然语言指令转化为逻辑代码"""
        if not self.llm: return ""
        tpl = """
        你是一个工业机器人运动规划专家。请根据用户指令调用以下 API：
        1. obj = get_grasp_candidates("物体名称")
        2. do_grasp(obj)
        要求：仅输出 Python 函数 plan() 的内部实现代码，不包含 Markdown 格式。
        用户指令: "{task}"
        """
        try:
            res = self.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": tpl.replace("{task}", prompt)}]
            )
            code = res.choices[0].message.content.replace("```python","").replace("```","").strip()
            return code.split("def plan():")[1].strip() if "def plan():" in code else code
        except:
            return ""

# ================= 交互式运行入口 =================
if __name__ == "__main__":
    system = RoboticVisionSystem()
    
    # 导出给 LLM 使用的 API 接口
    def get_grasp_candidates(name): return system.select_best_grasp(name)
    def do_grasp(obj): system.execute_grasp(obj)
    
    print("\n" + "="*40)
    print("      自主视觉抓取系统已准备就绪")
    print("="*40)
    print("提示：输入指令如 '帮我拿一下苹果'，输入 'q' 退出")
    
    while True:
        try:
            cmd = input("\n[用户指令] > ").strip()
            if cmd.lower() in ['q', 'exit']: break
            if not cmd: continue
            
            # 调用大语言模型进行逻辑生成
            plan_code = system.plan_from_llm(cmd)
            if plan_code:
                print(f"--- 规划路径 ---\n{plan_code}\n----------------")
                exec_scope = {'get_grasp_candidates': get_grasp_candidates, 'do_grasp': do_grasp, 'print': print}
                exec(plan_code, exec_scope)
            else:
                print(">>> 无法理解指令或规划失败")
                
        except KeyboardInterrupt: break
        except Exception as e: print(f"系统运行异常: {e}")
            
    if not system.mock:
        system.bot.disconnect()
        system.cam.stop()
