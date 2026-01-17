# -*- coding: utf-8 -*-
"""
ROKAE 机械臂 + 大寰 PGIA140 夹爪 统一驱动库
"""

import time
import sys
import signal
import math
import copy
import pymodbus
from pymodbus.client import ModbusSerialClient

# SDK 导入
try:
    import setup_path 
    from Release.linux import xCoreSDK_python
except ImportError:
    xCoreSDK_python = None

class PGIA140:
    """大寰(DH-Robotics) PGIA-140 夹爪控制类 (自适应版)"""
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200, slave_id=1):
        self.port = port
        self.slave_id = slave_id
        # 自动判断参数策略 ('slave' vs 'unit' vs 'none')
        self.id_strategy = 'slave' if pymodbus.__version__.startswith('3') else 'unit'
        self.strategy_fixed = False
        
        client_params = {'port': port, 'baudrate': baudrate, 'bytesize': 8, 'parity': 'N', 'stopbits': 1, 'timeout': 1}
        if not pymodbus.__version__.startswith('3'):
            client_params['method'] = 'rtu'
            
        self.client = ModbusSerialClient(**client_params)
        self.is_connected = False

    def connect(self):
        try:
            self.is_connected = self.client.connect()
            return self.is_connected
        except Exception as e:
            print(f"[夹爪] 连接异常: {e}")
            return False

    def disconnect(self):
        self.client.close()
        self.is_connected = False

    def _retry_api_call(self, func, *args, **kwargs):
        """智能兼容重试逻辑"""
        if self.strategy_fixed and self.id_strategy == 'none':
            return func(*args, **kwargs)

        api_kwargs = kwargs.copy()
        if self.id_strategy != 'none':
            api_kwargs[self.id_strategy] = self.slave_id
        
        try:
            return func(*args, **api_kwargs)
        except TypeError as e:
            if not self.strategy_fixed and "unexpected keyword argument" in str(e):
                if self.id_strategy == 'slave': self.id_strategy = 'unit'
                elif self.id_strategy == 'unit': self.id_strategy = 'none'
                print(f"[夹爪] 兼容性调整 -> 策略切换为: {self.id_strategy}")
                return self._retry_api_call(func, *args, **kwargs)
            if not self.strategy_fixed:
                 self.id_strategy = 'none'
                 self.strategy_fixed = True
                 return func(*args, **kwargs)
            raise e

    def initialize(self):
        if not self.is_connected: return False
        print("[夹爪] 正在初始化...")
        try:
            self._retry_api_call(self.client.write_register, 0x0100, 1)
            self.strategy_fixed = True
            time.sleep(3) 
            print("[夹爪] 初始化完成")
            return True
        except Exception as e:
            print(f"[夹爪] 初始化失败: {e}")
            return False

    def set_motion(self, position, force=50, speed=50):
        if not self.is_connected: return False
        position = max(0, min(1000, int(position)))
        try:
            print(f"[夹爪] 动作 -> 位置:{position}, 力量:{force}%")
            self._retry_api_call(self.client.write_registers, 0x0101, [force, speed, position])
            self.strategy_fixed = True
            return True
        except Exception as e:
            print(f"[夹爪] 运动指令失败: {e}")
            return False

    def open_gripper(self, speed=50):
        return self.set_motion(1000, 100, speed)

    def close_gripper(self, speed=50, force=40):
        return self.set_motion(0, force, speed)

class RobotController:
    """ROKAE 机械臂控制类"""
    def __init__(self, ip):
        self.ec = {}
        self.is_running = True
        if xCoreSDK_python is None:
            raise ImportError("无法导入 xCoreSDK，请检查环境路径。")
        self.robot = xCoreSDK_python.xMateRobot(ip)
        signal.signal(signal.SIGINT, self._emergency_stop)

    def connect(self):
        print(f"[机械臂] 正在连接...")
        self.robot.connectToRobot(self.ec)
        if self.ec.get('code', 0) != 0:
            print(f"!!! 连接失败: {self.ec}")
            sys.exit(1)
        self.robot.setOperateMode(xCoreSDK_python.OperateMode.automatic, self.ec)
        self.robot.setPowerState(True, self.ec)
        self.robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, self.ec)
        print("[机械臂] 已就绪")

    def disconnect(self):
        print("[机械臂] 断开连接")
        self.robot.disconnectFromRobot(self.ec)

    def _emergency_stop(self, sig, frame):
        print("\n!!! 紧急停止 !!!")
        try:
            self.robot.stop(self.ec)
        except: pass
        self.is_running = False
        sys.exit(0)

    def _wait_for_idle(self):
        while self.is_running:
            state = self.robot.operationState(self.ec)
            if state == xCoreSDK_python.OperationState.idle:
                break
            if self.ec.get('code', 0) != 0:
                print(f"!!! 运动报错: {self.ec}")
                self.is_running = False
                break
            time.sleep(0.05)

    def get_current_pose(self):
        p = self.robot.cartPosture(xCoreSDK_python.CoordinateType.endInRef, self.ec)
        return list(p.trans) + list(p.rpy)
    
    def move_j(self, joint_list, speed=20):
        if not self.is_running: return
        cmd = xCoreSDK_python.MoveAbsJCommand(xCoreSDK_python.JointPosition(joint_list), speed, 50)
        self.robot.moveAppend([cmd], xCoreSDK_python.PyString(), self.ec)
        self.robot.moveStart(self.ec)
        self._wait_for_idle()

    def move_l(self, pose_list, speed=50):
        if not self.is_running: return
        cmd = xCoreSDK_python.MoveLCommand(xCoreSDK_python.CartesianPosition(pose_list), speed, 50)
        self.robot.moveAppend([cmd], xCoreSDK_python.PyString(), self.ec)
        self.robot.moveStart(self.ec)
        self._wait_for_idle()

    def move_l_relative(self, dx=0, dy=0, dz=0, speed=30):
        current_pose = self.get_current_pose()
        target_pose = copy.deepcopy(current_pose)
        target_pose[0] += dx
        target_pose[1] += dy
        target_pose[2] += dz
        self.move_l(target_pose, speed)