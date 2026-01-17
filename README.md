# **ROKAE AI 视觉抓取系统 (ROKAE AI Visual Grasping System)**

基于 ROKAE 机械臂、Intel RealSense D435 相机与大寰 PGIA-140 夹爪构建的端到端视觉抓取部署项目。本项目通过 LLM 任务规划、YOLO 目标检测与 GraspNet 姿态估计，实现了复杂的“眼在手” (Eye-in-Hand) 自动化抓取闭环。

## **🚀 项目特性**

* **多模态任务规划**: 集成 LLM (如 DeepSeek)，将自然语言指令转化为机械臂运动逻辑。  
* **高精度感知链**: 结合 YOLO-World 进行开放词汇检测与 GraspNet 进行 6DOF 抓取位姿生成。  
* **严谨的坐标转换**: 专门针对 Eye-in-Hand 模式修正的矩阵变换逻辑，解决了轴向对齐与 TCP 长度补偿等核心工程痛点。  
* **工业级驱动封装**: 封装了 ROKAE xCoreSDK 与 Modbus RTU 夹爪控制，支持多种运动控制模式（MoveJ, MoveL, RelativeMove）。

## **🛠 硬件清单**

* **机械臂**: ROKAE xMate Robot (控制频率支持非实时模式)  
* **相机**: Intel RealSense D435/D435i  
* **末端执行器**: 大寰 (DH-Robotics) PGIA-140 工业型二指夹爪  
* **主机**: 建议搭载 NVIDIA GPU (用于检测与位姿生成)

## **📂 项目结构**

├── Release/              \# ROKAE 官方 SDK 库文件 (Windows/Linux)  
├── drivers/  
│   ├── robot\_arm\_lib.py  \# 机械臂与夹爪核心驱动  
│   └── camera\_realsense.py \# Realsense 相机接入逻辑  
├── graspnet/             \# 抓取检测算法核心 (详见下文 Attribution)  
├── yolo\_world/           \# 目标检测模块  
├── config/  
│   ├── hand\_eye\_result.txt \# 手眼标定矩阵 (4x4)  
│   └── keys.txt          \# LLM API 密钥  
├── setup\_path.py         \# 环境变量与库路径自动配置脚本  
├── calibration.py        \# 自动化手眼标定采集程序  
└── robot\_grasping.py     \# AI 抓取系统主程序 (V2 修正版)

## **⚙️ 安装与配置**

### **1\. 环境准备**

确保系统中已安装 pymodbus, open3d, numpy, opencv-python, scipy 及 openai。

### **2\. 硬件权限 (Linux)**

夹爪串口通常需要读写权限，请在终端执行：

sudo chmod 666 /dev/ttyUSB0

### **3\. SDK 路径**

setup\_path.py 会自动将 Release/ 下的动态库添加到 Python 路径中，请确保该脚本位于根目录。

## **📍 核心流程**

1. **手眼标定**: 运行 calibration.py 获取 hand\_eye\_result.txt。程序使用 IPPE 算法优化平面棋盘格位姿，确保相机到法兰中心的变换矩阵精确。  
2. **坐标转换逻辑**:  
   * 系统利用矩阵链乘法：$T\_{Base \\to Obj} \= T\_{Base \\to Flange(Capture)} \\times T\_{Flange \\to Cam} \\times T\_{Cam \\to Obj}$。  
   * 包含 **GraspNet 轴向修正**: 将算法定义的 X-Approach 转换为机械臂定义的 Z-Approach。  
   * 包含 **TCP 长度补偿**: 自动减去夹爪物理长度，防止碰撞物体。  
3. **AI 执行**: 主程序 robot\_grasping.py 监听用户语音/文本，由 LLM 生成逻辑代码并调用视觉反馈完成抓取。

## **⚖️ 开源协议与归属 (Attribution & License)**

### **GraspNet-Baseline**

本项目中的抓取位姿检测代码参考/使用了以下开源项目：

* **项目名称**: [GraspNet-1Billion](https://github.com/graspnet/graspnet-baseline)  
* **相关论文**: *"GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping" (CVPR 2020\)*  
* **归属**: **graspnet team, MVIG, SJTU**   
* **License 条款**:All data, labels, code and models belong to the graspnet team, MVIG, SJTU and are freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an email at fhaoshu at gmail\_dot\_com and cc lucewu at sjtu.edu.cn .

### **其他说明**

* 机械臂 SDK 与夹爪控制逻辑属于本项目原始开发，遵循基础开源许可。  
* 禁止利用本项目进行任何违反实验室安全准则的机械臂操作。

## **📧 联系与支持**

如有算法部署或硬件适配问题，请联系 amanthius@163.com。
