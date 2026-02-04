# **ROKAE 机械臂 AI 视觉抓取系统 (ROKAE AI Visual Grasping System)**

基于 **ROKAE (珞石) 机械臂**、**Intel RealSense D435** 相机与 **大寰 (DH-Robotics) PGIA-140** 夹爪构建的端到端 AI 抓取部署项目。

系统深度集成了 **LLM 任务规划**、**YOLO-World 开放词汇检测** 与 **GraspNet 6-DoF 姿态生成**。项目核心亮点在于实现了严谨的眼在手（Eye-in-Hand）坐标变换链条，能够将视觉生成的抓取位姿精准转化为机械臂的执行动作。

<img src="https://github.com/user-attachments/assets/f7e0e598-5314-429f-ae69-e1ca6207a6fe" width="400" />

## **🌟 核心特性**

* **LLM 语义驱动**: 集成 DeepSeek 接口，支持通过自然语言执行复杂的逻辑操作（如“帮我清理桌子上的橙子”）。  
* **开放词汇目标检测**: 采用 YOLOv8-world-v2，无需针对特定物体重新训练，通过 Flask 封装为轻量级 HTTP 接口。  
* **高性能抓取生成**: 基于上海交大 MVIG 实验室的 GraspNet-Baseline，在非结构化环境下实现鲁棒的 6-DoF 抓取预测。  
* **工程级坐标变换**: 包含轴向自动映射（Grasp X → Robot Z）、TCP 长度补偿及垂直下探约束。

## **📂 项目结构**

├── libs/  
│   ├── knn/              \# GraspNet 依赖的 KNN 算子  
│   └── pointnet2/        \# GraspNet 依赖的 PointNet2 算子  
├── yolo\_world/  
│   ├── yolo\_world.py     \# 基于 Flask 的目标检测服务端  
│   └── demo.py           \# 检测服务调用示例  
├── drivers/  
│   ├── robot\_arm\_lib.py  \# 珞石机器人与夹爪统一驱动库  
│   └── camera\_realsense.py \# 相机接入逻辑  
├── graspnet/             \# 抓取位姿生成算法模块  
├── config/  
│   ├── hand\_eye\_result.txt \# 手眼标定 4x4 矩阵  
│   └── keys.txt          \# LLM API 密钥存储  
├── robot\_grasping.py     \# AI 抓取系统主程序  
└── setup\_path.py         \# 环境库路径自动配置脚本

## **🛠️ 环境安装与编译**

### **1\. 基础依赖**

建议使用 **Python 3.10**。

可以直接使用下面命令：

pip install -r requirements.txt（推荐）

或者使用conda:

conda env create -f environment.yml

或者直接安装

pip install numpy opencv-python open3d scipy openai pymodbus requests flask ultralytics

### **2\. 编译 GraspNet 关键算子 (必选)**

GraspNet 需要编译 C++/CUDA 算子。请进入相应目录执行安装：

\# 安装 KNN 模块  
cd libs/knn  
python setup.py install

\# 安装 PointNet2 模块  
cd ../pointnet2  
python setup.py install

### **3\. 环境导出 (备份)**

如需导出当前 Python 环境配置：

pip freeze \> requirements.txt

## **🚀 使用方法**

### **第一步：启动视觉检测服务**

开放词汇目标检测作为独立的服务运行。客户端将图片进行 Base64 编码并指定类别文本传给服务端。

cd yolo\_world  
python yolo\_world.py

测试检测服务:  
运行 python yolo\_world/demo.py，确认 classes \= \['cup'\] 等类别是否能被正确框选。

<img width="180" height="200" alt="image" src="https://github.com/user-attachments/assets/eb520500-73cb-48f8-b391-ea71ba7f9076" />
<img width="180" height="200" alt="image" src="https://github.com/user-attachments/assets/866d9514-1159-4ab0-9479-942ec5adfb54" />


### **第二步：配置 LLM 密钥**

在项目根目录下创建 config/keys.txt，并将你的 **DeepSeek API Key** 写入该文件。

* **接口地址**: https://api.deepseek.com  
* **模型版本**: deepseek-coder

### **第三步：启动主程序**

确保机械臂 IP (192.168.0.160) 与夹爪串口 (/dev/ttyUSB0) 连接正常。

python robot\_grasping.py

程序启动后，在控制台直接输入中文指令即可开始抓取任务。

## **⚙️ 硬件注意事项**

1. **夹爪权限**: 若提示串口无法打开，请执行 sudo chmod 666 /dev/ttyUSB0。  
2. **TCP 偏移**: 在主程序中确保 TCP\_Z\_OFFSET 匹配你的实际夹爪长度（法兰中心到指尖）。  
3. **坐标变换**: 主程序已包含轴向对齐逻辑，将 GraspNet 默认的 X-Approach 映射为机械臂的 Z-Approach。

## **💬 交流与支持**

**遇到问题？** 如果你在部署过程中遇到任何疑问或 Bug，欢迎通过 Issues 提出，我会尽快回复。

**觉得有用？** 如果这个项目对你的研究或工程有所帮助，能否请你点亮右上角的 ⭐ Star？你的支持是我持续优化的动力！

## **⚖️ 致谢与版权声明 (License)**

### **GraspNet-Baseline**

本项目中 graspnet/ 模块及 libs/ 中的编译代码源自以下开源项目：

* **项目名称**: [GraspNet-1Billion](https://github.com/graspnet/graspnet-baseline)  
* **相关论文**: *"GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping" (CVPR 2020\)*  
* **版权所有**: 
* **许可协议条款**:All data, labels, code and models belong to the graspnet team, MVIG, SJTU and are freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an email at fhaoshu at gmail\_dot\_com and cc lucewu at sjtu.edu.cn .

### **其他**

* **YOLO-World**: 遵循 Ultralytics 开源许可。  
* **ROKAE SDK**: 版权归珞石(北京)科技有限公司所有。
