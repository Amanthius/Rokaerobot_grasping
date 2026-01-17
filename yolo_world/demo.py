import requests
import base64
import json
from matplotlib import pyplot as plt
from PIL import Image

def draw_detections(image_path, detections):
    # 打开图片
    image = Image.open(image_path)
    plt.imshow(image)

    # 绘制检测结果
    for detection in detections:
        x1, y1, x2, y2, score, class_name = detection
        # 转换坐标为像素值
        print(x1, y1, x2, y2)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # 绘制边界框
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red'))
        # 添加标签
        plt.gca().text(x1, y1 - 5, f'{score:.2f}', color='red', fontsize=10, backgroundcolor='none')

    plt.show()

def send_detection_request(image_path, classes):
    # 读取图片并转换为 Base64 编码
    with open(image_path, 'rb') as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    # 准备请求数据
    data = {
        'image': image_base64,
        'classes': classes
    }

    # 发送 POST 请求
    response = requests.post('http://127.0.0.1:5000/detect', json=data)

    # 处理响应
    if response.status_code == 200:
        detections = response.json().get('detections', [])
        print('Detections:', detections)
        return detections
    else:
        print('Error:', response.text)
        return []

# 使用示例
if __name__ == '__main__':
    image_path = '1.jpg'  # 替换为你的图片路径
    classes = ['banana']  # 替换为你需要的类别
    detections = send_detection_request(image_path, classes)

    draw_detections(image_path, detections)