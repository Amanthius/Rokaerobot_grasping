from flask import Flask, request, jsonify
from ultralytics import YOLOWorld
from PIL import Image
import io
import base64

app = Flask(__name__)




model = YOLOWorld("yolov8l-worldv2.pt")
# Define custom classes



@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    image_data = data.get('image')
    classes = data.get('classes')
    print('classes:', classes)
    model.set_classes(classes)
    if not image_data or not classes:
        return jsonify({'error': 'Missing image or classes'})

    # 解码 Base64 图片数据
    image_data = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_data))

    # 使用模型进行预测
    results = model.predict(source=image, conf=0.05)

    # 提取检测结果
    detections = results[0].boxes.data.tolist()

    print(detections)
    # 返回结果
    return jsonify({'detections': detections})

if __name__ == '__main__':
    app.run(host='127.0.0.1')