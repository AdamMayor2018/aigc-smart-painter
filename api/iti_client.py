# @Time : 2023/6/17 15:09 
# @Author : CaoXiang
# @Description:
import requests
import json
import numpy as np
import cv2
import base64
from util.painter import GridPainter
from PIL import Image
import time
from io import BytesIO

def decode_frame_json(data):
    data = base64.b64decode(data.encode())
    image = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
    return image

def encode_frame_json(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
    frame = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100])[1]
    res = frame.tobytes()
    res = base64.b64encode(res).decode()
    return res

if __name__ == '__main__':
    REST_API_URL = 'http://localhost:9900/sd/iti'
    painter = GridPainter()
    path = "/data/cx/ysp/aigc-smart-painter/sketch-mountains-input.jpg"

    init_image = cv2.imread(path)
    init_image.transpose(2, 0, 1)
    init_image = cv2.resize(init_image, (768, 512))
    init_image = encode_frame_json(init_image)
    body = {"data": [
        {"request_id": "1", "prompt": "A fantasy landscape, trending on artstation", "batch_size": 2,
         "num_inference_steps": 30, "guidance_scale": 7.5, "init_image": init_image},
    ]}
    #print(json.dumps(body))
    start = time.time()
    result = requests.post(REST_API_URL, json=json.dumps(body)).json()["result"]
    end = time.time()
    print(f"batch api inference time: {end - start}s")
    for request_result in result:
        images = [Image.fromarray(decode_frame_json(img)) for img in request_result["images"]]
        painter.image_grid(images, rows=1, cols=len(images))
        painter.image_show()
    print("finished")
