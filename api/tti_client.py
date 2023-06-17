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

REST_API_URL = 'http://localhost:9900/sd/tti'
painter = GridPainter()
def decode_frame_json(data):
    data = base64.b64decode(data.encode())
    image = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    #image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
    return image

body = {"data": [
    {"request_id": "1", "prompt":"a super nice car", "width":512, "height":512, "batch_size":2, "num_inference_steps":30, "guidance_scale":7.5},
    {"request_id": "2", "prompt":"a cat with cute face", "width":512, "height":512, "batch_size":2, "num_inference_steps":25, "guidance_scale":7.5},
]}
print(json.dumps(body))

result = requests.post(REST_API_URL, json=json.dumps(body)).json()["result"]
for request_result in result:
    images = [Image.fromarray(decode_frame_json(img)) for img in request_result["images"]]
    painter.image_grid(images, rows=1, cols= len(images))
    painter.image_show()