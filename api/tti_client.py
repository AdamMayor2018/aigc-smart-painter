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


def decode_frame_json(data):
    data = base64.b64decode(data.encode())
    image = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
    return image



if __name__ == '__main__':
    REST_API_URL = 'http://10.5.101.152:9900/sd/tti'
    painter = GridPainter()
    body = {"data": [
        {"request_id": "1", "prompt": "Dutch East India Company stocks", "width": 512, "height": 512, "batch_size": 5,
         "num_inference_steps": 30, "guidance_scale": 7.5},
        {"request_id": "2", "prompt": "Overseas trade", "width": 512, "height": 512, "batch_size": 5,
         "num_inference_steps": 30, "guidance_scale": 7.5},
        {"request_id": "3", "prompt": "17th century", "width": 512, "height": 512, "batch_size": 5,
         "num_inference_steps": 30, "guidance_scale": 7.5},
        {"request_id": "4", "prompt": "Shareholder interests", "width": 512, "height": 512, "batch_size": 5,
         "num_inference_steps": 30, "guidance_scale": 7.5},
        {"request_id": "5", "prompt": "Stock issuance", "width": 512, "height": 512, "batch_size": 5,
         "num_inference_steps": 30, "guidance_scale": 7.5}
    ]}


    # body = {"data": [
    #     {"request_id": "1", "prompt": "Dutch East India Company stocks,Overseas trade,17th century,Shareholder interests,Stock issuance", "width": 512, "height": 512, "batch_size": 5,
    #      "num_inference_steps": 30, "guidance_scale": 7.5},
    # ]}
    print(json.dumps(body))
    start = time.time()
    result = requests.post(REST_API_URL, json=json.dumps(body)).json()["result"]
    end = time.time()
    print(f"batch api inference time: {end - start}s")
    total_images = []
    for request_result in result:
        images = [Image.fromarray(decode_frame_json(img)) for img in request_result["images"]]
        total_images.extend(images)
    painter.image_grid(total_images, rows=1, cols=len(total_images) // 1)
    painter.image_show()
    print("finished")
