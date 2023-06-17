# @Time : 2023/6/16 10:22 
# @Author : CaoXiang
# @Description: 文生图http接口
import typing

import numpy as np
from flask import Flask, request, jsonify
from core.tti_predictor import StableDiffusionPredictor
from config.conf_loader import YamlConfigLoader
from core.prompt_loader import PromptManager
from collections import defaultdict
import argparse
import base64
import json
import cv2
from PIL import Image

app = Flask(__name__)

def encode_frame_json(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
    frame = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100])[1]
    res = frame.tobytes()
    res = base64.b64encode(res).decode()
    return res

def wrap_json(collect_images):
    result_data = {"result": []}
    result = result_data["result"]
    for request_result in collect_images:
        images = request_result["images"]
        request_result["images"] = [encode_frame_json(img) for img in images]
        result.append(request_result)
    return result_data




@app.post("/sd/tti")
def tti_infer():
    if request.method == "POST":
        data = request.get_json()
        if not data:
            return jsonify({"error info": "backend recieved no json request."}), 400
        try:
            res = json.loads(request.get_json())
        except Exception as e:
            print(e)
            return jsonify({"error info": "backend recieved invalid json data."}), 400
        if res["data"]:
            collect_images = []
            for prompt_request in res["data"]:
                prompt = prompt_request.get("prompt")
                batch_size = prompt_request.get("batch_size", 1)
                width = prompt_request.get("width", 512)
                height = prompt_request.get("height", 512)
                extra_params = {k:v for k,v in prompt_request.items() if k not in ['request_id', 'prompt', 'batch_size', 'height', 'width']}
                try:
                    images = sdp.tti_inference(prompt=[pt.generate_pos_prompt(prompt)] * batch_size,
                                               negative_prompt=[pt.generate_neg_prompt(prompt)] * batch_size,
                                               width=width, height=height, **extra_params).images
                except Exception as e:
                    return jsonify({"error info": f"bad params received: {e}"}), 500
                assert len(images) == batch_size, "produced images number unequal to request batch size."
                prompt_request["images"] = images
                collect_images.append(prompt_request)
            json_result = wrap_json(collect_images)
            return jsonify(json_result), 200
        else:
            return jsonify({"error info": "backend recieved invalid json data.missing key 'data'"}), 400



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # project level params
    parser.add_argument('--conf', type=str, default='/data/cx/ysp/aigc-smart-painter/config/general_config.yaml',
                        help='config yaml file path')
    parser.add_argument('--port', type=str, default=9900,
                        help='web server port')
    parser.add_argument('--ip', type=str, default="localhost",
                        help='web server ip binding')
    opt = vars(parser.parse_args())
    yaml_path = opt["conf"]
    port = opt["port"]
    ip = opt["ip"]
    conf_loader = YamlConfigLoader(yaml_path)
    sdp = StableDiffusionPredictor(config_loader=conf_loader)
    pt = PromptManager()
    app.run(port=port, debug=False, host=ip)
