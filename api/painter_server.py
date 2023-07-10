# @Time : 2023/6/16 10:22 
# @Author : CaoXiang
# @Description: 文生图http接口
import numpy as np
from flask import Flask, request, jsonify
from core.sd_predictor import StableDiffusionPredictor
from config.conf_loader import YamlConfigLoader
from core.prompt_loader import PromptManager
from core.controller import OpenPosePreProcessor
from PIL import Image
import argparse
import base64
import json
import cv2

app = Flask(__name__)


def encode_frame_json(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
    frame = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100])[1]
    res = frame.tobytes()
    res = base64.b64encode(res).decode()
    return res


def decode_frame_json(data):
    data = base64.b64decode(data.encode())
    image = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
    return image


def wrap_json(collect_images):
    result_data = {"result": []}
    result = result_data["result"]
    for request_result in collect_images:
        images = request_result["images"]
        request_result["images"] = [encode_frame_json(img) for img in images]
        result.append(request_result)
    return result_data


def resize_to_512(width, height):
    if width > height:
        infer_height = int(height * 512 / width)
        infer_width = 512
    else:
        infer_width = int(width * 512 / height)
        infer_height = 512
    return infer_width, infer_height


@app.post("/sd/tti")
def text2img_infer():
    if request.method == "POST":
        try:
            data = request.get_json()
            if isinstance(data, str):
                res = json.loads(data)
            elif isinstance(data, dict):
                res = data
            else:
                raise Exception("request data format error")
        except Exception as e:
            return jsonify({"error info": str(e)}), 400
        if res["data"]:
            collect_images = []
            for prompt_request in res["data"]:
                prompt = prompt_request.get("prompt")
                batch_size = prompt_request.get("batch_size", 1)
                width = prompt_request.get("width", 512)
                height = prompt_request.get("height", 512)
                if batch_size > 16:
                    return jsonify({"error info": "retrive batch_size is limited to 16 for cuda OOM issues."}), 400
                # 宽高中的长边缩放到512，短边进行等比缩放
                if smart_mode:
                    infer_width, infer_height = resize_to_512(width, height)
                else:
                    infer_height = height
                    infer_width = width

                extra_params = {k: v for k, v in prompt_request.items() if
                                k not in ['request_id', 'prompt', 'batch_size', 'height', 'width']}
                try:
                    images = sdp.tti_inference(prompt=prompt,
                                               negative_prompt=pm.generate_neg_prompt(prompt),
                                               num_images_per_prompt=batch_size,
                                               width=infer_width, height=infer_height, **extra_params).images
                except Exception as e:
                    return jsonify({"error info": f"bad params received: {e}"}), 500
                assert len(images) == batch_size, "produced images number unequal to request batch size."
                images = [cv2.resize(np.array(img), (width, height)) for img in images]

                prompt_request["images"] = images
                collect_images.append(prompt_request)
            json_result = wrap_json(collect_images)
            return jsonify(json_result), 200
        else:
            return jsonify({"error info": "backend recieved invalid json data.missing key 'data'"}), 400


@app.post("/sd/iti")
def img2img_infer():
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
                strength = prompt_request.get("strength", 0.75)
                batch_size = prompt_request.get("batch_size", 1)
                init_image = prompt_request.get("init_image")
                init_image = Image.fromarray(decode_frame_json(init_image))
                width = init_image.width
                height = init_image.height
                if not init_image:
                    return jsonify({"error info": "backend recieved invalid json data.missing key 'init_image'"}), 400
                if batch_size > 16:
                    return jsonify({"error info": "retrive batch_size is limited to 16 for cuda OOM issues."}), 400
                #宽高中的长边缩放到512，短边进行等比缩放
                if smart_mode:
                    infer_width, infer_height = resize_to_512(width, height)
                else:
                    infer_height = height
                    infer_width = width
                init_image = init_image.resize((infer_width, infer_height))

                extra_params = {k: v for k, v in prompt_request.items() if
                                k not in ['request_id', 'prompt', 'batch_size', 'height', 'width', 'init_image']}
                try:
                    images = sdp.iti_inference(prompt=prompt,
                                               negative_prompt=pm.generate_neg_prompt(prompt),
                                               init_image=init_image,
                                               strength=strength,
                                               num_images_per_prompt=batch_size, **extra_params).images
                except Exception as e:
                    return jsonify({"error info": f"bad params received: {e}"}), 500
                assert len(images) == batch_size, "produced images number unequal to request batch size."
                images = [cv2.resize(np.array(img), (width, height)) for img in images]
                images = [np.array(img) for img in images]
                prompt_request["images"] = images
                collect_images.append(prompt_request)
            json_result = wrap_json(collect_images)
            return jsonify(json_result), 200
        else:
            return jsonify({"error info": "backend recieved invalid json data.missing key 'data'"}), 400


@app.post("/sd/inpaint")
def inpaint_infer():
    if request.method == "POST":
        data = request.get_json()
        print(f"data: {data}")
        if not data:
            return jsonify({"error info": "backend recieved no json request."}), 400
        try:
            print(request.get_json())
            res = json.loads(request.get_json())
        except Exception as e:
            print(e)
            return jsonify({"error info": "backend recieved invalid json data."}), 400
        if res["data"]:
            collect_images = []
            for prompt_request in res["data"]:
                prompt = prompt_request.get("prompt")
                batch_size = prompt_request.get("batch_size", 1)
                init_image = prompt_request.get("init_image")
                mask_image = prompt_request.get("mask_image")
                init_image = decode_frame_json(init_image)
                mask_image = decode_frame_json(mask_image)
                # width = init_image.width
                # height = init_image.height
                height, width, _ = init_image.shape
                if batch_size > 16:
                    return jsonify({"error info": "retrive batch_size is limited to 16 for cuda OOM issues."}), 400
                #宽高中的长边缩放到512，短边进行等比缩放
                if smart_mode:
                    infer_width, infer_height = resize_to_512(width, height)
                else:
                    infer_height = height
                    infer_width = width
                init_image = cv2.resize(init_image, (infer_width, infer_height))
                mask_image = cv2.resize(mask_image, (infer_width, infer_height))
                print(infer_width, infer_height)
                init_image = Image.fromarray(init_image)
                mask_image = Image.fromarray(mask_image)
                # init_image = init_image.resize((infer_width, infer_height))
                # mask_image = mask_image.resize((infer_width, infer_height))

                extra_params = {k: v for k, v in prompt_request.items() if
                                k not in ['request_id', 'prompt', 'batch_size', 'height', 'width', 'init_image', 'mask_image']}
                try:
                    images = sdp.inpaint_inference(prompt=pm.generate_pos_prompt(prompt),
                                                   negative_prompt=pm.generate_neg_prompt(prompt),
                                                   init_image=init_image,
                                                   mask_image=mask_image,
                                                   height=infer_height,
                                                   width=infer_width,
                                                   num_images_per_prompt=batch_size, **extra_params).images
                except Exception as e:
                    return jsonify({"error info": f"bad params received: {e}"}), 500
                assert len(images) == batch_size, "produced images number unequal to request batch size."
                images = [cv2.resize(np.array(img), (width, height)) for img in images]
                images = [np.array(img) for img in images]
                prompt_request["images"] = images
                collect_images.append(prompt_request)
            json_result = wrap_json(collect_images)
            return jsonify(json_result), 200
        else:
            return jsonify({"error info": "backend recieved invalid json data.missing key 'data'"}), 400


@app.post("/sd/controlnet")
def controlnet_infer():
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
                init_image = prompt_request.get("init_image")
                mode = prompt_request.get("mode", "canny")
                init_image = Image.fromarray(decode_frame_json(init_image))
                init_image = cnet_preprocesser.aux_infer(init_image, mode)

                width = init_image.width
                height = init_image.height
                if not init_image:
                    return jsonify({"error info": "backend recieved invalid json data.missing key 'init_image'"}), 400
                if batch_size > 16:
                    return jsonify({"error info": "retrive batch_size is limited to 16 for cuda OOM issues."}), 400
                #宽高中的长边缩放到512，短边进行等比缩放
                if smart_mode:

                    infer_width, infer_height = resize_to_512(width, height)
                else:
                    infer_height = height
                    infer_width = width
                init_image = init_image.resize((infer_width, infer_height))

                extra_params = {k: v for k, v in prompt_request.items() if
                                k not in ['request_id', 'prompt', 'batch_size', 'height', 'width', 'init_image', 'mode']}
                try:
                    images = sdp.controlnet_inference(prompt=prompt,
                                                   negative_prompt=pm.generate_neg_prompt(prompt),
                                                   init_image=init_image,
                                                   mode=mode,
                                                   num_images_per_prompt=batch_size, **extra_params).images
                except Exception as e:
                    return jsonify({"error info": f"error while inferece error printing: {e}"}), 500
                assert len(images) == batch_size, "produced images number unequal to request batch size."
                images = [cv2.resize(np.array(img), (width, height)) for img in images]
                images = [np.array(img) for img in images]
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
    parser.add_argument('--ip', type=str, default="10.5.101.152",
                        help='web server ip binding')
    opt = vars(parser.parse_args())
    yaml_path = opt["conf"]
    port = opt["port"]
    ip = opt["ip"]
    conf_loader = YamlConfigLoader(yaml_path)
    smart_mode = conf_loader.attempt_load_param("smart_mode")
    sdp = StableDiffusionPredictor(config_loader=conf_loader)
    pm = PromptManager()
    cnet_preprocesser = OpenPosePreProcessor(aux_model_path=conf_loader.attempt_load_param("aux_model_path"))
    app.run(port=port, debug=False, host=ip)
