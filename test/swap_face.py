# @Time : 2023/6/25 15:49 
# @Author : CaoXiang
# @Description:
from PIL import Image, ImageDraw
import typing
import requests
import json
import time
import numpy as np
import cv2
from core.sam_predictor import RawSeger
import matplotlib.pyplot as plt
from util.painter import GridPainter
from api.inpaint_client import decode_frame_json, encode_frame_json

def draw_box(arr: np.ndarray, cords: typing.List[int], color: typing.Tuple[int, int, int],
             thickness: int) -> np.ndarray:
    """
        在原图上绘制出矩形框
    :param arr: 传入的原图ndarray
    :param cords: 框的坐标，按照【xmin,ymin,xmax,ymax】的方式进行组织
    :param color: 框的颜色
    :param thickness: 框线的宽度
    :return: 绘制好框后的图像仍然按照ndarray的数据格式s
    """
    assert len(cords) == 4, "cords must have 4 elements as xmin ymin xmax ymax."
    assert isinstance(arr, np.ndarray), "input must be type of numpy ndarray."
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    draw.rectangle(xy=cords, outline=color, width=thickness)
    img = np.array(img)
    return img

def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.axis('on')
    plt.show()

if __name__ == '__main__':
    seger = RawSeger()
    REST_API_URL = 'http://localhost:9900/sd/inpaint'
    painter = GridPainter()
    img_path = "/data/cx/ysp/aigc-smart-painter/assets/cloth1.jpg"
    image= Image.open(img_path)
    box = [220, 20, 500, 320]
    new_image = draw_box(np.array(image), cords=box, color=(255, 0, 0), thickness=2)
    show_image(new_image)
    # image = image.resize((512, 512))
    mask = seger.prompt_with_box(image, box=box, reverse=False)

    init_image = encode_frame_json(image)
    mask_image = encode_frame_json(mask)
    body = {"data": [
        {"request_id": "1", "prompt": "1 girl,head,realistic,single, simple background", "batch_size": 1,
         "num_inference_steps": 50, "guidance_scale": 7.5, "init_image": init_image, "mask_image": mask_image},
    ]}
    # print(json.dumps(body))
    start = time.time()
    result = requests.post(REST_API_URL, json=json.dumps(body)).json()["result"]
    end = time.time()
    print(f"batch api inference time: {end - start}s")
    for request_result in result:
        images = [Image.fromarray(decode_frame_json(img)) for img in request_result["images"]]
        painter.image_grid(images, rows=1, cols=len(images) // 1)
        painter.image_show()
    print("finished")


