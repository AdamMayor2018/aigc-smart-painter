# @Time : 2023/6/27 22:39 
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
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel
)
from core.controller import ControlNetPreProcessor
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
controlnet = ControlNetModel.from_pretrained("/data/cx/ysp/aigc-smart-painter/models/sd-controlnet-openpose")
text2img = StableDiffusionPipeline.from_pretrained("/data/cx/ysp/aigc-smart-painter/models/chilloutmix_NiPrunedFp32Fix").to("cuda:2")
#inpaint = StableDiffusionInpaintPipeline(**text2img.components)
inpaint = StableDiffusionControlNetInpaintPipeline(**text2img.components, controlnet=controlnet)
inpaint.to("cuda:2")
seger = RawSeger()
REST_API_URL = 'http://localhost:9900/sd/inpaint'
painter = GridPainter()
img_path = "/data/cx/ysp/aigc-smart-painter/assets/cloth1.jpg"
image = Image.open(img_path)
box = [240, 20, 500, 290]
new_image = draw_box(np.array(image), cords=box, color=(255, 0, 0), thickness=2)
show_image(new_image)
mask = seger.prompt_with_box(image, box=box, reverse=False)
mask = Image.fromarray(mask)
show_image(mask)
end = time.time()

# controlnet
cnet_preprocesser = ControlNetPreProcessor(aux_model_path="/data/cx/ysp/aigc-smart-painter/models/control-net-aux-models/Annotators")
pose_image = cnet_preprocesser.aux_infer(image, "openpose")
show_image(pose_image)

prompt = "1 girl,fashion model,standing, wearing a shirt"
images = inpaint(prompt=prompt, image=image, mask_image=mask, num_images_per_prompt=8,
         num_inference_steps=50, guidance_scale=7.5, control_image=pose_image).images

painter.image_grid(images, rows=2, cols=len(images) // 2)
painter.image_show()
print("finished")