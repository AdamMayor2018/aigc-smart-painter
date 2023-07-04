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
from config.conf_loader import YamlConfigLoader
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel
)
from core.controller import ControlNetPreProcessor
from model_plugin.face_detector import Yolov5FaceDetector

def expand_box(box, y_ratio = 0.6, x_ratio = 0.25):
    xmin, ymin, xmax, ymax = box
    width = xmax - xmin
    height = ymax - ymin
    xmin = xmin - x_ratio * width
    xmax = xmax + x_ratio * width
    ymin = ymin - y_ratio * height
    ymin = ymin if ymin >0 else 0
    ymax = ymax + 0.05 * height
    return [int(x) for x in [xmin, ymin, xmax, ymax]]



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
config_loader = YamlConfigLoader(yaml_path="/data/cx/ysp/aigc-smart-painter/config/general_config.yaml")
detector = Yolov5FaceDetector(config_loader)
painter = GridPainter()
img_path = "/data/cx/ysp/aigc-smart-painter/assets2/cloth1.jpg"
image = Image.open(img_path)
plt.imsave("/data/cx/ysp/aigc-smart-painter/assets2/orgin1.jpg", np.array(image))
box = detector.detect(np.array(image))[: 4]
box = expand_box(box)
new_image = draw_box(np.array(image), cords=box, color=(255, 0, 0), thickness=2)
plt.imsave("/data/cx/ysp/aigc-smart-painter/assets2/bbox.jpg", new_image)
show_image(new_image)
mask = seger.prompt_with_box(image, box=box, reverse=False)
plt.imsave("/data/cx/ysp/aigc-smart-painter/assets2/mask.jpg", mask)
show_image(mask)
end = time.time()

# controlnet
cnet_preprocesser = ControlNetPreProcessor(aux_model_path="/data/cx/ysp/aigc-smart-painter/models/control-net-aux-models/Annotators")
pose_image = cnet_preprocesser.aux_infer(np.array(image), "openpose")
show_image(pose_image)
plt.imsave("/data/cx/ysp/aigc-smart-painter/assets2/pose.jpg", np.array(pose_image))

prompt = "symmetry realistic,real life,photography,{masterpiece},{best quality},8K,HDR,highres,1 girl,fashion model,looking at viewer, wearing a shirt, simple background"
images = inpaint(prompt=prompt, image=image, mask_image=Image.fromarray(mask), num_images_per_prompt=8,
         num_inference_steps=50, guidance_scale=7.5, control_image=pose_image).images

painter.image_grid(images, rows=2, cols=len(images) // 2)
painter.image_show()
plt.imsave("/data/cx/ysp/aigc-smart-painter/assets2/result.jpg", np.array(painter.grid))
print("finished")