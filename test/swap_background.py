# @Time : 2023/6/27 22:39
# @Author : CaoXiang
# @Description: 测试AI模特换装
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
from model_plugin.diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel
)
from model_plugin.face_detector import Yolov5FaceDetector
from core.prompt_loader import PromptManager
from core.sd_predictor import StableDiffusionPredictor

from service.magic_closet import MagicCloset

#((1 girl)), (goddess-like happiness:1.2), Kpop idol, smile, (RAW photo:1.2), (photorealistic:1.4), (masterpiece:1.3), (intricate details:1.2), delicate, beautiful detailed, (detailed eyes), (detailed facial features), (20 years old female with fashion clothes), tall female, petite, small breasts, narrow waist, (facing camera), (looking_at_viewer:1.3), from_front, big circle earrings, slim_legs, (best quality:1.4), (ultra highres:1.2), cinema light, outdoors, (extreme detailed illustration), (lipgloss, eyelashes, best quality, ultra highres, depth of field, caustics, Broad lighting, natural shading, 85mm, f/1.4, ISO 200, 1/160s:0.75)

# 读取图片
img_path = "/data/cx/ysp/aigc-smart-painter/assets2/cloth1.jpg"
image = Image.open(img_path)

conf_loader = YamlConfigLoader(yaml_path="/data/cx/ysp/aigc-smart-painter/config/general_config.yaml")
# 初始化服务
cloth_service = MagicCloset(conf_loader)
#prompt = "indoors, (European style decoration style:1.2), Ivory long table, light green wallpaper, lights"
prompt = "outdoors background,seaside, beach,sand, sunset,"
input_data = np.array([162, 10, 630, 1000])
mask_image, images = cloth_service.swap_background(image, seg_method="sam", sam_method="box", input_data=input_data, input_label=None, prompt=prompt, num_images_per_prompt=2, num_inference_steps=30, strength=0.95, reverse=True, guidance_scale=14)
#mask_image, images = cloth_service.swap_background(image, sam_method="segformer", prompt=prompt, num_images_per_prompt=2, num_inference_steps=30, strength=0.95, reverse=False, guidance_scale=7.5)
#mask_image,control_images, images = cloth_service.swap_background(image, prompt, num_images_per_prompt=8,reverse=True)
plt.imsave(f"/data/cx/ysp/aigc-smart-painter/assets2/mask.jpg", np.array(mask_image))
# for i, image in enumerate(control_images):
#     plt.imsave(f"/data/cx/ysp/aigc-smart-painter/assets2/control_image_{i}.jpg", np.array(image))

for i, image in enumerate(images):
    plt.imsave(f"/data/cx/ysp/aigc-smart-painter/assets2/result_{i}.jpg", np.array(image))
# painter = GridPainter()
# painter.image_grid(images, rows=2, cols=len(images) // 2)
# plt.imsave("/data/cx/ysp/aigc-smart-painter/assets2/result.jpg", np.array(painter.grid))
