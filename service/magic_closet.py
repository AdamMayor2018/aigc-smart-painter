# -- coding: utf-8 --
# @Time : 2023/7/5 13:45
# @Author : caoxiang
# @File : magic_closet.py
# @Software: PyCharm
# @Description: 提供面向”魔法衣橱的“服务功能
import numpy as np
import cv2
from core.sd_predictor import StableDiffusionPredictor
from config.conf_loader import YamlConfigLoader
from core.prompt_loader import PromptManager
from PIL import Image

class BaseMagicTool:
    def __init__(self, config_loader: YamlConfigLoader):
        self.config_loader = config_loader
        self.base_predictor = StableDiffusionPredictor(self.config_loader)


class MagicCloset(BaseMagicTool):
    def __init__(self, config_loader):
        super().__init__(config_loader=config_loader)
        self.pm = PromptManager()

    # 更换模特头部
    def swap_head(self):
        pass

    # 更换背景
    def swap_background(self, image, prompt, num_images_per_prompt, num_inference_steps=50, strength=1,guidance_scale=7.5, reverse=True):
        image = image.convert('RGB')
        width, height = image.size
        mask_image = self.base_predictor.segformer_mask_inference(image=image, part="background", reverse=reverse)
        # mask_image = cv2.dilate(mask_image, kernel=np.ones((3,3),np.uint8), iterations=3)
        # mask_image = cv2.erode(mask_image, kernel=np.ones((3, 3), np.uint8), iterations=1)
        #mask_image = cv2.erode(mask_image, kernel=np.ones((5,5),np.uint8), iterations=2)
        images = self.base_predictor.inpaint_inference(prompt=prompt, init_image=image,
                                              mask_image=Image.fromarray(mask_image),
                                              num_images_per_prompt=num_images_per_prompt,
                                              num_inference_steps=num_inference_steps,
                                              strength=strength,
                                              guidance_scale=guidance_scale, width=width // 8 * 8, height=height // 8 * 8).images
        images = [img.resize((width, height)) for img in images]

        return mask_image, images


    # 保留衣服更换model
    def swap_model(self, image, parts, prompt, negative_prompt, num_images_per_prompt, num_inference_steps=50, strength=0.75, reverse=True):
        image = image.convert('RGB')
        if negative_prompt == "":
            negative_prompt = self.pm.generate_neg_prompt("big head")
        #1.获取衣服蒙版
        mask_image = self.base_predictor.segformer_multi_mask_inference(image, *parts, reverse=reverse)

        #2.获取模特的pose
        control_image_pose = self.base_predictor.controlnet_aux_inferece(mode="openpose", image=image, hand_and_face=True)
        #3.获取蒙版的边缘
        control_image_canny = self.base_predictor.controlnet_aux_inferece(mode="canny", image=mask_image)
        control_images = [control_image_pose, control_image_canny]
        #4.inpaint获取更改后的图像
        images = self.base_predictor.controlnet_inpaint_inference(mode="multi", prompt=self.pm.generate_pos_prompt(prompt), negative_prompt=negative_prompt, image=image,
                                                                  mask_image=Image.fromarray(mask_image), num_images_per_prompt=num_images_per_prompt,
                                                                  num_inference_steps=num_inference_steps, guidance_scale=7.5, control_images=control_images, strength=strength).images
        return mask_image, control_images, images


    # 保留model,自定义衣服(服装裂变)
    def swap_cloth(self):
        pass


