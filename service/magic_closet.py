# -- coding: utf-8 --
# @Time : 2023/7/5 13:45
# @Author : caoxiang
# @File : magic_closet.py
# @Software: PyCharm
# @Description: 提供面向”魔法衣橱的“服务功能
import numpy as np
import cv2
from core.sd_predictor import StableDiffusionPredictor
from model_plugin.diffusers.pipelines.controlnet import MultiControlNetModel
from config.conf_loader import YamlConfigLoader
from core.prompt_loader import PromptManager
from util.image_tool import resize_to_fold
from PIL import Image
import copy

class BaseMagicTool:
    def __init__(self, config_loader: YamlConfigLoader):
        self.config_loader = config_loader
        self.base_predictor = StableDiffusionPredictor(self.config_loader)


class MagicCloset(BaseMagicTool):
    def __init__(self, config_loader):
        super().__init__(config_loader=config_loader)
        self.pm = PromptManager()

    def retrieve_mask(self, image, seg_method="segformer", sam_method="box", input_data=None, input_label=None):
        # 获取蒙版
        pass

    # 更换模特头部
    def swap_head(self):
        pass

    # 更换背景
    def swap_background(self, image, seg_method="segformer", sam_method="box", input_data=None, input_label=None, prompt="", num_images_per_prompt=4, num_inference_steps=50, strength=1,
                        guidance_scale=7.5, reverse=True, smart_mode=False):
        image = image.convert('RGB')
        width, height = image.size
        if smart_mode:
            infer_width, infer_height = resize_to_fold(width, height)
            input_data = input_data * np.array([infer_width / width, infer_height/ height, infer_width/ width, infer_height/ height])
            input_data = np.trunc(input_data)
        else:
            infer_height = height
            infer_width = width
        init_image = cv2.resize(np.array(image), (infer_width, infer_height))
        #mask_image = cv2.resize(mask_image, (infer_width, infer_height))
        if seg_method == "segformer":
            mask_image = self.base_predictor.segformer_mask_inference(image=init_image, part="background", reverse=reverse)
            mask_image = cv2.dilate(mask_image, kernel=np.ones((3, 3), np.uint8), iterations=2)
        elif seg_method == "sam":
            scores, mask_image = self.base_predictor.sam_mask_inferece(image=init_image, method=sam_method, input_data=input_data, input_label=input_label, reverse=reverse)
        else:
            raise ValueError("seg_method must be 'segformer' or 'sam'")


        # images = self.base_predictor.inpaint_inference(prompt=prompt, init_image=init_image,
        #                                                mask_image=Image.fromarray(mask_image),
        #                                                num_images_per_prompt=num_images_per_prompt,
        #                                                num_inference_steps=num_inference_steps,
        #                                                strength=strength,
        #                                                guidance_scale=guidance_scale, width=width // 8 * 8,
        #                                                height=height // 8 * 8).images
        # 1.获取蒙版边缘
        control_image_canny = self.base_predictor.controlnet_aux_inferece(mode="canny", image=mask_image)
        #mask_image = copy.deepcopy(mask_image).resize((752, 1000))
        medias_init_image = cv2.resize(init_image, (512, 512))
        control_image_midas = self.base_predictor.controlnet_aux_inferece(mode="midas", image=medias_init_image)
        control_image_midas = control_image_midas.resize((width, height))
        control_image_midas.save("/data/cx/ysp/aigc-smart-painter/assets2/medias.jpg")
        control_images = [control_image_canny, control_image_midas]
        #control_images = [control_image_canny, control_image_midas]
        # 只使用单个controlnet 索引1是canny 索引2是depth
        self.base_predictor.prepared_pipes["controlnet_inpaint_multi"].controlnet = MultiControlNetModel(self.base_predictor.controlnets[1:])
        init_image = Image.fromarray(init_image) if isinstance(init_image, np.ndarray) else init_image
        mask_image = Image.fromarray(mask_image) if isinstance(mask_image, np.ndarray) else mask_image
        #mask_image = mask_image.resize((750, 1000))
        images = self.base_predictor.controlnet_inpaint_inference(mode="multi",
                                                                  prompt=self.pm.generate_pos_prompt(prompt),
                                                                  image=init_image,
                                                                  mask_image=mask_image,
                                                                  num_images_per_prompt=num_images_per_prompt,
                                                                  num_inference_steps=num_inference_steps,
                                                                  guidance_scale=guidance_scale, control_images=control_images,
                                                                  strength=strength).images
        if smart_mode:
            images = [img.resize((width, height)) for img in images]
            mask_image = mask_image.resize((width, height))



        return mask_image, images

    # 保留衣服更换model
    def swap_model(self, image, parts, prompt, negative_prompt, num_images_per_prompt, num_inference_steps=50,
                   strength=0.75, reverse=True):
        image = image.convert('RGB')
        if negative_prompt == "":
            negative_prompt = self.pm.generate_neg_prompt("big head")
        # 1.获取衣服蒙版
        mask_image = self.base_predictor.segformer_multi_mask_inference(image, *parts, reverse=reverse)

        # 2.获取模特的pose
        control_image_pose = self.base_predictor.controlnet_aux_inferece(mode="openpose", image=image,
                                                                         hand_and_face=True)
        # 3.获取蒙版的边缘
        control_image_canny = self.base_predictor.controlnet_aux_inferece(mode="canny", image=mask_image)
        control_images = [control_image_pose, control_image_canny]
        # 4.inpaint获取更改后的图像
        images = self.base_predictor.controlnet_inpaint_inference(mode="multi",
                                                                  prompt=self.pm.generate_pos_prompt(prompt),
                                                                  negative_prompt=negative_prompt, image=image,
                                                                  mask_image=Image.fromarray(mask_image),
                                                                  num_images_per_prompt=num_images_per_prompt,
                                                                  num_inference_steps=num_inference_steps,
                                                                  guidance_scale=7.5, control_images=control_images,
                                                                  strength=strength).images
        return mask_image, control_images, images

    # 保留model,自定义衣服(服装裂变)
    def swap_cloth(self):
        pass
