# @Time : 2023/6/16 10:56 
# @Author : CaoXiang
# @Description:文生图本地推理
import copy
import typing

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.nn as nn
from config.conf_loader import YamlConfigLoader
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel
)
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler
)

import torch
from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation

part_pairs = {
    "background": "0",
    "hat": "1",
    "hair": "2",
    "sunglasses": "3",
    "upper-clothes": "4",
    "skirt": "5",
    "pants": "6",
    "dress": "7",
    "belt": "8",
    "left-shoe": "9",
    "right-shoe": "10",
    "face": "11",
    "left-leg": "12",
    "right-leg": "13",
    "left-arm": "14",
    "right-arm": "15",
    "bag": "16",
    "scarf": "17"
}


class BasePredictor:
    def load_pipes(self):
        raise NotImplementedError("load_pipe method not implemented!")


class StableDiffusionPredictor:
    def __init__(self, config_loader: YamlConfigLoader):
        self.config_loader = config_loader
        self.model_path = self.config_loader.attempt_load_param("base_model_path")
        self.fp16 = torch.float16 if self.config_loader.attempt_load_param("fp16") else torch.float32
        self.device = f"{self.config_loader.attempt_load_param('device')}"
        self.scheduler = self.config_loader.attempt_load_param("scheduler")
        self.load_pipes()

    def load_pipes(self):
        pipes = self.config_loader.attempt_load_param("pipes")
        self.prepared_pipes = {}
        for name, config in pipes.items():
            extra_params = {}
            if config.get("mode") == "on":
                for subname, subconfig in config.items():
                    if isinstance(subconfig, dict) and "class_name" in subconfig.keys():
                        class_name = subconfig.get("class_name")
                        subconfig["torch_dtype"] = eval(subconfig.get("torch_dtype"))
                        fix_subconfig = copy.deepcopy(subconfig)
                        fix_subconfig.pop("class_name")
                        extra_params = {subname: globals()[class_name].from_pretrained(**fix_subconfig)}
                try:
                    scheduler = globals()[config.get("scheduler")].from_pretrained(self.model_path, subfolder="scheduler")
                    loaded_pipe = globals()[config.get("class_name")].from_pretrained(config.get("pretrained_model_name_or_path"), scheduler=scheduler, torch_dtype=eval(config.get("torch_dtype")), **extra_params)
                except KeyError:
                    loaded_pipe = globals()[config.get("class_name")].from_pretrained(config.get("pretrained_model_name_or_path"), **extra_params)
                try:
                    loaded_pipe.to(f"{config.get('device')}")
                except:
                    pass
                self.prepared_pipes[name] = loaded_pipe

    # 文生图接口
    def tti_inference(self, prompt: typing.Union[str, typing.Sequence[str]], **kwargs):
        images = self.prepared_pipes["txt2img"](prompt, **kwargs)
        return images

    # 图生图接口（text guided）
    def iti_inference(self, prompt: typing.Union[str, typing.Sequence[str]], init_image, **kwargs):
        images = self.prepared_pipes["img2img"](prompt=prompt, image=init_image, **kwargs)
        return images

    # 蒙版重绘接口
    def inpaint_inference(self, prompt: typing.Union[str, typing.Sequence[str]], init_image, mask_image, **kwargs):
        images = self.prepared_pipes["inpaint"](prompt=prompt, image=init_image,mask_image=mask_image,  **kwargs)
        return images

    # controlnet接口
    def controlnet_inference(self, mode: str, init_image, **kwargs):
        if mode == "openpose":
            images = self.prepared_pipes["openpose-control"](image=init_image, hand_and_face=True, **kwargs)
        elif mode == "canny":
            images = self.prepared_pipes["canny-control"](image=init_image, **kwargs)
        else:
            raise ValueError("mode must be one of ['openpose', 'canny']")
        return images

    def segformer_mask_inference(self, image, part="upper-clothes"):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        inputs = self.prepared_pipes["cloth_feature_extractor"](images=image, return_tensors="pt")
        outputs = self.prepared_pipes["segformer"](**inputs)
        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        if part.lower() != "background":
            pred_seg[pred_seg != int(part_pairs[part.lower()])] = 0
        else:
            pred_seg[pred_seg == 0] = 255
            pred_seg[pred_seg != 255] = 0
        arr_seg = pred_seg.cpu().numpy().astype("uint8")
        arr_seg *= 255
        return arr_seg



if __name__ == '__main__':
    sdp = StableDiffusionPredictor(YamlConfigLoader("/data/cx/ysp/aigc-smart-painter/config/general_config.yaml"))
    image = Image.open("/data/cx/ysp/aigc-smart-painter/assets/cloth1.jpg")
    image = np.array(image)
    mask = sdp.segformer_mask_inference(image, part="face")
    plt.imshow(mask)
    plt.show()


