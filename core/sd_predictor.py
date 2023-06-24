# @Time : 2023/6/16 10:56 
# @Author : CaoXiang
# @Description:文生图本地推理
import copy
import typing

import numpy as np

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
                scheduler = globals()[config.get("scheduler")].from_pretrained(self.model_path, subfolder="scheduler")
                loaded_pipe = globals()[config.get("class_name")].from_pretrained(config.get("pretrained_model_name_or_path"), scheduler=scheduler, torch_dtype=eval(config.get("torch_dtype")), **extra_params)
                loaded_pipe.to(f"{config.get('device')}")
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


if __name__ == '__main__':
    sdp = StableDiffusionPredictor(YamlConfigLoader("/data/cx/ysp/aigc-smart-painter/config/general_config.yaml"))

