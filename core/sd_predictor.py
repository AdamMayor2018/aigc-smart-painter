# @Time : 2023/6/16 10:56 
# @Author : CaoXiang
# @Description:文生图本地推理
import typing

import numpy as np

from config.conf_loader import YamlConfigLoader
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
import torch


class BasePredictor:
    def load_pipe(self):
        raise NotImplementedError("load_pipe method not implemented!")

    def tti_inference(self):
        raise NotImplementedError("tti_inference method not implemented!")


class StableDiffusionPredictor:
    def __init__(self, config_loader: YamlConfigLoader):
        self.config_loader = config_loader
        self.model_path = self.config_loader.attempt_load_param("base_model_path")
        self.fp16 = self.config_loader.attempt_load_param("fp16")
        self.fp16 = torch.float16 if self.fp16 else torch.float32
        self.device = f"cuda:{self.config_loader.attempt_load_param('device')}"
        self.scheduler = self.config_loader.attempt_load_param("scheduler")
        self.load_pipes()

    def load_pipes(self):
        scheduler = globals()[self.scheduler].from_pretrained(self.model_path, subfolder="scheduler")
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(self.model_path, torch_dtype=self.fp16, scheduler=scheduler)
        self.iti_pipe = StableDiffusionImg2ImgPipeline(**self.sd_pipe.components)
        self.inpaint_pipe = StableDiffusionInpaintPipeline(**self.sd_pipe.components)
        self.sd_pipe.to(self.device)
        self.iti_pipe.to(self.device)
        self.inpaint_pipe.to(self.device)

    # 文生图接口
    def tti_inference(self, prompt: typing.Union[str, typing.Sequence[str]], **kwargs):
        images = self.sd_pipe(prompt, **kwargs)
        return images

    # 图生图接口（text guided）
    def iti_inference(self, prompt: typing.Union[str, typing.Sequence[str]], init_image, **kwargs):
        images = self.iti_pipe(prompt=prompt, image=init_image, **kwargs)
        return images

    # 蒙版重绘接口
    def inpaint_inference(self, prompt: typing.Union[str, typing.Sequence[str]], init_image, mask_image, **kwargs):
        images = self.inpaint_pipe(prompt=prompt, image=init_image,mask_image=mask_image,  **kwargs)
        return images
