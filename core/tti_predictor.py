# @Time : 2023/6/16 10:56 
# @Author : CaoXiang
# @Description:文生图本地推理
import typing

from config.conf_loader import YamlConfigLoader
from diffusers import StableDiffusionPipeline
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
        self.model_path = self.config_loader.attempt_load_param("model_path")
        self.fp16 = self.config_loader.attempt_load_param("fp16")
        self.fp16 = torch.float16 if self.fp16 else torch.float32
        self.device = f"cuda:{self.config_loader.attempt_load_param('device')}"
        self.scheduler = self.config_loader.attempt_load_param("scheduler")
        self.load_pipe()


    def load_pipe(self):
        scheduler = globals()[self.scheduler].from_pretrained(self.model_path, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_path, torch_dtype=self.fp16, scheduler=scheduler)
        self.pipe.to(self.device)

    def tti_inference(self, prompt:typing.Union[str, typing.Sequence[str]], **kwargs):
        images = self.pipe(prompt, **kwargs)
        return images