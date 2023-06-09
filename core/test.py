# @Time : 2023/6/9 11:44 
# @Author : CaoXiang
# @Description:
import torch
import diffusers
from diffusers import DiffusionPipeline
from prompt_loader import PromptManager
pm = PromptManager()

pipeline = DiffusionPipeline.from_pretrained("/data/cx/ysp/aigc-smart-painter/models/stable-diffusion-v1-5")
pipeline.to('cuda:2')
prompt = "An image of a squirrel in Picasso style"
prompt = pm.generate_pos_prompt(prompt)
image = pipeline(prompt).images[0]
image.save("squirrel.png")