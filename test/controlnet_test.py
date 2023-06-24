# @Time : 2023/6/24 17:19 
# @Author : CaoXiang
# @Description:测试controlnet

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from core.prompt_loader import PromptManager
from controlnet_aux import CannyDetector, OpenposeDetector

def show_image(image):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    plt.figure()
    plt.imshow(image)
    plt.show()

canny = CannyDetector()
pm = PromptManager()
image = Image.open("/data/cx/ysp/aigc-smart-painter/assets/input_image_vermeer.png")
show_image(image)
controlnet = ControlNetModel.from_pretrained("/data/cx/ysp/aigc-smart-painter/models/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "/data/cx/ysp/aigc-smart-painter/models/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.to("cuda:6")

canny_image = canny(image)
show_image(canny_image)

out_image = pipe(
    pm.generate_pos_prompt("disco dancer with colorful lights"), num_inference_steps=20, image=canny_image, negative_prompt=pm.generate_neg_prompt()
).images[0]

show_image(out_image)

