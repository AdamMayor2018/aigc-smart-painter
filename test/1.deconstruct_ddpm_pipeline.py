# -- coding: utf-8 --
# @Time : 2023/6/12 20:16
# @Author : caoxiang
# @File : 1.deconstruct_ddpm_pipeline.py
# @Software: PyCharm
# @Description: 解构一个pipeline


from diffusers import DDPMPipeline
import matplotlib.pyplot as plt
import numpy as np

pretrained_path = "/data/cx/ysp/aigc-smart-painter/models/ddpm-cat-256"

# 1.ddpm的pipeline
ddpm = DDPMPipeline.from_pretrained(pretrained_path).to("cuda")
image = ddpm(num_inference_steps=50).images[0]
plt.figure()
plt.imshow(np.array(image))
plt.show()
#print(image.shape)

#2.解构ddpm的pipeline
#以上过程包含一个Unet2DModel和一个DDPMScheduler，Unet2DModel用来预测当前噪声和随机噪声之间的残差，DDPMScheduler用残差生成一个noise降低的版本，
#然后重复用这个扩散的过程,直到预设值的扩散步数，   最后得到一个高清版本的图片。

from diffusers import DDPMScheduler, UNet2DModel

#step1 加载model和scheduler:
scheduler = DDPMScheduler.from_pretrained(pretrained_path)
model = UNet2DModel.from_pretrained(pretrained_path).to("cuda")

#设置去噪的时间步
scheduler.set_timesteps(50)

#step3 Setting the scheduler timesteps creates a tensor with evenly spaced elements in it, 50 in this example.
# Each element corresponds to a timestep at which the model denoises an image.
# When you create the denoising loop later, you’ll iterate over this tensor to denoise an image:
print(scheduler.timesteps)

#step4 Create some random noise with the same shape as the desired output:

import torch
sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size)).to("cuda")


#step5 Now write a loop to iterate over the timesteps. At each timestep, the model does a UNet2DModel.forward() pass and returns the noisy residual.
# The scheduler’s step() method takes the noisy residual, timestep, and input and it predicts the image at the previous timestep.
# This output becomes the next input to the model in the denoising loop, and it’ll repeat until it reaches the end of the timesteps array.
input = noise

for t in scheduler.timesteps:
    with torch.no_grad():
        noisy_residual = model(input, t).sample
    previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
    input = previous_noisy_sample

#The last step is to convert the denoised output into an image:

from PIL import Image
import numpy as np

image = (input / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
image = Image.fromarray((image * 255).round().astype("uint8"))
