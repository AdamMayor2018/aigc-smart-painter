# -- coding: utf-8 --
# @Time : 2023/7/4 16:28
# @Author : caoxiang
# @File : segformer_cloth.py
# @Software: PyCharm
# @Description: 测试segformer 对于cloth的分割
import torch
from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn

extractor = AutoFeatureExtractor.from_pretrained("/data/cx/ysp/aigc-smart-painter/models/segformer_b2_clothes")
model = SegformerForSemanticSegmentation.from_pretrained("/data/cx/ysp/aigc-smart-painter/models/segformer_b2_clothes")

image_path = r"/data/cx/ysp/aigc-smart-painter/assets/cloth1.jpg"

image = Image.open(image_path)
inputs = extractor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits.cpu()

upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
)

pred_seg = upsampled_logits.argmax(dim=1)[0]
plt.imshow(pred_seg)
plt.show()


def get_mask(image_path):
    image = Image.open(image_path)
    inputs = extractor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]
    return pred_seg