# -- coding: utf-8 --
# @Time : 2023/8/21 15:30
# @Author : caoxiang
# @File : gradio_dino_test.py
# @Software: PyCharm
# @Description:

import numpy as np
import gradio as gr
from PIL import Image
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict, annotate
from typing import Tuple, List
import torch

def load_image(input_data: np.ndarray) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.fromarray(input_data).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def dino_predict(image, prompt):
    image_source, image = load_image(image)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=prompt,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    return annotated_frame


if __name__ == '__main__':
    TEXT_PROMPT = "human body"
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25
    model = load_model("/data/cx/ysp/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
                       "/data/cx/ysp/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth")


    demo = gr.Interface(
        fn=dino_predict,
        inputs=["image", "text"],
        outputs="image",
    )

    demo.launch(server_port=6688, share=True)