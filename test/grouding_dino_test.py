# @Time : 2023/8/20 22:32 
# @Author : CaoXiang
# @Description: 测试grounding dino的自动标注功能
from groundingdino.util.inference import load_model, load_image, predict, annotate
import numpy as np
import cv2
import matplotlib.pyplot as plt
model = load_model("/data/cx/ysp/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
                   "/data/cx/ysp/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth")
IMAGE_PATH = "/data/cx/datasets/panyan/climb_all/climb_N_5287.jpg"
#TEXT_PROMPT = "chair . person . dog ."
TEXT_PROMPT = "human body"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
plt.show()
#cv2.imshow("annotated_image", annotated_frame)
#cv2.imwrite("annotated_image.jpg", annotated_frame)
