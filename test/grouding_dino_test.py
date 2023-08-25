# @Time : 2023/8/20 22:32 
# @Author : CaoXiang
# @Description: 测试grounding dino的自动标注功能
import time
import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate
import numpy as np
from pathlib import Path
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from torchvision.ops import box_convert

# IMAGE_PATH = "/data/cx/datasets/nianhui/FormatFactoryPart1.mp4_20230821_134056.484.jpg"
# TEXT_PROMPT = "chair . person . dog ."


def dino_predict(image_path):
    image_source, image = load_image(image_path)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    h, w, _ = image_source.shape
    xyxy = box_convert(boxes=boxes * torch.Tensor([w, h, w, h]), in_fmt="cxcywh", out_fmt="xyxy").numpy()
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    return annotated_frame, xyxy


# plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
# plt.show()
# cv2.imshow("annotated_image", annotated_frame)
# cv2.imwrite("annotated_image.jpg", annotated_frame)

def create_if_not_exsits(*dirs):
    for dir in dirs:
        Path.mkdir(dir, parents=True, exist_ok=True)


def create_output_dirs(dir):
    # 创建输出目录
    assert os.path.exists(dir), f"{dir} not exists"
    path = Path(dir).resolve().parent
    labeled_images_dir = path / "labeled_images"
    label_path = path / "labels"
    create_if_not_exsits(labeled_images_dir, label_path)
    return labeled_images_dir, label_path


def label_all_images(image_dir, label_dir, labled_images_dir):
    for file in tqdm(os.listdir(image_dir)):
        if file.endswith(".jpg"):
            image_path = os.path.join(image_dir, file)
            annotated_frame, xyxys = dino_predict(image_path)
            cv2.imwrite(os.path.join(labled_images_dir, file.replace(".jpg", "_labeled.jpg")), annotated_frame)
            with open(os.path.join(label_dir, file.replace(".jpg", ".txt")), "w") as f:
                for xyxy in xyxys:
                    f.write("0 " + " ".join([str(int(x)) for x in xyxy]) + "\n")


if __name__ == '__main__':
    TEXT_PROMPT = "human body"
    BOX_TRESHOLD = 0.4
    TEXT_TRESHOLD = 0.25
    model = load_model("/data/cx/ysp/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
                       "/data/cx/ysp/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth")
    input_dir_list = ["/data/cx/datasets/chouyang/10mjump/images",
                      "/data/cx/datasets/chouyang/climb/images",
                      "/data/cx/datasets/chouyang/skate/images",
                      "/data/cx/datasets/chouyang/streetdance/images",
                      "/data/cx/datasets/chouyang/ticao/images"
                      ]
    for input_dir in input_dir_list:
        labeled_images_dir, label_path = create_output_dirs(input_dir)
        start = time.time()
        label_all_images(input_dir, label_path, labeled_images_dir)
        end = time.time()
        print(f"total time spent: {(end - start)}s")
