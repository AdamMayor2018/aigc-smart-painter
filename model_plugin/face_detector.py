# @Time : 2023/7/1 16:09 
# @Author : CaoXiang
# @Description: 人脸检测工具类
from config.conf_loader import YamlConfigLoader
from model_plugin.yolov5_face.models.yolo import Model
from model_plugin.yolov5_face.utils.general import non_max_suppression_face, scale_coords, check_img_size
import torch
from model_plugin.yolov5_face.utils.datasets import letterbox
from util.painter import draw_box, show_image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import copy


class BaseDetector:
    def __init__(self, config_loader):
        self.config_loader = config_loader

    def load_model(self):
        raise NotImplementedError

    def detect(self, image):
        raise NotImplementedError

class Yolov5FaceDetector(BaseDetector):
    def __init__(self, config_loader):
        super().__init__(config_loader)
        self.face_config = config_loader.attempt_load_param("yolo")
        self.device = self.face_config.get("device")
        self.weight_yaml = self.face_config.get("weight_yaml_path")
        self.checkpoint_path = self.face_config.get("checkpoint_path")
        self.use_fp16 = self.face_config.get("use_fp16")
        self.img_size = self.face_config.get("img_size")
        self.conf_thres = self.face_config.get("conf_thres")
        self.iou_thres = self.face_config.get("iou_thres")
        self.load_model()

    def load_model(self):
        model = Model(self.weight_yaml)
        weights = torch.load(self.checkpoint_path, map_location="cpu")
        model.load_state_dict(weights)
        model = model.to(self.device)
        if self.use_fp16:
            model = model.fuse().half().eval()
        else:
            model = model.fuse().float().eval()
        self.model = model

    def process_image(self, image):
        img_cp = copy.deepcopy(image)
        h0, w0 = image.shape[:2]
        r = self.img_size / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img_cp = cv2.resize(image, (int(w0 * r), int(h0 * r)), interpolation=interp)
        try:
            imgsz = check_img_size(self.img_size, s=self.model.stride.max())
        except Exception:
            imgsz = check_img_size(self.img_size, s=32)
        # letter box 外接矩形加灰条，auto=False是长边到imgsz， 然后再加灰条
        img_lb = letterbox(img_cp, new_shape=imgsz, auto=False)[0]
        # convert BGR -> RGB C,H,W
        # cv2.imwrite("xx.jpg", img_lb)
        img_lb = img_lb[:, :, ::-1].transpose(2, 0, 1).copy()
        return img_lb

    def detect(self, image):
        input = self.process_image(image)
        input = input / 255.0
        if input.ndim == 3:
            input = input[np.newaxis, ...]
        input = torch.from_numpy(input)
        if self.use_fp16:
            input = input.half()
        preds = self.model(input.to(self.device))[0]
        preds = non_max_suppression_face(preds, self.conf_thres, self.iou_thres)[0][:, :5]
        largest_box = []
        if preds.shape[0] > 0:
            preds[:, :4] = scale_coords(input.shape[2:], preds[:, :4], image.shape).round()
            preds = preds.detach().cpu().numpy()
            boxes = preds[:, :4].astype(int)
            conf = preds[:, 4]
            # 只返回一张图里置信度最大的
            largest_box = boxes[np.lexsort((conf,))].tolist()[-1]
        return largest_box


if __name__ == '__main__':
    config_loader = YamlConfigLoader(yaml_path="/data/cx/ysp/aigc-smart-painter/config/general_config.yaml")
    detector = Yolov5FaceDetector(config_loader)
    image = Image.open("/data/cx/ysp/aigc-smart-painter/assets/cloth1.jpg")
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    show_image(image)
    result = detector.detect(image=image)
    image = draw_box(image, cords=result[:4], color=(255, 0, 0), thickness=2)
    show_image(image)
