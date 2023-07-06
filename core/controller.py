# -- coding: utf-8 --
# @Time : 2023/6/20 16:45
# @Author : caoxiang
# @File : controller.py
# @Software: PyCharm
# @Description: controlnet控制相关的模块功能

from controlnet_aux import OpenposeDetector, CannyDetector


class OpenPosePreProcessor:
    def __init__(self, aux_model_path):
        self.canny = None
        self.openpose = None
        self.aux_model_path = aux_model_path
        self.load_aux_models()

    def load_aux_models(self, openpose=True, canny=True):
        self.openpose = OpenposeDetector.from_pretrained(self.aux_model_path) if openpose else None
        self.canny = CannyDetector() if canny else None

    def aux_infer(self, img, mode: str):
        if mode == "openpose":
            processed_image = self.openpose(img, hand_and_face=True)
        elif mode == "canny":
            processed_image = self.canny(img)
        else:
            raise ValueError("mode must be one of ['openpose', 'canny']")
        return processed_image
