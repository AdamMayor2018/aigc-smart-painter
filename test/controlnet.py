# -- coding: utf-8 --
# @Time : 2023/6/20 16:45
# @Author : caoxiang
# @File : controller.py
# @Software: PyCharm
# @Description: controlnet控制相关的模块功能
import requests
from PIL import Image
from io import BytesIO
from controlnet_aux import CannyDetector, OpenposeDetector
from util.painter import GridPainter
# load image
url = "/data/cx/ysp/aigc-smart-painter/assets/pose.png"
img = Image.open(url).convert("RGB").resize((512, 512))

# load checkpoints
# hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
# midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
# mlsd = MLSDdetector.from_pretrained("lllyasviel/Annotators")
open_pose = OpenposeDetector.from_pretrained("/data/cx/ysp/aigc-smart-painter/models/control-net-aux-models/Annotators")
# pidi = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
# normal_bae = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
# lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
# lineart_anime = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
# zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
# sam = SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")
# leres = LeresDetector.from_pretrained("lllyasviel/Annotators")

# instantiate
# canny = CannyDetector()
# content = ContentShuffleDetector()
# face_detector = MediapipeFaceDetector()


# process
# processed_image_hed = hed(img)
# processed_image_midas = midas(img)
# processed_image_mlsd = mlsd(img)
processed_image_open_pose = open_pose(img, hand_and_face=True)
# processed_image_pidi = pidi(img, safe=True)
# processed_image_normal_bae = normal_bae(img)
# processed_image_lineart = lineart(img, coarse=True)
# processed_image_lineart_anime = lineart_anime(img)
# processed_image_zoe = zoe(img)
# processed_image_sam = sam(img)
# processed_image_leres = leres(img)

# processed_image_canny = canny(img)
# processed_image_content = content(img)
# processed_image_mediapipe_face = face_detector(img)


painter = GridPainter()
painter.image_grid([processed_image_open_pose], rows=1, cols=1)
painter.image_show()