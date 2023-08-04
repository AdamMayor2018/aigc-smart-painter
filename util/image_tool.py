# -- coding: utf-8 --
# @Time : 2023/8/3 14:49
# @Author : caoxiang
# @File : image_tool.py
# @Software: PyCharm
# @Description:

# 长边缩放到512，短边等比例缩放

def resize_to_fold(width, height, max_size=512):
    if width > height:
        infer_height = int(height * max_size / width)
        infer_width = max_size
    else:
        infer_width = int(width * max_size / height)
        infer_height = max_size
    return infer_width, infer_height