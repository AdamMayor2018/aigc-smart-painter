# @Time : 2023/6/16 11:28 
# @Author : CaoXiang
# @Description: 画画工具类

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class GridPainter:
    def __init__(self, figsize=(16, 8)):
        self.figsize = figsize

    def image_grid(self, imgs, rows, cols):
        assert len(imgs) == rows * cols
        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols * w, rows * h))
        grid_w, grid_h = grid.size
        for i, img in enumerate(imgs):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        self.grid = grid

    def image_show(self):
        plt.figure(figsize=self.figsize)
        plt.imshow(np.array(self.grid))
        plt.show()
