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


class Sampainter:
    def __init__(self, figsize=(16, 8)):
        self.figsize = figsize

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)

    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))