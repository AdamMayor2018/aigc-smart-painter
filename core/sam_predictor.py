# @Time : 2023/6/24 20:36 
# @Author : CaoXiang
# @Description: segment-anything(SAM)工具类 用于分割得到各种蒙版
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from util.painter import Sampainter
from config.conf_loader import YamlConfigLoader
from segment_anything import sam_model_registry, SamPredictor

class RawSeger:
    def __init__(self):
        pass
    def prompt_with_box(self, image, box, reverse=False):
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        mask = np.zeros_like(image)
        xmin, ymin, xmax, ymax = box
        mask[ymin:ymax, xmin:xmax, :] = 255
        mask = mask.astype(np.uint8)
        if reverse:
            mask = 255 - mask
        return mask


class SamSeger:
    def __init__(self, config_loader: YamlConfigLoader):
        self.config_loader = config_loader
        self.sam_config = config_loader.attempt_load_param("sam")
        self.checkpoint = self.sam_config.get("checkpoint")
        self.model_type = self.sam_config.get("model_type")
        self.device = self.sam_config.get("device")
        self.model = None
        self.predictor = None
        self.load_model()

    def load_model(self):
        try:
            self.model = sam_model_registry[self.model_type](self.checkpoint)
        except Exception as e:
            print(f"Error while loading sam model.check 'sam_model_type' and 'sam_checkpoint' in config file. {e}")
        self.model.to(self.device)
        self.predictor = SamPredictor(self.model)

    def sam_prepare(self, image):
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        self.predictor.set_image(image)

    def prompt_with_points(self, input_point, input_label=np.array([1]), multimask_output=True, select_best=True):
        '''
            使用points作为prompt进行指向分割
        :param input_point:分割的参考点
        :param input_label:mask的数值
        :param multimask_output:
        :return:
        '''
        assert len(input_point) == len(input_label), "num points must equal to num labels"
        if not isinstance(input_point, np.ndarray):
            input_point = np.array(input_point)
        if not isinstance(input_label, np.ndarray):
            input_label = np.array(input_label)
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=multimask_output,
        )
        masks = masks.astype(np.uint8)
        if select_best:
            masks = masks[np.argmax(scores)]
            scores = np.max(scores)
        return scores, masks

    def prompt_with_boxes(self, input_box,  multimask_output=True, select_best=True, reverse=False):
        """
            使用boundingbox作为prompt进行指向分割
        :param input_box: 输入的指向box坐标 xyxy格式
        :param multimask_output:
        :param reverse:是否进行蒙版反向 默认是蒙版为1
        :return:
        """
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=multimask_output,
        )
        masks = masks.astype(np.uint8)
        if reverse:
            masks = 1 - masks
        if select_best:
            masks = masks[np.argmax(scores)]
            scores = np.max(scores)
        return scores, masks




if __name__ == '__main__':
    image = Image.open("/data/cx/ysp/aigc-smart-painter/assets/cloth1.jpg")
    config_loader = YamlConfigLoader(yaml_path="/data/cx/ysp/aigc-smart-painter/config/general_config.yaml")
    sam = SamSeger(config_loader)
    sam.sam_prepare(image)
    input_point = np.array([[500, 375], [300, 400], [350, 200], [400, 100]])
    input_label = np.array([0, 0, 1, 1])
    painter = Sampainter(figsize=(10, 10))

    scores, masks = sam.prompt_with_points(input_point, input_label)
    print(scores, masks)

    plt.imshow(masks)
    plt.show()

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    painter.show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show()

    # plt.imshow(image)
    # plt.axis('on')
    # plt.show()
    # input_box = np.array([220, 20, 500, 300])
    # # input_point = np.array([[575, 750]])
    # input_label = np.array([0])
    # scores, masks = sam.prompt_with_boxes(input_box=input_box, reverse=True)
    # plt.imshow(masks)
    # plt.show()
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # painter.show_mask(masks, plt.gca())
    # painter.show_box(input_box, plt.gca())
    # #painter.show_points(input_point, input_label, plt.gca())
    # plt.axis('off')
    # plt.show()