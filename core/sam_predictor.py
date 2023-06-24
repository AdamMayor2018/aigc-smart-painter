# @Time : 2023/6/24 20:36 
# @Author : CaoXiang
# @Description: segment-anything(SAM)工具类 用于分割得到各种蒙版
from config.conf_loader import YamlConfigLoader
from segment_anything import sam_model_registry, SamPredictor


class SamModel:
    def __init__(self, config_loader: YamlConfigLoader):
        self.config_loader = config_loader
        self.sam_config = config_loader.attempt_load_param("sam")
        self.checkpoint = self.sam_config.get("sam_checkpoint")
        self.model_type = self.sam_config.get("sam_model_type")
        self.device = self.sam_config.get("device")
        self.model = None
        self.predictor = None

    def load_model(self):
        try:
            self.model = sam_model_registry[self.model_type](self.checkpoint)
        except Exception as e:
            print(f"Error while loading sam model.check 'sam_model_type' and 'sam_checkpoint' in config file. {e}")
        self.model.to(self.device)
        self.predictor = SamPredictor(self.model)

    def sam_prepare(self):
        pass
