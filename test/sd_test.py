# @Time : 2023/6/16 11:41 
# @Author : CaoXiang
# @Description:

from core.tti_predictor import StableDiffusionPredictor
from util.painter import GridPainter
from config.conf_loader import YamlConfigLoader
conf_loader = YamlConfigLoader(yaml_path="/data/cx/ysp/aigc-smart-painter/config/general_config.yaml")
sdp = StableDiffusionPredictor(config_loader=conf_loader)
painter = GridPainter(figsize=(16, 8))
if __name__ == '__main__':
    prompt = "a photograph of an astronaut riding a horse"
    images = sdp.tti_inference(prompt).images
    painter.image_grid(images, rows=1, cols=1)
    painter.image_show()