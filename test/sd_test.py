# @Time : 2023/6/16 11:41 
# @Author : CaoXiang
# @Description:

from core.sd_predictor import StableDiffusionPredictor
from util.painter import GridPainter
from config.conf_loader import YamlConfigLoader
from core.prompt_loader import PromptManager
conf_loader = YamlConfigLoader(yaml_path="/data/cx/ysp/aigc-smart-painter/config/general_config.yaml")
sdp = StableDiffusionPredictor(config_loader=conf_loader)
painter = GridPainter(figsize=(16, 8))
pt = PromptManager()
if __name__ == '__main__':
    prompt = "a photograph of an astronaut riding a horse"
    images = sdp.tti_inference(prompt=pt.generate_pos_prompt(prompt), negative_prompt=pt.generate_neg_prompt()).images
    painter.image_grid(images, rows=1, cols=1)
    painter.image_show()