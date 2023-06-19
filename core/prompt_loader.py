# @Time : 2023/6/9 21:35 
# @Author : CaoXiang
# @Description: 用来管理出图的基础prompt

BASE_POS_PROMPT = "(masterpeice),(best quality),highres,HDR"
BASE_NEG_PROMPT = "nsfw,(worst quality,low quality,extra digits:1.2),right lantern,brightness,(nipples:1.2),pussy," \
                  "EasyNegative,(worst quality:2),(low quality:2),(normal quality:2),lowres,((monochrome))," \
                  "((grayscale)),skin spots,acnes,skin blemishes,age spot,glans,extra fingers,fewer fingers," \
                  "strange fingers,bad hand,bare thights,mole,anime,(painting by bad-artist:0.9),watermark,text," \
                  "error,blurry,jpeg artifacts,cropped,signature,username,artist name,bad anatomy,extra fingers," \
                  "fewer fingers,poor drawn hand,occlusion"


class PromptManager:
    def __init__(self, pos_prompt="", neg_prompt=""):
        self._base_pos_prompt = pos_prompt if pos_prompt else BASE_POS_PROMPT
        self._base_neg_prompt = neg_prompt if neg_prompt else BASE_NEG_PROMPT

    @property
    def base_pos_prompt(self):
        return self._base_pos_prompt

    @base_pos_prompt.setter
    def base_pos_prompt(self, new_prompt):
        self._base_pos_prompt = new_prompt

    @property
    def base_neg_prompt(self):
        return self._base_neg_prompt

    @base_neg_prompt.setter
    def base_neg_prompt(self, new_prompt):
        self._base_neg_prompt = new_prompt

    def generate_pos_prompt(self, new_info=""):
        return self._base_pos_prompt + "," + new_info



    def generate_neg_prompt(self, new_info=""):
        return self._base_neg_prompt + "," + new_info
