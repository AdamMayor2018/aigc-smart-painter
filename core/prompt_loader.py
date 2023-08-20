# @Time : 2023/6/9 21:35 
# @Author : CaoXiang
# @Description: 用来管理出图的基础prompt

BASE_POS_PROMPT = "(masterpeice),(best quality),highres,HDR,"
BASE_NEG_PROMPT = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"


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
