# @Time : 2022/8/22 16:03 
# @Author : CaoXiang
# @Description: 项目相关的自定义异常
class ParamLoadError(Exception):
    detail = "Param 【%s】 load error, check opt and config path."

    def __init__(self, param_name):
        if param_name is not None:
            self.param_name = param_name

    def __str__(self):
        return str(self.detail % self.param_name)


if __name__ == '__main__':
    raise ParamLoadError("model_path")