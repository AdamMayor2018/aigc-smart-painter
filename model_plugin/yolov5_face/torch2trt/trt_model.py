# @Time : 2022/12/29 14:21 
# @Author : CaoXiang
# @Description:
import pycuda.driver as cuda
import tensorrt as trt
from tensorrt import DataType
import numpy as np
# 声明是使用显示模式 这样的各个维度都是dynamic的 目前官方推荐统一使用显示模式
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)








def GiB(val):
    return val * 1 << 30


def ONNX_to_TRT(onnx_model_path=None, trt_engine_path=None, fp16_mode=False):
    """
    仅适用TensorRT V8版本
    生成cudaEngine，并保存引擎文件(仅支持固定输入尺度)
    Serialized engines不是跨平台和tensorRT版本的 也只
    fp16_mode: True则fp16预测
    onnx_model_path: 将加载的onnx权重路径
    trt_engine_path: trt引擎文件保存路径
    """
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    config.max_workspace_size = GiB(1)
    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
    with open(onnx_model_path, 'rb') as model:
        assert parser.parse(model.read())
        serialized_engine = builder.build_serialized_network(network, config)

    with open(trt_engine_path, 'wb') as f:
        f.write(serialized_engine)  # 序列化

    print('TensorRT file in ' + trt_engine_path)
    print('============ONNX->TensorRT SUCCESS============')


# class TrtModel():
#     '''
#     TensorRT infer
#     '''
#
#     def __init__(self, trt_path, device=1):
#         self.ctx = cuda.Device(device).make_context()
#         stream = cuda.Stream()
#         TRT_LOGGER = trt.Logger(trt.Logger.INFO)
#         runtime = trt.Runtime(TRT_LOGGER)
#
#         # Deserialize the engine from file
#         with open(trt_path, "rb") as f:
#             engine = runtime.deserialize_cuda_engine(f.read())
#         context = engine.create_execution_context()
#
#         host_inputs = []
#         cuda_inputs = []
#         host_outputs = []
#         cuda_outputs = []
#         bindings = []
#
#         for binding in engine:
#             print('bingding:', binding, engine.get_binding_shape(binding))
#             size = trt.volume(engine.get_binding_shape(binding)) #像素数量
#             dtype = trt.nptype(engine.get_binding_dtype(binding)) #np.float32
#             # Allocate host and device buffers
#             host_mem = cuda.pagelocked_empty(size, dtype) #开辟pagelock内存空间
#             cuda_mem = cuda.mem_alloc(host_mem.nbytes) #开辟显存空间 size * 4 （一个np.float32占四个字节）
#             # Append the device buffer to device bindings.
#             bindings.append(int(cuda_mem))
#             # Append to the appropriate list.
#             if engine.binding_is_input(binding):
#                 self.input_w = engine.get_binding_shape(binding)[-1]
#                 self.input_h = engine.get_binding_shape(binding)[-2]
#                 host_inputs.append(host_mem)
#                 cuda_inputs.append(cuda_mem)
#             else:
#                 host_outputs.append(host_mem)
#                 cuda_outputs.append(cuda_mem)
#
#         # Store
#         self.stream = stream
#         self.context = context
#         self.engine = engine
#         self.host_inputs = host_inputs
#         self.cuda_inputs = cuda_inputs
#         self.host_outputs = host_outputs
#         self.cuda_outputs = cuda_outputs
#         self.bindings = bindings
#         self.batch_size = engine.max_batch_size
#
#     def __call__(self, img_np_nchw):
#         '''
#         TensorRT推理
#         :param img_np_nchw: 输入图像
#         '''
#         self.ctx.push()
#
#         # Restore
#         stream = self.stream
#         context = self.context
#         engine = self.engine
#         host_inputs = self.host_inputs
#         cuda_inputs = self.cuda_inputs
#         host_outputs = self.host_outputs
#         cuda_outputs = self.cuda_outputs
#         bindings = self.bindings
#         # gpu无法直接访问内存数据，需要临时创建pagelock内存。这里相当于提前将pagelock内存创建好了。
#         np.copyto(host_inputs[0], img_np_nchw.ravel()) #将数据从内存拷贝到pagelock内存
#         cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream) #将数据从pagelock内存里进一步拷贝到cuda显存中
#         context.execute_async_v2(bindings=bindings, stream_handle=stream.handle) #tensorRT异步并发推理
#         cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream) #将数据从显存传输到pagelock内存
#         stream.synchronize()
#         self.ctx.pop()
#         return host_outputs[0]
#
#     def destroy(self):
#         # Remove any context from the top of the context stack, deactivating it.
#         self.ctx.pop()


class TrtModel():
    '''
    TensorRT infer
    '''

    def __init__(self, trt_path, device=1):
        self.cfx = cuda.Device(device).make_context()
        self.stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        # 启动一个tensorRT pyton runtime
        runtime = trt.Runtime(TRT_LOGGER)
        # 反序列化模型，使用runtime加载模型
        with open(trt_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        # 创建执行上下文
        self.context = self.engine.create_execution_context()
        # 模型输入的尺寸
        intype = self.engine.get_binding_dtype("input")
        insize = trt.volume(self.engine.get_binding_shape("input"))
        # fp16 占2个字节 fp32 占4个字节
        insize = insize * 2 if intype == DataType.HALF else insize * 4
        # 模型输出的尺寸
        otype = self.engine.get_binding_dtype("output")
        osize = trt.volume(self.engine.get_binding_shape("output"))
        osize = osize * 2 if otype == DataType.HALF else osize * 4
        otype = np.float16 if otype == DataType.HALF else np.float32
        # 分配输入输出的显存
        self.cuda_mem_input = cuda.mem_alloc(insize)
        self.cuda_mem_output = cuda.mem_alloc(osize)
        self.bindings = [int(self.cuda_mem_input), int(self.cuda_mem_output)]
        self.output = np.empty(self.engine.get_binding_shape("output"), dtype=otype)


    def __call__(self, img_np_nchw):
        '''
        TensorRT推理
        :param img_np_nchw: 输入图像
        '''
        self.cfx.push()
        #将数据从内存拷贝到显存中
        cuda.memcpy_htod_async(self.cuda_mem_input, img_np_nchw.ravel(), self.stream)
        #tensorRT异步并发推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        #将数据从显存传输到内存
        cuda.memcpy_dtoh_async(self.output, self.cuda_mem_output, self.stream)
        # 等待所有cuda核完成计算
        self.stream.synchronize()
        self.cfx.pop()
        return self.output

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()