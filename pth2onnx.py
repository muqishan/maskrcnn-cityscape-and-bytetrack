from torch.autograd import Variable
import torch
import onnx
import onnxsim
from model.mark_R_CNN import MaskRcnnResnet50Fpn

##-------------------------------densenet121------------------------------#
input_model = 'net/Epoch_010_Box0.7685_Segm0.7816.pth'# 待转换模型
output_model_ = input_model.replace('.pth', '.onnx') #转换后模型输出
# IMAGE_SIZE = 64   # 输入形状
print("=====> load pytorch checkpoint...")

net = MaskRcnnResnet50Fpn()

checkpoint = torch.load(input_model, map_location=torch.device('cpu'))

net.load_state_dict(checkpoint)
print("=====> convert pytorch model to onnx...")
dummy_input = Variable(torch.randn(1, 3, 1024, 1024))
input_names = ["input"]
output_names = ["output"]
torch.onnx.export(net,
                  dummy_input,
                  output_model_,
                  verbose=True,
                  input_names=input_names,
                  output_names=output_names)

print("====> check onnx model...")

model = onnx.load(output_model_)
onnx.checker.check_model(model)

print("====> Simplifying...")
model_opt = onnxsim.simplify(output_model_)

model_opt, check = onnxsim.simplify(output_model_)
print("onnx model simplify Ok!")
