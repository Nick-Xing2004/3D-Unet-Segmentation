import torch
import torch.nn as nn
from model_2 import initialize_Unet3D_2  # 替换为你的模型定义所在的模块

# Step 1: 实例化模型
device = "cpu"
model = initialize_Unet3D_2(device)
model = model.cpu()
model.eval()

# Step 2: 创建一个假输入（例如 1×1×128×128×128）
dummy_input = torch.randn(1, 1, 128, 128, 128)  # (batch_size, channels, depth, height, width)


# Step 3: 导出为 ONNX
torch.onnx.export(
    model,
    dummy_input,
    "unet3d.onnx",
    verbose=True,
    input_names=["input"],
    output_names=["output"],
    opset_version=11  # Netron 建议使用 11 或更高
)

print("Exported UNet3D to unet3d.onnx")
