import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_mlir.fx import export_and_import
from model import ChessNet

model = ChessNet()
model.load_state_dict(torch.load("ChessNetv1.pth", weights_only=True))
model.eval()

input = torch.rand(1, 3, 64, 64)

mlir_module = export_and_import(model, input, output_type="linalg-on-tensors")
mlir_str = mlir_module.operation.get_asm()

# Save to a file
with open("build/chessnet.mlir", "w") as f:
    f.write(mlir_str)

print("MLIR module saved to chessnet.mlir")

