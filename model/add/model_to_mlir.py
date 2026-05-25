import torch
from torch_mlir import torchscript

class AddModel(torch.nn.Module):
    def forward(self, a, b):
        return a + b

model = AddModel()
a = torch.ones(2, 2, dtype=torch.int32)
b = torch.ones(2, 2, dtype=torch.int32)

module_torch = torchscript.compile(
    model, 
    (a, b), 
    output_type="torch"
)
with open("./build/add_torch.mlir", "w") as f:
    f.write(str(module_torch))

module_linalg = torchscript.compile(
    model, 
    (a, b), 
    output_type="linalg-on-tensors"
)
with open("./build/add_linalg.mlir", "w") as f:
    f.write(str(module_linalg))

print("Done")