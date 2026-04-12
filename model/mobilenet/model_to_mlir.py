import os
import torch
import re
from torch.func import functional_call
from torch_mlir.fx import export_and_import

os.makedirs("build", exist_ok=True)

model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_mobilenetv2_x1_0", pretrained=True)
model.eval()
model.requires_grad_(False)

for module in model.modules():
    if hasattr(module, 'num_batches_tracked'):
        delattr(module, 'num_batches_tracked')

state_dict = dict(model.named_parameters())
state_dict.update(dict(model.named_buffers()))
weight_keys = list(state_dict.keys())
weight_values = list(state_dict.values())

# Dynamically generate Explicit Wrapper
arg_names = [f"w{i}" for i in range(len(weight_keys))] + ["x"]
signature = ", ".join(arg_names)

class_code = f"""
import torch
class ExplicitWrapper(torch.nn.Module):
    def forward(self, {signature}):
        args = [{signature}]
        w_dict = {{k: v for k, v in zip(weight_keys, args[:-1])}}
        x = args[-1]
        return functional_call(model, w_dict, (x,))
"""

local_vars = {"torch": torch, "weight_keys": weight_keys, "functional_call": functional_call, "model": model}
exec(class_code, local_vars)
ExplicitWrapper = local_vars["ExplicitWrapper"]

wrapper = ExplicitWrapper()
wrapper.eval()

dummy_input = torch.randn(1, 3, 32, 32)
all_inputs = tuple(weight_values) + (dummy_input,)

print("Exporting explicit stateless model via FX... (this takes a moment)")
with torch.no_grad():
    module = export_and_import(
        wrapper, 
        *all_inputs, 
        output_type="linalg-on-tensors", 
        func_name="forward"
    )

mlir_text = str(module)

# Inject the C-Interface flag
mlir_text = re.sub(
    r'(func\.func @forward[^\{]+)\{', 
    r'\1 attributes {llvm.emit_c_interface} {', 
    mlir_text, 
    count=1
)

with open("build/mobilenet.mlir", "w") as f:
    f.write(mlir_text)

# Generate the C++ Loader Boilerplate
cpp_header = "#pragma once\n#include <vector>\n#include <fstream>\n#include <iostream>\n#include <cstdint>\n\n"
cpp_header += "template <typename T, size_t N> struct MemRefDescriptor { T *allocated; T *aligned; intptr_t offset; intptr_t sizes[N]; intptr_t strides[N]; };\n\n"

num_weights = len(weight_values)

with open("build/weights.bin", "wb") as f_bin:
    cpp_header += "void load_weights(std::ifstream& bin_file, std::vector<void*>& memrefs, std::vector<std::vector<float>*>& vectors) {\n"
    
    for idx, tensor in enumerate(weight_values):
        data = tensor.detach().cpu().numpy()
        f_bin.write(data.tobytes())
        
        shape = data.shape if len(data.shape) > 0 else (1,)
        sizes = ", ".join(map(str, shape))
        strides = ", ".join(map(str, tensor.stride() if len(data.shape) > 0 else (1,)))
        
        cpp_header += f"  auto* vec_{idx} = new std::vector<float>({data.size});\n"
        cpp_header += f"  bin_file.read(reinterpret_cast<char*>(vec_{idx}->data()), {data.size} * sizeof(float));\n"
        cpp_header += f"  vectors.push_back(vec_{idx});\n"
        cpp_header += f"  auto* memref_{idx} = new MemRefDescriptor<float, {len(shape)}>{{vec_{idx}->data(), vec_{idx}->data(), 0, {{{sizes}}}, {{{strides}}}}};\n"
        cpp_header += f"  memrefs.push_back(memref_{idx});\n"
    cpp_header += "}\n\n"

cpp_header += 'extern "C" void _mlir_ciface_forward(MemRefDescriptor<float, 2>* out'
for i in range(num_weights):
    cpp_header += f', void* w{i}'
cpp_header += ', MemRefDescriptor<float, 4>* in);\n\n'

cpp_header += 'void run_forward(MemRefDescriptor<float, 2>* out, MemRefDescriptor<float, 4>* in, std::vector<void*>& weights) {\n'
cpp_header += '  _mlir_ciface_forward(out'
for i in range(num_weights):
    cpp_header += f', weights[{i}]'
cpp_header += ', in);\n}\n'

with open("build/weights.h", "w") as f_h:
    f_h.write(cpp_header)

print(f"Exported mobilenet.mlir, weights.bin, and weights.h ({num_weights} float32 weights correctly mapped)!")