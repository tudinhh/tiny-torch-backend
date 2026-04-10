import torch
import torchvision
import torchvision.transforms as transforms
import struct
from torch_mlir.fx import export_and_import

model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_mobilenetv2_x1_0", pretrained=True)
model.eval()

input = torch.randn(1, 3, 32, 32)

module = export_and_import(
    model, 
    input, 
    output_type="linalg-on-tensors"
)

with open("build/cifar10_mobilenet.mlir", "w") as f:
    f.write(str(module))
print("Model exported to cifar10_mobilenet.mlir")

# 3. Fetch CIFAR-10 and Export to Binary for C++
# Using the exact normalization stats required by this specific pre-trained model
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=1, shuffle=False)

num_images_to_export = 100
exported = 0

with open("build/cifar10_test.bin", "wb") as f:
    for images, labels in test_loader:
        if exported >= num_images_to_export:
            break
        f.write(struct.pack('i', labels[0].item()))
        f.write(images[0].numpy().tobytes())
        exported += 1

print(f"Exported {exported} CIFAR-10 images to cifar10_test.bin")