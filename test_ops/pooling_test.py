import torch
import torch.nn as nn

x = torch.randn(1,3,3,3)
print(x)

x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
print(x)

