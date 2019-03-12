''' AlexNet in Pytorch'''
import torch
import torch.nn as nn
from lrn import LRN

class AlexNet(nn.Module):
    def __init__(self, num_classes = 1000):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
                nn.ReLU(inplace=True),
                LRN(local_size=5, alpha=0.0001, beta=0.75),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
                nn.ReLU(inplace=True),
                LRN(local_size=5, alpha=0.0001, beta=0.75),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(256,384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384,384, kernel_size=3, padding=1, groups=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(384,256, kernel_size=3, padding=1, groups=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                )

        self.classifier = nn.Sequential(
                nn.Linear(256 * 5 * 5, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
                )
                
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 5 * 5)
        x = self.classifier(x)

        return x

x = torch.randn(1,3,32,32)
net = AlexNet()
y = net(x)
print(x)

