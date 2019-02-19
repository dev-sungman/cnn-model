import torch
import torch.nn as nn
import torch.nn.functional as F
from spp_layer import spatial_pyramid_pool

class SPPNet(nn.Module):
    def __init__(self, opt, input_nc, ndf=64, gpu_ids=[]):
        super(SPPNet, self).__init__()
        self.gpu_ids = gpu_ids
        self.output_num = [4, 2, 1]
        
        # ZF-5
        self.features = nn.Sequentail(
                nn.Conv2d(input_nc, 96, filter_size=7, stride=2, bias=False),
                nn.ReLU(inplace=True),
                LRN(local_size=3, alpha=0.0001, beta=0.75),
                nn.Conv2d(96, 256, filter_size=5, stride=2, bias=False),
                nn.ReLU(inplace=True),
                LRN(local_size=3, alpha=0.0001, beta=0.75),
                nn.Conv2d(256, 384, filter_size=3, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 384, filter_size=3, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 384, filter_size=3, bias=False),
                nn.ReLU(inplace=True),
                )


    def forward(self, x):
        x = self.features(x)
        x = self.ReLU(x)
