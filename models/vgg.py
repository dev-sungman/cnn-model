'''VGG in Pytorch'''
import torch
import torch.nn as nn
from lrn import LRN
'''
class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.classifier = nn.Sequential(
            nn.Linear(512,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,num_classes),
            )
        
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), 512)
        out = self.classifier(out)
        return out
'''

def conv3x3(inplanes, outplanes, stride=1):
    return nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(inplanes, outplanes, stride=1):
    return nn.conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=1, bias=False)

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifiers = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0),-1)
        out = self.classifiers(out)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    layers = []
    inchannels = 3
    print(cfg)
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'L':
            layers += [LRN(local_size=5, alpha=0.0001, beta=0.75)]
        else:
            f = int(v.split('_')[0])
            s = int(v.split('_')[1])
            if batch_norm:
                if f is 1:
                    layers += [conv1x1(inchannels, s), nn.BatchNorm2d(s), nn.ReLU(inplace = True)]
                else:
                    layers += [conv3x3(inchannels, s), nn.BatchNorm2d(s), nn.ReLU(inplace=True)]
            else:
                if f is 1:
                    layers += [conv1x1(inchannels, s), nn.ReLU(inplace = True)]
                else:
                    layers += [conv3x3(inchannels, s), nn.ReLU(inplace=True)]
            inchannels = s
    return nn.Sequential(*layers)

cfg = {
    'A': ['3_64', 'M', '3_128', 'M', '3_256', '3_256', 'M', '3_512', '3_512', 'M', '3_512', '3_512', 'M'],
    'A_LRN': ['3_64', 'L', 'M', '3_128', 'M', '3_256', '3_256', 'M', '3_512', '3_512', 'M', '3_512', '3_512', 'M'],
    'B': ['3_64', '3_64', 'M', '3_128', '3_128', 'M', '3_256', '3_256', 'M', '3_512', '3_512', 'M', '3_512', '3_512', 'M'],
    'C': ['3_64', '3_64', 'M', '3_128', '3_128', 'M', '3_256', '3_256', '1_256', 'M', '3_512', '3_512', '1_512', 'M', '3_512', '3_512', '1_512', 'M'],
    'D': ['3_64', '3_64', 'M', '3_128', '3_128', 'M', '3_256', '3_256', '3_256', 'M', '3_512', '3_512', '3_512', 'M', '3_512', '3_512', '3_512', 'M'],
    'E': ['3_64', '3_64', 'M', '3_128', '3_128', 'M', '3_256', '3_256', '3_256', '3_256', 'M', '3_512', '3_512', '3_512', '3_512', 'M', '3_512', '3_512', '3_512', '3_512', 'M'],
    }

def vgg11(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model

def vgg11_bn(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model

def vgg11_lrn(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A_LRN']), **kwargs)
    return model

def vgg11_lrn_bn(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A_LRN'], batch_norm=True), **kwargs)
    return model

def vgg13(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model

def vgg13_bn(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_nrom=True), **kwargs)
    return model

def vgg16(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model

def vgg16_bn(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model

def vgg16_1x1(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['C']), **kwargs)
    return model

def vgg19(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model

def vgg19_bn(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model

def test():
    #net = VGG16()
    net = vgg11_lrn()
    x = torch.randn(2,3,224,224)
    y = net(x)
    print(net)
    print(y.shape)

test()
