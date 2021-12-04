import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
        # m.weight.data.fill_(1.0)


from torch.autograd import Variable

from torch.nn.utils import spectral_norm

"""
https://gist.github.com/rosinality/a96c559d84ef2b138e486acf27b5a56e
"""


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, stride, bn=True):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False)
        self.use_bn = bn
        if bn:
            self.bn = nn.BatchNorm2d(out_channel)
        init_conv(self.conv)
        self.conv = spectral_norm(self.conv)

    def forward(self, input):
        out = self.conv(input)
        if self.use_bn:
            out = self.bn(out)
        out = F.leaky_relu(out, negative_slope=0.2)

        return out


class NeuralNetwork(nn.Module):
    def __init__(self, in_channels=1, number_of_actions=4, bn_bool=False):
        super(NeuralNetwork, self).__init__()
        self.number_of_actions = number_of_actions
        self.conv = nn.Sequential(
            ConvBlock(in_channels, 32, [6, 6], 0, 3, bn=bn_bool),
            ConvBlock(32, 64, [4, 4], 0, 2, bn=bn_bool),
            ConvBlock(64, 64, [3, 3], 0, 1, bn=bn_bool),
            # ConvBlock(64, 64, [2, 2], 0, 1, bn=bn_bool),
        )
        self.fc = nn.Sequential(
            nn.Linear(10 * 10 * 64, 256),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(256, self.number_of_actions),
        )
        # self.fc.apply(init_weights)
        self.module = nn.ModuleList()
        self.module.append(self.conv)
        self.module.append(self.fc)
        # torch.nn.init.kaiming_normal_(self.fc4.weight, mode='fan_in', nonlinearity='leaky_relu')
        # torch.nn.init.kaiming_normal_(self.fc5.weight, mode='fan_in', nonlinearity='leaky_relu')
        # torch.nn.init.kaiming_normal_(self.fc6.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DiscretePolicyValue(nn.Module):
    def __init__(self, in_channels=1, number_of_actions=4, bn_bool=False):
        super(DiscretePolicyValue, self).__init__()
        self.number_of_actions = number_of_actions
        self.conv = nn.Sequential(
            ConvBlock(in_channels, 32, [6, 6], 0, 3, bn=bn_bool),
            ConvBlock(32, 64, [4, 4], 0, 2, bn=bn_bool),
            ConvBlock(64, 64, [3, 3], 0, 1, bn=bn_bool),
            # ConvBlock(64, 64, [2, 2], 0, 1, bn=bn_bool),
        )
        self.l = nn.Sequential(
            nn.Linear(10 * 10 * 64, 256),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
        )
        self.pi = nn.Linear(256, self.number_of_actions)
        self.v = nn.Linear(256, 1)
        self.module = nn.ModuleList()
        self.module.append(self.conv)
        self.module.append(self.l)
        self.module.append(self.pi)
        self.module.append(self.v)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.l(x)
        return F.softmax(self.pi(x), dim=-1), self.v(x)


cfg = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
