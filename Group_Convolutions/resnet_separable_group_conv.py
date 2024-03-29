import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable

model_urls = {
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}
number_of_groups=4

class DepthwiseConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=number_of_groups, bias=True, padding_mode='zeros', device=None, dtype=None) -> None:
        super().__init__()

        self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups,
                                        bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.depthwise_conv(x)

        return x


class PointwiseConv2D(nn.Module):
    def __init__(self,in_channels, out_channels, bias=True, device=None, dtype=None, groups=1) -> None:
        super().__init__()

        self.pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1, padding=0,
                                        dilation=1, groups=groups, bias=bias, padding_mode='zeros', device=device, dtype=dtype)
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pointwise_conv(x)

        return x


class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros', device=None, dtype=None) -> None:
        super().__init__()
        print("This is the number of groups TESTING :", number_of_groups)
        self.depthwise_conv = DepthwiseConv2D(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
        groups=number_of_groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)

        self.pointwise_conv = PointwiseConv2D(in_channels=in_channels, out_channels=out_channels, bias=bias, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        return x

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return SeparableConv2D(in_channels=in_planes, out_channels=out_planes, kernel_size=3,stride=stride, padding=1, bias=False)
    #return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=number_of_groups)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        # cfg should be a number in this case
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, cfg, stride)
        self.bn1 = nn.BatchNorm2d(cfg)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = SeparableConv2D(in_channels=inplanes, out_channels=planes, kernel_size=1, bias=False)
        #self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, groups=number_of_groups)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SeparableConv2D(in_channels=planes, out_channels=planes, kernel_size=3,stride=stride, padding=1, bias=False)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=number_of_groups)
        self.bn2 = nn.BatchNorm2d(planes)
        #self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False, groups=number_of_groups)
        self.conv3 = SeparableConv2D(in_channels=planes, out_channels=planes*4, kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, cfg=None, num_classes=10):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if cfg == None:
            cfg = [[64] * layers[0], [128]*layers[1], [256]*layers[2], [512]*layers[3]]
            cfg = [item for sub_list in cfg for item in sub_list]
        # Don't put groups here because this is the input shape convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        count = 0
        self.layer1 = self._make_layer(block, 64, layers[0], cfg[:layers[0]])
        count += layers[0]
        self.layer2 = self._make_layer(block, 128, layers[1], cfg[count:count+layers[1]], stride=2)
        count += layers[1]
        self.layer3 = self._make_layer(block, 256, layers[2], cfg[count:count+layers[2]], stride=2)
        count += layers[2]
        self.layer4 = self._make_layer(block, 512, layers[3], cfg[count:count+layers[3]], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(cfg[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SeparableConv2D(in_channels=self.inplanes, out_channels=planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                #nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, groups=number_of_groups),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0],stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet34(pretrained=False,groups=2, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    number_of_groups= groups
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

