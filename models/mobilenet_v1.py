import torch
import torch.nn as nn

from models.conv_type import SFPConv2d


def conv1x1(in_planes, out_planes, stride=1):
    return SFPConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        conv1x1(inp, oup),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


class MobileNet(nn.Module):
    def __init__(self, num_classes):
        super(MobileNet, self).__init__()

        in_planes = 32
        cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

        self.conv1 = conv_bn(3, in_planes, stride=2)

        self.features = self._make_layers(in_planes, cfg, conv_dw)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc_mask = nn.Linear(cfg[-1], num_classes)
        self.fc_no_mask = nn.Linear(cfg[-1], num_classes)

    def forward(self, x, no_mask=False):
        x = self.conv1(x)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        
        if no_mask:
            return self.fc_no_mask(x)
        x = self.fc_mask(x)

        return x

    def _make_layers(self, in_planes, cfg, layer):
        layers = []
        for x in cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(layer(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)


def mobilenet_v1(pretrained=False, num_classes=1000):
    if pretrained:
        pass
    return MobileNet(num_classes=num_classes)


if __name__ == "__main__":
    model = mobilenet_v1(num_classes=1000)
    data = torch.randn(1, 3, 224, 224)
    print(model(data).shape)
