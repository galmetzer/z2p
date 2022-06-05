import torch
import torch.nn.functional as F
from torch import nn

W_SIZE = 512


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adain(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat[0].size()[:2]) and (
                content_feat.size()[:2] == style_feat[1].size()[:2])
    size = content_feat.size()
    style_mean, style_std = style_feat
    style_mean, style_std = style_mean.unsqueeze(-1).unsqueeze(-1), style_std.unsqueeze(-1).unsqueeze(-1)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class FullyConnected(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, layers=3):
        super(FullyConnected, self).__init__()
        self.channels = torch.linspace(input_channels, output_channels, layers + 1).long()
        self.layers = nn.Sequential(
            *[nn.Linear(self.channels[i].item(), self.channels[i + 1].item()) for i in range(len(self.channels) - 1)]
        )

    def forward(self, x):
        return self.layers(x)


class Affine(nn.Module):
    def __init__(self, input_channels: int, output_channels):
        super(Affine, self).__init__()
        self.lin = nn.Linear(input_channels, output_channels)
        bias = torch.zeros(output_channels)
        nn.init.normal_(bias, 0, 1)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        return self.lin(x) + self.bias


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, ada=False, padding='zeros'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.ada = ada
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode=padding)
        if ada:
            self.a1_mean = Affine(W_SIZE, mid_channels)
            self.a1_std = Affine(W_SIZE, mid_channels)
        else:
            self.norm1 = nn.InstanceNorm2d(mid_channels, affine=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding)

        if ada:
            self.a2_mean = Affine(W_SIZE, out_channels)
            self.a2_std = Affine(W_SIZE, out_channels)
        else:
            self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x, w=None):
        if self.ada:
            assert w is not None

        x = self.conv1(x)

        if self.ada:
            x = adain(x, (self.a1_mean(w), self.a1_std(w)))
        else:
            x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.ada:
            x = adain(x, (self.a2_mean(w), self.a2_std(w)))
        else:
            x = self.norm2(x)
        x = self.relu(x)

        return x


class DiluteConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, dilation, padding='zeros'):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=1 + dilation, dilation=dilation, padding_mode=padding)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x, y=None):
        if y is not None:
            x = torch.cat([x, y], dim=1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, ada=False, padding='zeros'):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.double_conv = DoubleConv(in_channels, out_channels, ada=ada, padding=padding)

    def forward(self, x, w=None):
        x = self.max_pool(x)
        return self.double_conv(x, w)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, ada=False, padding='zeros'):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, ada=ada)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, ada=ada)

    def forward(self, x1, x2, w=None):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, w)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding='zeros'):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding_mode=padding)

    def forward(self, x):
        return self.conv(x)
