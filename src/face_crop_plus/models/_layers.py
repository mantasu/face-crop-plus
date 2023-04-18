import torch
import torch.nn as nn
import torch.nn.functional as F

from math import ceil
from itertools import product

########################################################################
############################### GENERAL ################################
########################################################################

class LoadMixin():
    URL_ROOT = "https://github.com/mantasu/face-crop-plus/releases/download/v1.0.0/"
    WEIGHTS_FILENAME = None

    def load(self, device: str | torch.device = "cpu"):
        weights = self.get_weights(device)
        self.load_state_dict(weights)
        self.to(device)
        self.eval()
        
        for param in self.parameters():
            param.requires_grad = False
        
        return self
    
    def get_weights(self, device):
        if self.WEIGHTS_FILENAME is None:
            raise ValueError(f"Please ensure 'WEIGHTS_FILENAME' is specified "
                              "for the class that inherits this mixin.")
        
        url = self.URL_ROOT + self.WEIGHTS_FILENAME
        weights = torch.hub.load_state_dict_from_url(url, map_location=device)

        return weights

########################################################################
######################### RETINA FACE MODULES ##########################
########################################################################

class PriorBox():
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.steps = [8, 16, 32]
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.features = [tuple(ceil(x / n) for x in size) for n in self.steps]

    def forward(self):
        anchors = []
        
        for k, f in enumerate(self.features):
            min_sizes = self.min_sizes[k]

            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    a, b = min_size / self.size[1], min_size / self.size[0]
                    dx = [x * self.steps[k] / self.size[1] for x in [j + 0.5]]
                    dy = [y * self.steps[k] / self.size[0] for y in [i + 0.5]]
                    anchors.extend([(x, y, a, b) for y, x in product(dy, dx)])
        
        return torch.tensor(anchors).view(-1, 4)

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        leaky = 0.1 if out_channel <= 64 else 0
        out_dim = out_channel // 4

        self.conv3X3 = self.conv_bn_no_relu(in_channel, out_channel // 2)
        self.conv5X5_1 = self.conv_bn(in_channel, out_dim, leaky = leaky)
        self.conv5X5_2 = self.conv_bn_no_relu(out_dim, out_dim)
        self.conv7X7_2 = self.conv_bn(out_dim, out_dim, leaky = leaky)
        self.conv7x7_3 = self.conv_bn_no_relu(out_dim, out_dim)
    
    def conv_bn(self, inp, oup, stride=1, leaky=0):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(negative_slope=leaky, inplace=True)
        )

    def conv_bn_no_relu(self, inp, oup, stride=1):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, input):
        conv3X3 = self.conv3X3(input)
        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)
        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        return F.relu(torch.cat([conv3X3, conv5X5, conv7X7], dim=1))

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()

        leaky = 0.1 if out_channels <= 64 else 0
        out = out_channels

        self.output1 = self.conv_bn1X1(in_channels_list[0], out, leaky = leaky)
        self.output2 = self.conv_bn1X1(in_channels_list[1], out, leaky = leaky)
        self.output3 = self.conv_bn1X1(in_channels_list[2], out, leaky = leaky)

        self.merge1 = self.conv_bn(out, out, leaky = leaky)
        self.merge2 = self.conv_bn(out, out, leaky = leaky)
    
    def conv_bn(self, inp, oup, stride = 1, leaky = 0):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(negative_slope=leaky, inplace=True)
        )
    
    def conv_bn1X1(self, inp, oup, stride=1, leaky=0):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(negative_slope=leaky, inplace=True)
        )

    def forward(self, input):
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        size_up3 = [output2.size(2), output2.size(3)]
        size_up2 = [output1.size(2), output1.size(3)]

        up3 = F.interpolate(output3, size=size_up3, mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=size_up2, mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        return [output1, output2, output3]

class Head(nn.Module):
    def __init__(self, num_out, in_channels=512):
        super().__init__()
        self.num_out = num_out
        self.conv1x1 = nn.Conv2d(in_channels, 2 * num_out, 1)
    
    def forward(self, x):
        x = self.conv1x1(x)
        out = x.permute(0, 2, 3, 1).contiguous()
        
        return out.view(out.size(0), -1, self.num_out)
    
    @classmethod
    def make(cls, num_out, in_channels=512):
        head = nn.ModuleList([cls(num_out, in_channels) for _ in range(3)])
        return head

########################################################################
############################# RRDB MODULES #############################
########################################################################

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super().__init__()

        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)

        return out * 0.2 + x

########################################################################
############################# BISE MODULES #############################
########################################################################

class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super().__init__()

        self.conv1 = self.conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = self.conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None

        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
            )
    
    def conv3x3(self, in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        shortcut = x

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)

        return out

class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = self.create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = self.create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = self.create_layer_basic(256, 512, bnum=2, stride=2)
    
    def create_layer_basic(self, in_chan, out_chan, bnum, stride=1):
        layers = [BasicBlock(in_chan, out_chan, stride=stride)]

        for _ in range(bnum-1):
            layers.append(BasicBlock(out_chan, out_chan, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)

        return feat8, feat16, feat32

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, ks, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))

        return x

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, 1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)

        return x

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)

        return out

class ContextPath(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32))

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16))
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8))
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, 1, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat

        return feat_out
