import torch
import torch.nn as nn
import torch.nn.functional as F
from ._layers import LoadMixin, RRDB


class RRDBNet(nn.Module, LoadMixin):
    # WEIGHTS_FILENAME = "bsrgan_x4_enhancer.pth"

    WEIGHTS_FILENAME = "BSRGAN.pth"
    URL_ROOT = "https://github.com/cszn/KAIR/releases/download/v1.0/"

    def __init__(self, thr: float = 0.001):
        super().__init__()
        # Initialize first layers that produce features
        self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
        self.RRDB_trunk = nn.Sequential(*[RRDB(64, 32) for _ in range(23)])
        self.trunk_conv = nn.Conv2d(64, 64, 3, 1, 1)
        self.upconv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.upconv2 = nn.Conv2d(64, 64, 3, 1, 1)

        # Final layers that produce enhanced image
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fea = (x := self.conv_first(x)) + self.trunk_conv(self.RRDB_trunk(x))
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2)))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2)))

        return self.conv_last(self.lrelu(self.HRconv(fea)))

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        upscaled = self(images.div(255))
        enhanced = F.interpolate(upscaled, scale_factor=0.25, mode="bicubic")
        images = enhanced.clamp(0, 1).mul(255).round()

        return images
