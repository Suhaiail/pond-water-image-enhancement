# model.py — PondNet GAN
# Generator:     ResidualUNet + CBAM attention + SE bottleneck
# Discriminator: PatchGAN with Spectral Normalisation (70x70 patches)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
# ATTENTION MODULES
# ══════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    """CBAM Channel Attention — recalibrates which channels matter."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.fc(self.avg_pool(x)) +
                                self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    """CBAM Spatial Attention — focuses on where (fish vs water)."""
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg  = x.mean(dim=1, keepdim=True)
        mx,_ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Full CBAM: Channel then Spatial attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class SEBlock(nn.Module):
    """Squeeze-Excitation for bottleneck feature recalibration."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


# ══════════════════════════════════════════════════════════════
# GENERATOR BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = ConvBlock(c_in, c_out)
        self.cbam = CBAM(c_out)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        f = self.cbam(self.conv(x))
        return f, self.pool(f)


class Up(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up   = nn.ConvTranspose2d(c_in, c_in // 2, 2, stride=2)
        self.conv = ConvBlock(c_in // 2 + c_skip, c_out)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            dy = skip.size(-2) - x.size(-2)
            dx = skip.size(-1) - x.size(-1)
            x  = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.conv(torch.cat([skip, x], dim=1))


# ══════════════════════════════════════════════════════════════
# GENERATOR — ResidualUNet + CBAM + SE
# ══════════════════════════════════════════════════════════════

class ResidualUNet(nn.Module):
    """
    PondNet Generator.
    Learns delta: output = clamp(input + delta, 0, 1)
    This allows output to EXCEED the target — not capped by L1 alone.
    CBAM focuses on fish regions. SE recalibrates bottleneck features.
    """
    def __init__(self, base=48):
        super().__init__()
        # Encoder
        self.d1  = Down(3,        base)
        self.d2  = Down(base,     base*2)
        self.d3  = Down(base*2,   base*4)
        self.d4  = Down(base*4,   base*8)
        # Bottleneck
        self.mid = ConvBlock(base*8, base*16)
        self.se  = SEBlock(base*16)
        # Decoder
        self.u4  = Up(base*16, base*8,  base*8)
        self.u3  = Up(base*8,  base*4,  base*4)
        self.u2  = Up(base*4,  base*2,  base*2)
        self.u1  = Up(base*2,  base,    base)
        # Output
        self.out = nn.Conv2d(base, 3, 1)

    def forward(self, x_in):
        s1, p1 = self.d1(x_in)
        s2, p2 = self.d2(p1)
        s3, p3 = self.d3(p2)
        s4, p4 = self.d4(p3)
        m      = self.se(self.mid(p4))
        x      = self.u4(m,  s4)
        x      = self.u3(x,  s3)
        x      = self.u2(x,  s2)
        x      = self.u1(x,  s1)
        delta  = self.out(x)
        return torch.clamp(x_in + delta, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════
# DISCRIMINATOR — PatchGAN with Spectral Normalisation
# ══════════════════════════════════════════════════════════════

class PatchDiscriminator(nn.Module):
    """
    70x70 PatchGAN discriminator (Isola et al., Pix2Pix CVPR 2017).
    Spectral normalisation prevents discriminator overfitting on small datasets.
    Takes [input | enhanced] concatenated as 6-channel input.
    Output: 30x30 grid of real/fake scores for 256x256 input.
    """
    def __init__(self):
        super().__init__()
        sn = nn.utils.spectral_norm

        def block(c_in, c_out, stride, norm=True):
            layers = [sn(nn.Conv2d(c_in, c_out, 4, stride, 1, bias=False))]
            if norm:
                layers.append(nn.InstanceNorm2d(c_out, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            # No norm on first layer (standard practice)
            *block(6,   64,  stride=2, norm=False),   # 256→128
            *block(64,  128, stride=2, norm=True),    # 128→64
            *block(128, 256, stride=2, norm=True),    # 64→32
            *block(256, 512, stride=1, norm=True),    # 32→31
            sn(nn.Conv2d(512, 1, 4, stride=1, padding=1))  # 31→30
        )

    def forward(self, x, y):
        """x = input image, y = enhanced/target image."""
        return self.net(torch.cat([x, y], dim=1))
    