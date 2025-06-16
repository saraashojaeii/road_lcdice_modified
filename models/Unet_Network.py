import torch
import torch.nn as nn
import torch.nn.functional as F
from semseg_utils import *
from unet_utils import *



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv = self.conv_block(x)
        return self.pool(conv), conv

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(mid_channels + out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Resize x to match skip connection
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv_block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels_mask=2):  # out_channels_mask=2 for binary classification
        super(UNet, self).__init__()
        self.encoder1 = EncoderBlock(in_channels, 32)
        self.encoder2 = EncoderBlock(32, 64)
        self.encoder3 = EncoderBlock(64, 128)
        self.center = ConvBlock(128, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 32, 32)
        self.out_conv_mask = nn.Conv2d(32, out_channels_mask, kernel_size=1)  # Change to 2 channels

    def forward(self, x):
        e1_pool, e1 = self.encoder1(x)
        e2_pool, e2 = self.encoder2(e1_pool)
        e3_pool, e3 = self.encoder3(e2_pool)
        center = self.center(e3_pool)
        d3 = self.decoder3(center, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)
        mask_output = self.out_conv_mask(d1)  # No sigmoid, output raw logits
        return mask_output

