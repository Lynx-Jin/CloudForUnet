import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.global_avg_pool = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 全局平均池化的替代
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv3_rate1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv3_rate3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.conv3_rate5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=5, dilation=5)
        self.concat_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        global_pool = torch.mean(x, dim=(2, 3), keepdim=True)  # 手动全局均值池化
        global_pool = self.global_avg_pool(global_pool)
        global_pool = F.interpolate(global_pool, size=x.shape[2:], mode='bilinear', align_corners=True)

        conv1 = self.conv1(x)
        conv3_rate1 = self.conv3_rate1(x)
        conv3_rate3 = self.conv3_rate3(x)
        conv3_rate5 = self.conv3_rate5(x)

        out = torch.cat([global_pool, conv1, conv3_rate1, conv3_rate3, conv3_rate5], dim=1)
        return self.relu(self.bn(self.concat_conv(out)))


class DilateBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dilated_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.aspp = ASPP(out_channels, out_channels)
        self.dilated_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.dilated_conv1(x)))
        x = self.aspp(x)
        x = self.relu(self.bn(self.dilated_conv2(x)))
        return x


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.depthwise_separable = DepthwiseSeparableConv(in_channels, in_channels)
        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca
        # 空间注意力
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_pool, max_pool], dim=1)
        sa = self.spatial_attention(sa_input)
        return x * sa

# 5. 改进 U-Net 模型（解码器使用两个 3x3 卷积）
class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # 编码器（Encoder）
        self.enc1 = DilateBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DilateBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DilateBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DilateBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DilateBlock(512, 1024)

        # 跳跃连接层
        self.skip1 = CBAM(64)
        self.skip2 = CBAM(128)
        self.skip3 = CBAM(256)
        self.skip4 = CBAM(512)

        # 解码器（每个 Up-Block 包括两个 3x3 卷积）
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
            nn.Conv2d(1024, 512, kernel_size=1)
        )
        self.dec4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=1)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=1)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=1)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # 输出层
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 编码
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # 跳跃连接处理
        skip4 = self.skip4(enc4)
        skip3 = self.skip3(enc3)
        skip2 = self.skip2(enc2)
        skip1 = self.skip1(enc1)

        # 解码
        dec4 = self.dec4(torch.cat([self.up4(bottleneck), skip4], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), skip3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), skip2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), skip1], dim=1))

        # 输出
        return self.sigmoid(self.out_conv(dec1))


# 模型测试
if __name__ == "__main__":
    model = ImprovedUNet(in_channels=3, out_channels=1)
    input_tensor = torch.randn((1, 3, 128, 128))  # 输入尺寸为 128x128 的 RGB 图像
    output = model(input_tensor)
    print("Output shape:", output.shape)  # 输出应为 (1, 1, 128, 128)
