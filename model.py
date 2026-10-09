import torch
import torch.nn as nn
import torch.nn.functional as F


class BlindSpotConv1d(nn.Conv1d):
    """
    带有物理掩码的自定义 1D 卷积层，强行阻断中心信息的传递
    """

    def __init__(self, in_channels, out_channels, kernel_size, blind_spot_width=3, **kwargs):
        assert kernel_size % 2 == 1, "Kernel_size 必须为奇数"
        assert blind_spot_width % 2 == 1, "Blind_spot_width 必须为奇数"

        # 使用 SAME padding 保持输入输出长度一致
        padding = kernel_size // 2
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        # 注册一个不可训练的掩码矩阵
        self.register_buffer('mask', torch.ones_like(self.weight))

        # 计算盲区范围并将掩码中心区域置零
        center = kernel_size // 2
        half_blind = blind_spot_width // 2
        self.mask[:, :, center - half_blind: center + half_blind + 1] = 0.0

    def forward(self, x):
        # 每次前向传播时，强行将中心区域的权重归零
        masked_weight = self.weight * self.mask
        return F.conv1d(
            x, masked_weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups
        )


class SpectralBSN(nn.Module):
    """
    专为痕量拉曼光谱去噪定制的盲点网络
    """

    def __init__(self):
        super().__init__()

        # 第一层：大感受野提取物理特征，中心存在 3 个像素的盲点
        self.feature_extractor = BlindSpotConv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=31,
            blind_spot_width=3
        )
        self.act1 = nn.LeakyReLU(0.01)

        # 后续层：1x1 卷积（Pointwise Conv），绝对不扩大感受野，防止信息泄露
        self.regressor = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 32, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 1, kernel_size=1)

        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.act1(x)
        y_pred = self.regressor(x)
        return y_pred

