import torch
from torch import nn
from torchsummary import summary


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_1conv=False, stride=1):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        # 第一个卷积：bias=False 因为后面有 BN
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二个卷积
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # shortcut 分支
        if use_1conv:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        # 初始卷积层
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 四个残差阶段（每个阶段两个 BasicBlock）
        self.block2 = nn.Sequential(
            ResidualBlock(64, 64, use_1conv=False, stride=1),
            ResidualBlock(64, 64, use_1conv=False, stride=1),
        )
        self.block3 = nn.Sequential(
            ResidualBlock(64, 128, use_1conv=True, stride=2),
            ResidualBlock(128, 128, use_1conv=False, stride=1),
        )
        self.block4 = nn.Sequential(
            ResidualBlock(128, 256, use_1conv=True, stride=2),
            ResidualBlock(256, 256, use_1conv=False, stride=1),
        )
        self.block5 = nn.Sequential(
            ResidualBlock(256, 512, use_1conv=True, stride=2),
            ResidualBlock(512, 512, use_1conv=False, stride=1),
        )

        self.block6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

        # 可选权重初始化（原版使用 Kaiming 正态分布）
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x


# 测试
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(in_channels=3, num_classes=2).to(device)
    dummy = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy)
    print(output.shape)  # 应为 [1, 2]
    summary(model, (3, 224, 224))
