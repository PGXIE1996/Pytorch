import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class InceptionV1(nn.Module):
    """Inception v1 模块"""

    def __init__(
        self, in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj
    ):
        super(InceptionV1, self).__init__()
        # 1x1 分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1), nn.ReLU(inplace=True)
        )
        # 3x3 分支：1x1 降维 + 3x3 卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # 5x5 分支：1x1 降维 + 5x5 卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        # 池化分支：3x3 池化 + 1x1 卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        # 在通道维度上拼接
        return torch.cat([out1, out2, out3, out4], dim=1)


class GoogLeNet(nn.Module):
    """经典 GoogLeNet (Inception v1) 完整网络"""

    def __init__(self, num_classes=1000, aux_logits=True):
        super().__init__()
        self.aux_logits = aux_logits

        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception 模块（3a, 3b）
        self.inception3a = InceptionV1(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionV1(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception 模块（4a, 4b, 4c, 4d, 4e）
        self.inception4a = InceptionV1(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionV1(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionV1(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionV1(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionV1(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception 模块（5a, 5b）
        self.inception5a = InceptionV1(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionV1(832, 384, 192, 384, 48, 128, 128)

        # 全局平均池化 + Dropout + 全连接
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        # 辅助分类器（auxiliary classifiers）
        if self.aux_logits:
            self.aux1 = self._make_aux(512, num_classes)  # 接在 inception4a 之后
            self.aux2 = self._make_aux(528, num_classes)  # 接在 inception4d 之后
        else:
            self.aux1 = self.aux2 = None

        # 权重初始化
        self._initialize_weights()

    def _make_aux(self, in_channels, num_classes):
        """构建辅助分类器"""
        return nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        # 初始层
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        # Inception 3a, 3b
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # Inception 4a
        x = self.inception4a(x)
        aux1 = self.aux1(x) if self.training and self.aux_logits else None

        # Inception 4b, 4c, 4d
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux2(x) if self.training and self.aux_logits else None

        # Inception 4e
        x = self.inception4e(x)
        x = self.maxpool4(x)

        # Inception 5a, 5b
        x = self.inception5a(x)
        x = self.inception5b(x)

        # 分类头
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        # 训练时返回主输出 + 辅助输出，评估时只返回主输出
        if self.training and self.aux_logits:
            return x, aux1, aux2
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# 测试代码
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoogLeNet(num_classes=10, aux_logits=True).to(device)
    dummy_input = torch.randn(2, 1, 224, 224).to(device)
    output = model(dummy_input)
    if isinstance(output, tuple):
        print(f"主输出形状: {output[0].shape}")  # [2, 10]
        print(f"辅助输出1: {output[1].shape}")  # [2, 10]
        print(f"辅助输出2: {output[2].shape}")  # [2, 10]
    else:
        print(f"输出形状: {output.shape}")

    # 打印模型结构（可选，会输出大量信息）
    summary(model, input_size=(1, 224, 224), batch_size=1)
