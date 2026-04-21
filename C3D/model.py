import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class C3D(nn.Module):
    def __init__(self, num_classes=101, pretrained=False):
        super().__init__()

        # 定义卷积块的辅助函数，减少冗余代码
        def conv_bn_relu(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)
                ),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
            )

        # 网络结构
        self.conv1 = conv_bn_relu(3, 64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = conv_bn_relu(64, 128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = conv_bn_relu(128, 256)
        self.conv3b = conv_bn_relu(256, 256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = conv_bn_relu(256, 512)
        self.conv4b = conv_bn_relu(512, 512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = conv_bn_relu(512, 512)
        self.conv5b = conv_bn_relu(512, 512)
        self.pool5 = nn.MaxPool3d(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)
        )

        self.flatten = nn.Flatten()

        # 全连接层也可以考虑加入 BatchNorm1d，但 C3D 传统做法是只在卷积层加
        self.fc6 = nn.Linear(512 * 4 * 4, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(0.5)

        # 权重初始化
        self.__initialize_weights()

        #
        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3b(self.conv3a(x)))
        x = self.pool4(self.conv4b(self.conv4a(x)))
        x = self.pool5(self.conv5b(self.conv5a(x)))

        x = self.flatten(x)
        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        x = F.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.fc8(x)

        return x

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
            # Conv1
            "features.0.weight": "conv1.0.weight",
            "features.0.bias": "conv1.0.bias",
            # Conv2
            "features.3.weight": "conv2.0.weight",
            "features.3.bias": "conv2.0.bias",
            # Conv3a
            "features.6.weight": "conv3a.0.weight",
            "features.6.bias": "conv3a.0.bias",
            # Conv3b
            "features.8.weight": "conv3b.0.weight",
            "features.8.bias": "conv3b.0.bias",
            # Conv4a
            "features.11.weight": "conv4a.0.weight",
            "features.11.bias": "conv4a.0.bias",
            # Conv4b
            "features.13.weight": "conv4b.0.weight",
            "features.13.bias": "conv4b.0.bias",
            # Conv5a
            "features.16.weight": "conv5a.0.weight",
            "features.16.bias": "conv5a.0.bias",
            # Conv5b
            "features.18.weight": "conv5b.0.weight",
            "features.18.bias": "conv5b.0.bias",
            # fc6
            "classifier.0.weight": "fc6.weight",
            "classifier.0.bias": "fc6.bias",
            # fc7
            "classifier.3.weight": "fc7.weight",
            "classifier.3.bias": "fc7.bias",
        }

        p_dict = torch.load("output/c3d-pretrained.pth")
        new_state_dict = {}
        for p_key, model_key in corresp_name.items():
            if p_key in p_dict:
                new_state_dict[model_key] = p_dict[p_key]
        self.load_state_dict(new_state_dict, strict=False)  # 允许缺失 BN 和 fc8
        print(f"Loaded {len(new_state_dict)} keys from pretrained weights")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = C3D(num_classes=101, pretrained=True).to(device)

    # 输入：(Batch=1, Channels=3, Frames=16, H=112, W=112)
    input_tensor = torch.randn(1, 3, 16, 112, 112).to(device)

    output = model(input_tensor)
    print(f"Output shape: {output.shape}")

    # 打印详细参数
    summary(model, input_size=(3, 16, 112, 112), batch_size=1, device=device.type)
