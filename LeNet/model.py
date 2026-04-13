import torch
from torch import nn
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.activate = nn.Tanh()
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.activate(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.activate(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activate(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    data = torch.randn(1, 1, 28, 28).to(device)
    output = model(data)
    print(output)
    # summary(model, input_size=(1, 28, 28), batch_size=64)
