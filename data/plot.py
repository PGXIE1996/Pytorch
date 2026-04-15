from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torch
import matplotlib.pyplot as plt

# MNIST数据集：train60000个，test10000个
train_data = FashionMNIST(
    root="./",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True,
    num_workers=0,
)

classes = train_data.classes
print(classes)

# 取第一个 batch
images, labels = next(iter(train_loader))

# 查看形状
# print("images shape:", images.shape)  # torch.Size([64, 1, 28, 28])
# print("labels shape:", labels.shape)  # torch.Size([64])

# 可视化一个Batch的图形
plt.figure(figsize=(12, 5))
for idx, image in enumerate(images):
    plt.subplot(4, 16, idx + 1)
    plt.imshow(images[idx].squeeze(), cmap="gray")
    plt.title(classes[labels[idx]], size=10)
    plt.axis("off")

plt.tight_layout()
plt.show()
