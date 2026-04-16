import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm


def calculate_mean_std(data_root, batch_size=128, num_workers=4, device="cuda"):
    # 只做 Resize + ToTensor，不归一化
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # 输出 [0,1] 范围的张量
        ]
    )
    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    mean = torch.zeros(3).to(device)
    std = torch.zeros(3).to(device)
    total_pixels = 0

    for images, _ in tqdm(loader, desc="Computing mean & std"):
        images = images.to(device)  # 移到 GPU，形状 (B, 3, H, W)
        B, C, H, W = images.shape
        # 展平为 (B, C, H*W) 以便求和
        images_flat = images.view(B, C, -1)  # (B, C, H*W)

        # 累加每个通道的像素和
        mean += images_flat.sum(dim=0).sum(
            dim=1
        )  # 先对 batch 求和，再对像素位置求和，得到 (C,)
        # 累加每个通道的平方和
        std += (images_flat**2).sum(dim=0).sum(dim=1)
        total_pixels += B * H * W

    mean /= total_pixels
    std = torch.sqrt(std / total_pixels - mean**2)
    return mean.cpu().tolist(), std.cpu().tolist()


if __name__ == "__main__":
    mean, std = calculate_mean_std("../data/cat-dog_data/train")
    print(mean, std)
