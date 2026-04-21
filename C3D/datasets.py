import os
import pandas as pd
import numpy as np
import cv2
import torchvision
import torch
import tqdm


# ---------- 自定义视频变换 ----------
class ToTensor3D:
    """将 (C, T, H, W) 的 uint8 numpy 数组转换为 (C, T, H, W) 的 float32 张量，并缩放到 [0,1]"""

    def __call__(self, video):
        # video: numpy.ndarray, dtype=uint8, 值域 [0,255], shape (C, T, H, W)
        return torch.from_numpy(video).float() / 255.0


class Normalize3D:
    """对视频张量进行标准化： (x - mean) / std，mean/std 可以是标量或 (C,) 数组"""

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1, 1)

    def __call__(self, video):
        return (video - self.mean) / self.std


class UCF101(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        root="../data/UCF101",
        split="train",
        length=16,
        crop_size=112,
        transform=None,  # 可选的外部变换
        target_transform=None,
    ):
        # 正确调用父类初始化
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self.length = length
        self.crop_size = crop_size

        # 根据 split 选择 CSV 文件
        filename = "train.csv" if split == "train" else "test.csv"
        csv_path = os.path.join(self.root, filename)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        dataset = pd.read_csv(csv_path, header=None)
        self.fnames = dataset[0].values  # 相对路径
        self.labels = dataset[1].values  # 标签索引

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # 加载视频并转换为 float32
        path = os.path.join(self.root, self.fnames[index])
        video = self._load_video(path)

        # 采样和裁剪
        c, f, h, w = video.shape

        # --- 时间随机采样（训练/测试都随机）---
        # 确保能取出 length 帧，步长 step = f // length
        if f >= self.length:
            step = f // self.length
            max_start = f - self.length * step
            start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            frame_indices = start_idx + np.arange(self.length) * step
        else:
            # 帧数不足：重复最后一帧补齐
            frame_indices = np.arange(self.length) % f

        # --- 空间随机裁剪（训练/测试都随机）---
        # 确保 crop_size 不超过原尺寸
        max_h = max(1, h - self.crop_size)
        max_w = max(1, w - self.crop_size)
        top = np.random.randint(0, max_h)
        left = np.random.randint(0, max_w)

        # 确保边界有效（防止 crop_size 大于尺寸）
        top = max(0, min(top, h - self.crop_size))
        left = max(0, min(left, w - self.crop_size))

        video = video[
            :, frame_indices, top : top + self.crop_size, left : left + self.crop_size
        ]

        # 应用外部 transform（
        if self.transform:
            video = self.transform(video)

        label = self.labels[index]
        if self.target_transform:
            label = self.target_transform(label)

        return video, label

    def _load_video(self, filename: str) -> np.ndarray:
        """加载视频，返回 (C, T, H, W) 的 float32 数组，值域 [0,255]"""
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        cap = cv2.VideoCapture(filename)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if not frames:
            raise RuntimeError(f"Video {filename} has no frames")

        # 堆叠为 (T, H, W, C) -> 转为 (C, T, H, W)
        video = np.stack(frames, axis=0)  # (T, H, W, 3)
        video = video.transpose(3, 0, 1, 2)  # (3, T, H, W)
        video = video.astype(np.float32)  # 转为 float32

        return video


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_mean_and_std(
    dataset: torch.utils.data.Dataset, samples: int = 128, batch_size: int = 8
):
    if samples is not None and len(dataset) > samples:
        indices = np.random.choice(len(dataset), samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    n = 0  # number of elements taken (should be equal to samples by end of for loop)
    s1 = 0.0  # sum of elements along channels (ends up as np.array of dimension (channels,))
    s2 = 0.0  # sum of squares of elements along channels (ends up as np.array of dimension (channels,))
    for (x, *_) in tqdm.tqdm(dataloader):
        x = x.transpose(0, 1).contiguous().view(3, -1)
        n += x.shape[1]
        s1 += torch.sum(x, dim=1).numpy()
        s2 += torch.sum(x**2, dim=1).numpy()
    mean = s1 / n  # type: np.ndarray
    std = np.sqrt(s2 / n - mean**2)  # type: np.ndarray

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    return mean, std


if __name__ == "__main__":
    set_seed(42)
    dataset1 = UCF101(root="../data/UCF101", split="train")
    train_loader = torch.utils.data.DataLoader(
        dataset1, batch_size=100, shuffle=True, num_workers=4
    )

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels.size())

        if i == 1:
            break
