import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from AlexNet.model import AlexNet

# ===================== 配置中心（严格类型定义） =====================
BATCH_SIZE = 128
LR = 0.001
TRAIN_SPLIT = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
PIN_MEMORY = True
LR_STEP_SIZE = 15
LR_GAMMA = 0.1
EPOCHS = 50

# 路径
OUTPUT_DIR = "output"
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best.pth")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pth")
LOG_PATH = os.path.join(OUTPUT_DIR, "train_log.csv")
FIG_PATH = os.path.join(OUTPUT_DIR, "result.png")

# 自动创建目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("../data", exist_ok=True)


# ===================== 数据加载 =====================
def get_data_loaders():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(size=(227, 227))]
    )
    dataset = FashionMNIST(
        root="../data", train=True, transform=transform, download=True
    )
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    return train_loader, val_loader


# ===================== 单轮训练/验证 =====================
def run_epoch(model, dataloader, is_train, optimizer, criterion, device):
    """
    运行一个 epoch 的训练或验证
    返回 (平均损失, 准确率)
    """
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        pbar = tqdm(
            dataloader,
            desc="Train" if is_train else "Val",
        )
        for data, label in pbar:
            data, label = data.to(device), label.to(device)
            output = model(data)

            loss = criterion(output, label)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += data.size(0)

            # 更新进度条显示
            pbar.set_postfix(
                {"loss": f"{total_loss / total:.4f}", "acc": f"{correct / total:.4f}"}
            )

    return total_loss / total, correct / total


# ===================== 主训练流程 =====================
def train_model(model, train_loader, val_loader, epochs=EPOCHS, device=DEVICE):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA
    )

    start_epoch = 0
    best_acc = 0.0

    # 加载断点（如果存在）
    if os.path.exists(CHECKPOINT_PATH):
        print(f"🔄 恢复断点训练：{CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint.get("best_acc", 0.0)

    # 准备日志文件（追加模式，写入表头如果文件为空）
    if not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0:
        with open(LOG_PATH, "w") as f:
            f.write("epoch,phase,loss,acc,time\n")

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # 训练阶段
        start_time = time.time()
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            is_train=True,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
        )
        train_time = time.time() - start_time

        # 验证阶段
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            is_train=False,
            optimizer=None,
            criterion=criterion,
            device=DEVICE,
        )
        val_time = time.time() - start_time - train_time

        # 记录日志（追加）
        with open(LOG_PATH, "a") as f:
            f.write(
                f"{epoch+1},train,{train_loss:.6f},{train_acc:.6f},{train_time:.2f}\n"
            )
            f.write(f"{epoch+1},val,{val_loss:.6f},{val_acc:.6f},{val_time:.2f}\n")

        # 保存历史
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 保存最佳模型（基于验证准确率）
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        # 更新学习率
        scheduler.step()

        # 保存断点
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            },
            CHECKPOINT_PATH,
        )

    return pd.DataFrame(history)


# ===================== 绘图函数 =====================
def plot_loss_acc(df, save_path=FIG_PATH):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Loss 曲线
    ax1.plot(df.epoch, df.train_loss, "#ff4757", label="Train Loss", lw=2)
    ax1.plot(df.epoch, df.val_loss, "#5352ed", label="Val Loss", lw=2)
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Accuracy 曲线
    ax2.plot(df.epoch, df.train_acc, "#2ed573", label="Train Acc", lw=2)
    ax2.plot(df.epoch, df.val_acc, "#ffa502", label="Val Acc", lw=2)
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"图片已保存至 {save_path}")


# ===================== 主函数 =====================
if __name__ == "__main__":
    model = AlexNet().to(DEVICE)
    train_loader, val_loader = get_data_loaders()
    history_df = train_model(model, train_loader, val_loader, epochs=EPOCHS)
    plot_loss_acc(history_df)
