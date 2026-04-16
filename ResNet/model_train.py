import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from ResNet.model import ResNet18

# ===================== 配置中心 =====================
BATCH_SIZE = 128
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
EARLY_STOP_PATIENCE = 5

# 文件路径
OUTPUT_DIR = "output"
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best.pth")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pth")
LOG_CSV_PATH = os.path.join(OUTPUT_DIR, "log.csv")
FIG_PATH = os.path.join(OUTPUT_DIR, "result.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("../data", exist_ok=True)


# ===================== 数据加载 =====================
def get_data_loaders():
    # 图片路径
    train_root = "../data/cat-dog_data/train"
    val_root = "../data/cat-dog_data/val"

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4879566431045532, 0.4547620713710785, 0.41685497760772705],
                std=[0.25928181409835815, 0.25256675481796265, 0.255256712436676],
            ),
        ]
    )
    # 加载数据集
    train_data = ImageFolder(root=train_root, transform=transform)
    val_data = ImageFolder(root=val_root, transform=transform)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader


# ===================== 单轮训练/验证 =====================
def run_epoch(model, dataloader, is_train, optimizer, criterion, device):
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        pbar = tqdm(dataloader, desc="Train" if is_train else "Val")
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

            pbar.set_postfix(
                {"loss": f"{total_loss / total:.4f}", "acc": f"{correct / total:.4f}"}
            )

    return total_loss / total, correct / total


# ===================== 主训练流程 =====================
def train_model(
    model,
    train_loader,
    val_loader,
    epochs=EPOCHS,
    device=DEVICE,
    patience=EARLY_STOP_PATIENCE,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=3,
    )

    start_epoch = 0
    best_acc = 0.0
    best_loss = float("inf")
    patience_counter = 0

    # 加载断点
    if os.path.exists(CHECKPOINT_PATH):
        print(f"恢复断点训练：{CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint.get("best_acc", 0.0)
        best_loss = checkpoint.get("best_loss", float("inf"))
        patience_counter = checkpoint.get("patience_counter", 0)

    # 初始化 log CSV（如果文件不存在则写入表头）
    if not os.path.exists(LOG_CSV_PATH) or os.path.getsize(LOG_CSV_PATH) == 0:
        with open(LOG_CSV_PATH, "w") as f:
            f.write("epoch,train_loss,train_acc,train_time,val_loss,val_acc,val_time\n")

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # 训练
        start_time = time.time()
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            is_train=True,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        train_time = time.time() - start_time

        # 验证
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            is_train=False,
            optimizer=None,
            criterion=criterion,
            device=device,
        )
        val_time = time.time() - start_time - train_time

        # 记录到 history CSV（每 epoch 一行）
        with open(LOG_CSV_PATH, "a") as f:
            f.write(
                f"{epoch+1},{train_loss:.6f},{train_acc:.6f},{train_time},{val_loss:.6f},{val_acc:.6f},{val_time}\n"
            )

        # 保存最佳模型（基于验证准确率）
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        # 早停检测（基于验证损失）
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        # 更新学习率
        scheduler.step(val_loss)

        # 保存断点（包含早停计数器）
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
                "best_loss": best_loss,
                "patience_counter": patience_counter,
            },
            CHECKPOINT_PATH,
        )


# ===================== 绘图函数  =====================
def plot_loss_acc(df: pd.DataFrame, save_path=FIG_PATH):  # 加上类型提示
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # 改用 df['列名'] 的形式获取数据
    ax1.plot(df["epoch"], df["train_loss"], "#ff4757", label="Train Loss", lw=2)
    ax1.plot(df["epoch"], df["val_loss"], "#5352ed", label="Val Loss", lw=2)
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 改用 df['列名'] 的形式获取数据
    ax2.plot(df["epoch"], df["train_acc"], "#2ed573", label="Train Acc", lw=2)
    ax2.plot(df["epoch"], df["val_acc"], "#ffa502", label="Val Acc", lw=2)
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
    train_loader, val_loader = get_data_loaders()

    # 加载模型
    model = ResNet18(num_classes=2).to(DEVICE)
    train_model(model, train_loader, val_loader, epochs=EPOCHS)

    df = pd.read_csv(LOG_CSV_PATH)
    plot_loss_acc(df, FIG_PATH)
