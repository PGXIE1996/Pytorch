import time
import torch
from torch import nn
from tqdm import tqdm
import os
from datetime import datetime
import socket
from torch.utils.tensorboard import SummaryWriter  # 推荐使用 PyTorch 内置
from C3D.model import C3D
from C3D.datasets import UCF101, ToTensor3D, Normalize3D
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd


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


def train_model(
    model,
    output_dir="output",
    epochs=50,
    batch_size=32,
    device="cuda",
    lr_patience=5,  # 学习率降低的耐心值
    early_stop_patience=10,  # 早停耐心值
    lr=0.001,
    momentum=0.9,
    weight_decay=5e-4,
    run_test=False,
):
    os.makedirs(output_dir, exist_ok=True)
    model = model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=lr_patience
    )

    start_epoch = 0
    best_acc = 0.0
    best_loss = float("inf")
    early_stop_counter = 0

    # 恢复断点（略，与原代码相同但注意 patience_counter 改为 early_stop_counter）
    checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
    best_model_path = os.path.join(output_dir, "best.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint.get("best_acc", 0.0)
        best_loss = checkpoint.get("best_loss", float("inf"))
        early_stop_counter = checkpoint.get("early_stop_counter", 0)

    # 初始化日志文件
    log_path = os.path.join(output_dir, "log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,train_acc,train_time,val_loss,val_acc,val_time\n")

    # 数据变换
    video_transform = transforms.Compose(
        [
            ToTensor3D(),
            Normalize3D(mean=[0.438, 0.424, 0.382], std=[0.264, 0.260, 0.271]),
        ]
    )

    # 创建数据集（确保 UCF101 支持 split='val'）
    trainset = UCF101(split="train", transform=video_transform)
    valset = UCF101(split="val", transform=video_transform)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )

    # TensorBoard writer（只创建一次）
    log_dir = os.path.join(
        output_dir,
        "runs",
        datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + socket.gethostname(),
    )
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}", flush=True)

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
        val_start = time.time()
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            is_train=False,
            optimizer=None,
            criterion=criterion,
            device=device,
        )
        val_time = time.time() - val_start

        # 记录 CSV
        with open(log_path, "a") as f:
            f.write(
                f"{epoch+1},{train_loss:.6f},{train_acc:.6f},{train_time:.2f},"
                f"{val_loss:.6f},{val_acc:.6f},{val_time:.2f}\n"
            )

        # TensorBoard 记录
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_acc", train_acc, epoch)
        writer.add_scalar("train_time", train_time, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("val_acc", val_acc, epoch)
        writer.add_scalar("val_time", val_time, epoch)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        # 早停判断（基于验证损失）
        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # 更新学习率
        scheduler.step(val_loss)

        # 保存断点
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
                "best_loss": best_loss,
                "early_stop_counter": early_stop_counter,
            },
            checkpoint_path,
        )

    writer.close()
    print(f"Training finished. Best val accuracy: {best_acc:.4f}")

    # Load best weights
    if epochs != 0:
        checkpoint = torch.load(os.path.join(best_model_path))
        model.load_state_dict(checkpoint["state_dict"])

    if run_test:
        for split in ["val", "test"]:
            dataset = UCF101(split=split, transform=video_transform)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
            )
            loss, acc = run_epoch(model, dataloader, False, None, criterion, device)
            print(f"{split}: {loss:.4f}, {acc:.4f}")


# ===================== 绘图函数  =====================
def plot_loss_acc(df: pd.DataFrame, save_path):  # 加上类型提示
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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = C3D(num_classes=101, pretrained=True).to(device)
    train_model(model, batch_size=16)
    df = pd.read_csv("output/log.csv")
    plot_loss_acc(df, "output/result.png")
