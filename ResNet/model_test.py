import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from ResNet.model import ResNet18

# 配置文件
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径
OUTPUT_DIR = "output"
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best.pth")
CM_SAVE_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

# 类别名称（可选）
CLASS_NAMES = [
    "Cat",
    "Dog",
]


def get_test_loaders():
    test_root = "../data/cat-dog_data/test"
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
    test_data = ImageFolder(root=test_root, transform=transform)
    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return test_loader


def test_model(model, dataloader, device=DEVICE):
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Testing")
        for data, label in pbar:
            data, label = data.to(device), label.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            # 收集结果（移到 CPU 并转为 numpy 以便后续计算）
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

            # 实时更新进度条显示当前准确率
            current_acc = (np.array(all_preds) == np.array(all_labels)).mean()
            pbar.set_postfix({"acc": f"{current_acc:.4f}"})

    # 计算总体准确率
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    # 打印分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    return accuracy, cm, all_labels, all_preds


def plot_confusion_matrix(cm, class_names, save_path):
    """绘制混淆矩阵热力图"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix on Test Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"混淆矩阵已保存至 {save_path}")


if __name__ == "__main__":
    # 加载模型
    model = ResNet18(num_classes=2).to(DEVICE)
    # 加载训练好的权重（注意：确保 best.pth 是完整的 state_dict）
    state_dict = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print(f"成功加载模型权重：{BEST_MODEL_PATH}")

    test_loader = get_test_loaders()
    test_acc, cm, _, _ = test_model(model, test_loader)
    print(f"\n测试准确率：{test_acc:.4f}")

    # 绘制并保存混淆矩阵
    plot_confusion_matrix(cm, CLASS_NAMES, CM_SAVE_PATH)
