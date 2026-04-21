import pandas as pd
import torch
import cv2
import numpy as np
import os
from C3D.model import *

# 全局缓存类别映射（避免重复读取）
_IDX_TO_CLASS = None

def get_class_mapping(csv_path="../data/UCF101/class_indices.csv"):
    global _IDX_TO_CLASS
    if _IDX_TO_CLASS is None:
        # 获取脚本所在目录，构建绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        abs_csv_path = os.path.join(script_dir, csv_path)
        class_df = pd.read_csv(abs_csv_path, header=None, names=["index", "class"])
        _IDX_TO_CLASS = dict(zip(class_df["index"], class_df["class"]))
    return _IDX_TO_CLASS


def inference(model, video):
    """
    video: torch.Tensor 形状 (C, T, H, W) 或 (1, C, T, H, W)
    返回: (predicted_class, confidence)
    """
    # 自动添加 batch 维度
    if video.dim() == 4:
        video = video.unsqueeze(0)

    # 确保设备一致
    device = next(model.parameters()).device
    video = video.to(device)

    # 获取类别映射
    idx_to_class = get_class_mapping()

    model.eval()
    with torch.no_grad():
        logits = model(video)
        probs = torch.softmax(logits, dim=1)
        prob, idx = torch.max(probs, dim=1)

    class_name = idx_to_class[idx.item()]
    confidence = prob.item()
    return class_name, confidence


def load_and_crop_video(filepath, clip_len=16, target_size=(112, 112)):
    """返回形状 (C, T, H, W) 的 numpy 数组，值域 [0, 255] (float32)"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Video not found: {filepath}")
    cap = cv2.VideoCapture(filepath)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"No frames read from {filepath}")
    if len(frames) < clip_len:
        indices = list(range(len(frames))) + [len(frames) - 1] * (
            clip_len - len(frames)
        )
    else:
        indices = np.linspace(0, len(frames) - 1, clip_len, dtype=int)
    sampled = [frames[i] for i in indices]
    video = np.stack(sampled, axis=0)  # (T, H, W, 3)
    video = video.transpose(3, 0, 1, 2)  # (3, T, H, W)
    video = video.astype(np.float32)  # 值域 [0,255]
    return video


def transformer(video):
    """输入: (C, T, H, W) numpy 数组，值域 [0,255]；输出: (C, T, H, W) torch.Tensor，已归一化"""
    video = torch.from_numpy(video).float() / 255.0
    mean = torch.tensor([0.438, 0.424, 0.382], dtype=torch.float32).view(3, 1, 1, 1)
    std = torch.tensor([0.264, 0.260, 0.271], dtype=torch.float32).view(3, 1, 1, 1)
    video = (video - mean) / std
    return video


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = C3D(num_classes=101, pretrained=False)
    # 确保 best.pth 文件存在
    model.load_state_dict(torch.load("output/best.pth", map_location=device))
    model = model.to(device)

    video_path = "../data/UCF101/YoYo/v_YoYo_g01_c01.avi"
    # 确保路径正确，或者使用绝对路径
    video_np = load_and_crop_video(video_path)
    video_tensor = transformer(video_np)  # 形状 (3,16,112,112)

    pred_class, conf = inference(model, video_tensor)
    print(f"预测动作类别：{pred_class}，置信度：{conf:.4f}")
