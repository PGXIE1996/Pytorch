#!/usr/bin/env python3
"""
UFC101 数据集划分脚本（支持训练/验证/测试）

功能：
- 扫描数据集根目录，每个子目录视为一个动作类别。
- 按比例划分：训练集:测试集 = train_ratio : (1-train_ratio)，再在训练集中划分训练:验证 = train_inner_ratio : (1-train_inner_ratio)
- 默认：train_ratio=0.8（训练+验证占80%），train_inner_ratio=0.8（训练占训练集的80%，即验证占20%）
- 输出 train.csv, val.csv, test.csv，每行格式：<文件相对路径>,<标签索引>
- 同时输出 class_indices.csv，记录类别名与索引的映射关系。

支持参数：
--data_dir          : 数据集根目录（默认 '../data/UCF101'）
--output_dir        : 输出列表文件的目录（默认 '../data/UCF101'）
--train_ratio       : 训练集+验证集占总样本的比例（默认 0.8，即测试集占0.2）
--train_inner_ratio : 训练集占（训练集+验证集）的比例（默认 0.8，即验证集占0.2）
--seed              : 随机种子（默认 42）
--stratify          : 是否按类别分层划分（默认 True）
--ext               : 视频文件扩展名（默认 '.avi'，可指定多个）
--format            : 输出格式，可选 'txt', 'csv', 'both'（默认 'csv'）
"""

import os
import random
import argparse
from collections import defaultdict
from glob import glob


def get_video_files(class_dir, extensions):
    files = []
    for ext in extensions:
        pattern = os.path.join(class_dir, f"*{ext}")
        files.extend(glob(pattern))
    files = [f for f in files if os.path.isfile(f)]
    return files


def write_list(file_path, samples, fmt):
    """写入列表文件，fmt: 'txt' 用空格分隔，'csv' 用逗号分隔"""
    with open(file_path, "w", encoding="utf-8") as f:
        for rel_path, label in samples:
            if fmt == "csv":
                f.write(f"{rel_path},{label}\n")
            else:  # txt
                f.write(f"{rel_path} {label}\n")


def write_class_map(file_path, class_to_idx, fmt):
    with open(file_path, "w", encoding="utf-8") as f:
        for name, idx in class_to_idx.items():
            if fmt == "csv":
                f.write(f"{idx},{name}\n")
            else:
                f.write(f"{idx} {name}\n")


def main():
    parser = argparse.ArgumentParser(description="划分UFC101数据集（训练/验证/测试）")
    parser.add_argument(
        "--data_dir", type=str, default="../data/UCF101", help="数据集根目录"
    )
    parser.add_argument(
        "--output_dir", type=str, default="../data/UCF101", help="输出目录"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="训练集+验证集占总样本的比例（测试集比例 = 1 - train_ratio）",
    )
    parser.add_argument(
        "--train_inner_ratio",
        type=float,
        default=0.8,
        help="训练集占（训练集+验证集）的比例（验证集比例 = 1 - train_inner_ratio）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--stratify", action="store_true", default=True, help="是否按类别分层划分"
    )
    parser.add_argument(
        "--ext", type=str, nargs="+", default=[".avi"], help="视频文件扩展名"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="csv",
        choices=["txt", "csv", "both"],
        help="输出格式：txt, csv 或 both",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"数据集目录不存在: {args.data_dir}")

    class_names = [
        d
        for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    ]
    class_names.sort()
    if not class_names:
        raise RuntimeError(f"在 {args.data_dir} 下未找到任何子目录（类别）")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    class_files = defaultdict(list)
    for class_name in class_names:
        class_dir = os.path.join(args.data_dir, class_name)
        files = get_video_files(class_dir, args.ext)
        if not files:
            print(f"警告: 类别 '{class_name}' 下没有找到扩展名为 {args.ext} 的视频文件")
        class_files[class_name] = files

    train_samples = []  # 最终训练集
    val_samples = []  # 验证集
    test_samples = []  # 测试集

    # 分层划分：每个类别独立处理
    for class_name, files in class_files.items():
        n_total = len(files)
        if n_total == 0:
            continue

        shuffled = files.copy()
        random.shuffle(shuffled)

        # 第一级划分：训练+验证  vs 测试
        n_trainval = int(round(args.train_ratio * n_total))
        # 保证至少有一个样本进入训练+验证集，且测试集非空（如果总数>=2）
        if n_trainval == 0:
            n_trainval = 1
        if n_trainval == n_total:
            n_trainval = n_total - 1

        trainval_files = shuffled[:n_trainval]
        test_files = shuffled[n_trainval:]

        # 第二级划分：在 trainval_files 中划分训练集和验证集
        n_trainval_total = len(trainval_files)
        n_train = int(round(args.train_inner_ratio * n_trainval_total))
        if n_train == 0:
            n_train = 1
        if n_train == n_trainval_total:
            n_train = n_trainval_total - 1

        # 再次打乱 trainval_files 以确保随机性（虽然已经打乱过，但为了清晰）
        random.shuffle(trainval_files)
        train_files = trainval_files[:n_train]
        val_files = trainval_files[n_train:]

        label = class_to_idx[class_name]

        for f in train_files:
            rel_path = os.path.relpath(f, start=args.data_dir)
            train_samples.append((rel_path, label))
        for f in val_files:
            rel_path = os.path.relpath(f, start=args.data_dir)
            val_samples.append((rel_path, label))
        for f in test_files:
            rel_path = os.path.relpath(f, start=args.data_dir)
            test_samples.append((rel_path, label))

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 根据 format 参数决定输出哪些文件
    if args.format == "both":
        formats = ["txt", "csv"]
    else:
        formats = [args.format]

    for fmt in formats:
        ext = fmt
        train_path = os.path.join(args.output_dir, f"train.{ext}")
        val_path = os.path.join(args.output_dir, f"val.{ext}")
        test_path = os.path.join(args.output_dir, f"test.{ext}")
        class_map_path = os.path.join(args.output_dir, f"class_indices.{ext}")

        write_list(train_path, train_samples, fmt)
        write_list(val_path, val_samples, fmt)
        write_list(test_path, test_samples, fmt)
        write_class_map(class_map_path, class_to_idx, fmt)

        print(f"已生成 {fmt.upper()} 文件:")
        print(f"  - {train_path}")
        print(f"  - {val_path}")
        print(f"  - {test_path}")
        print(f"  - {class_map_path}")

    print(f"\n数据集根目录: {args.data_dir}")
    print(f"类别数: {len(class_names)}")
    print(f"训练集样本数: {len(train_samples)}")
    print(f"验证集样本数: {len(val_samples)}")
    print(f"测试集样本数: {len(test_samples)}")
    total = len(train_samples) + len(val_samples) + len(test_samples)
    print(f"训练集比例: {len(train_samples)/total:.2f}")
    print(f"验证集比例: {len(val_samples)/total:.2f}")
    print(f"测试集比例: {len(test_samples)/total:.2f}")


if __name__ == "__main__":
    main()
