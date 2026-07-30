# -*- coding: utf-8 -*-
"""检查 CVAT 标注与图像是否匹配，并打印标注分布（用于验证数据集质量）"""

import argparse
from pathlib import Path
from collections import Counter, defaultdict

from config import DATA_ROOT, TASKS
from cvat_parser import parse_cvat_video_xml, get_task_indices_and_masks


def _is_single_video_dir(p: Path) -> bool:
    if not p.exists() or not p.is_dir():
        return False
    xml1 = p / "annotations.xml"
    xml2 = p / "annotations" / "annotations.xml"
    if not (xml1.exists() or xml2.exists()):
        return False
    return (p / "frames").exists() or (p / "images").exists()


def _discover_video_dirs(root: Path):
    """多视频根目录下自动发现 video 子目录。"""
    out = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith(".") and _is_single_video_dir(child):
            out.append(child)
    return out


def _inspect_one_video(video_dir: Path):
    """返回 rows, missing_count, total_count"""
    xml_path = video_dir / "annotations.xml"
    if not xml_path.exists():
        xml_path = video_dir / "annotations" / "annotations.xml"
    frames_dir = video_dir / "frames"
    if not frames_dir.exists():
        frames_dir = video_dir / "images"

    frames = parse_cvat_video_xml(str(xml_path), frames_dir=str(frames_dir))
    rows = get_task_indices_and_masks(frames, str(video_dir))

    missing = 0

    # 与训练一致：按 stem 匹配真实文件（忽略 .png/.jpg 大小写差异）
    img_dir = video_dir / ("frames" if (video_dir / "frames").exists() else "images")
    files_by_stem = {}
    if img_dir.exists():
        for fp in img_dir.iterdir():
            if fp.is_file():
                files_by_stem[fp.stem.lower()] = fp

    for r in rows:
        stem = Path(r["image_path"]).stem.lower()
        if stem not in files_by_stem:
            missing += 1
    return rows, missing, len(rows), xml_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    args = parser.parse_args()

    data_root = Path(args.data_root)

    # 单视频 or 多视频根目录自动识别
    if _is_single_video_dir(data_root):
        video_dirs = [data_root]
    else:
        video_dirs = _discover_video_dirs(data_root)

    if not video_dirs:
        print("未找到 annotations.xml，请指定 --data_root 为单个视频目录，或多视频根目录（其下每个子目录含 annotations.xml + images/frames）。")
        return 1

    # 全局汇总
    global_valid = {t: 0 for t in TASKS.keys()}
    global_total = 0
    global_cnt_by_task = {t: Counter() for t in TASKS.keys()}
    global_missing = 0

    for vdir in video_dirs:
        rows, missing, total, xml_path = _inspect_one_video(vdir)
        global_total += total
        global_missing += missing

        print("\n==============================")
        print("视频目录:", vdir)
        print("解析 XML:", xml_path)
        print(f"帧数: {total}")
        if missing:
            print(f"警告: {missing} 个帧对应的图像文件不存在")

        print("各任务标注数量与分布:")
        for task_name, (labels, _) in TASKS.items():
            valid = sum(1 for r in rows if r.get(f"{task_name}_valid"))
            global_valid[task_name] += valid
            print(f"  {task_name}: {valid}/{total} 帧有标注")

            cnt = Counter()
            for r in rows:
                idx = r.get(f"{task_name}_idx", -1)
                if idx >= 0:
                    cnt[idx] += 1
                    global_cnt_by_task[task_name][idx] += 1
            if cnt:
                for idx in sorted(cnt.keys()):
                    name = labels[idx] if idx < len(labels) else str(idx)
                    print(f"    - {name}: {cnt[idx]}")

    print("\n==============================")
    print("全局汇总（所有视频合并）")
    if global_missing:
        print(f"全局警告: {global_missing} 个帧对应的图像文件不存在")

    print("\n各任务标注数量与分布:")
    for task_name, (labels, _) in TASKS.items():
        valid = global_valid[task_name]
        print(f"  {task_name}: {valid}/{global_total} 帧有标注")
        cnt = global_cnt_by_task[task_name]
        if cnt:
            for idx in sorted(cnt.keys()):
                name = labels[idx] if idx < len(labels) else str(idx)
                print(f"    - {name}: {cnt[idx]}")

    print("\n检查完成。")
    return 0


if __name__ == "__main__":
    exit(main())
