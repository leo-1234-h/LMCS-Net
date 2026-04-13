#!/usr/bin/env python3
"""
yolo_eval_per_class.py

用法示例：
python yolo_eval_per_class.py \
  --model runs/detect/train/weights/best.pt \
  --data /data1/lsl/lxw/data/bdz_yolo/data.yaml \
  --labels_root /data1/lsl/lxw/data/bdz_yolo/labels \
  --device 0

说明：
- 会在当前 runs/val 下写入验证产物（save_json=True）
- 输出每类的 P/R/AP50/AP(0.5:0.95)/F1（若能从 ultralytics 返回）
- 若无法从 model.val() 获取 per-class 数组，会提示并仅打印整体指标（仍会保存 JSON）
"""

import os
import argparse
import yaml
import math
from collections import defaultdict

# 防止 headless 环境 matplotlib TkAgg 报错
import matplotlib
matplotlib.use("Agg")

from ultralytics import YOLO

def count_gt_from_labels(labels_root):
    """统计 labels_root 下每个类别的 ground-truth bbox 数量（递归）"""
    gt_counts = defaultdict(int)
    for dirpath, _, files in os.walk(labels_root):
        for fn in files:
            if not fn.lower().endswith(".txt"):
                continue
            fp = os.path.join(dirpath, fn)
            try:
                with open(fp, "r") as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        parts = s.split()
                        try:
                            cls = int(float(parts[0]))
                        except:
                            continue
                        gt_counts[cls] += 1
            except Exception:
                pass
    return gt_counts

def try_extract_per_class_metrics(results):
    """
    尝试从 model.val() 返回的 Results 对象中提取 per-class arrays。
    不同版本的 ultralytics 可能属性名不同，尽量兼容。
    返回 dict: { 'precision': list_or_none, 'recall':..., 'ap50':..., 'ap':..., 'f1':..., 'classes': [ids] }
    """
    out = dict(precision=None, recall=None, ap50=None, ap=None, f1=None, classes=None)
    metrics = getattr(results, "metrics", None)
    if not metrics:
        return out

    # DetMetrics object often has .box attribute that contains p,r,f1,all_ap,ap_class_index or methods ap(), ap50()
    box = getattr(metrics, "box", None)
    if box:
        # try p, r, f1
        p = getattr(box, "p", None)
        r = getattr(box, "r", None)
        f1 = getattr(box, "f1", None)
        out['precision'] = list(p) if p is not None else None
        out['recall'] = list(r) if r is not None else None
        out['f1'] = list(f1) if f1 is not None else None

        # try ap50 / ap
        try:
            ap50 = box.ap50() if hasattr(box, "ap50") else None
        except Exception:
            ap50 = None
        try:
            ap = box.ap() if hasattr(box, "ap") else None
        except Exception:
            ap = None

        # sometimes ap50/ap are numpy arrays or lists
        out['ap50'] = list(ap50) if ap50 is not None else None
        out['ap'] = list(ap) if ap is not None else None

        # classes present / number of classes
        nc = getattr(box, "nc", None)
        if nc is not None:
            out['classes'] = list(range(int(nc)))
        else:
            # maybe ap_class_index or ap_class exists
            ap_cls_idx = getattr(box, "ap_class_index", None) or getattr(box, "ap_class", None)
            if ap_cls_idx is not None:
                try:
                    out['classes'] = [int(x) for x in ap_cls_idx]
                except Exception:
                    out['classes'] = None
        return out

    # fallback: maybe metrics has direct lists or methods
    for name in ("p", "precision"):
        p = getattr(metrics, name, None)
        if p is not None:
            out['precision'] = list(p) if hasattr(p, "__len__") else None
            break
    for name in ("r", "recall"):
        r = getattr(metrics, name, None)
        if r is not None:
            out['recall'] = list(r) if hasattr(r, "__len__") else None
            break
    # AP arrays
    for name in ("ap", "ap_class", "ap_per_class"):
        a = getattr(metrics, name, None)
        if a is not None:
            out['ap'] = list(a) if hasattr(a, "__len__") else None
            break
    # attempt ap50 method or map50
    ap50 = None
    if hasattr(metrics, "map50"):
        try:
            ap50 = metrics.map50()
        except Exception:
            ap50 = None
    out['ap50'] = list(ap50) if (ap50 is not None and hasattr(ap50, "__len__")) else None

    return out

def pretty_print_table(class_ids, class_names, gt_counts, metrics_dict):
    """在控制台打印表格"""
    headers = ["class_id", "class_name", "GT_count", "Precision", "Recall", "AP@0.5", "AP@0.5:0.95", "F1"]
    print("\n" + "-"*100)
    print("{:>8}  {:>20}  {:>8}  {:>9}  {:>9}  {:>9}  {:>13}  {:>7}".format(*headers))
    print("-"*100)
    for i, cid in enumerate(class_ids):
        name = class_names.get(cid, "") if class_names else ""
        gt = gt_counts.get(cid, 0)
        p = metrics_dict.get('precision')
        r = metrics_dict.get('recall')
        ap50 = metrics_dict.get('ap50')
        ap = metrics_dict.get('ap')
        f1 = metrics_dict.get('f1')

        def safe_get(arr, idx):
            try:
                return None if arr is None else float(arr[idx])
            except Exception:
                return None

        p_v = safe_get(p, i)
        r_v = safe_get(r, i)
        ap50_v = safe_get(ap50, i)
        ap_v = safe_get(ap, i)
        f1_v = safe_get(f1, i)

        def fm(x):
            return f"{x:.3f}" if (x is not None and (not (isinstance(x, float) and math.isnan(x)))) else "--"

        print(f"{cid:>8}  {name:>20}  {gt:8d}  {fm(p_v):>9}  {fm(r_v):>9}  {fm(ap50_v):>9}  {fm(ap_v):>13}  {fm(f1_v):>7}")
    print("-"*100 + "\n")

def load_class_names_from_yaml(data_yaml):
    """尝试从 data.yaml 里读取 names 列表（class 名称）"""
    if not data_yaml or not os.path.isfile(data_yaml):
        return {}
    try:
        with open(data_yaml, "r") as f:
            d = yaml.safe_load(f)
            # 常见格式： {names: [...]} 或 {train:..., val:..., nc:..., names: [...]}
            names = d.get("names") or d.get("class_names") or d.get("labels") or None
            if isinstance(names, dict):
                # dict mapping id->name
                return {int(k): v for k, v in names.items()}
            if isinstance(names, list):
                return {i: n for i, n in enumerate(names)}
    except Exception:
        pass
    return {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="训练好的权重路径，如 runs/.../weights/best.pt")
    parser.add_argument("--data", type=str, required=True, help="data yaml (必须)，用于知道 class names 与 nc")
    parser.add_argument("--labels_root", type=str, default=None, help="（可选）用于统计 GT 的 labels 根目录")
    parser.add_argument("--project", type=str, default="runs/val", help="val 结果保存的 project")
    parser.add_argument("--name", type=str, default="exp", help="val 结果保存 name")
    parser.add_argument("--device", type=str, default="0", help="CUDA device 或 cpu")
    parser.add_argument("--plots", action="store_true", help="是否允许绘图（headless 环境建议不启用）")
    parser.add_argument("--conf", type=float, default=0.001, help="val 时的 conf 阈值")
    args = parser.parse_args()

    # 读取 data.yaml 里的 class 名称与 nc
    class_names = load_class_names_from_yaml(args.data)
    nc = None
    try:
        with open(args.data, "r") as f:
            d = yaml.safe_load(f)
            nc = int(d.get("nc") or d.get("num_classes") or (len(class_names) if class_names else None))
    except Exception:
        pass

    # 加载模型
    print("加载模型：", args.model)
    model = YOLO(args.model)

    # run validation
    print("开始验证（model.val）...（plots=%s）" % ("True" if args.plots else "False"))
    results = model.val(
        data=args.data,
        device=args.device,
        plots=args.plots,    # 默认False 时不会绘图，避免 TkAgg 错误
        save_json=True,      # 保存详细 json 以备查证
        project=args.project,
        name=args.name,
        conf=args.conf
    )

    # 尝试从 results 中提取 per-class metrics
    per_class = try_extract_per_class_metrics(results)
    # fallback class list
    if per_class.get('classes') is None:
        if nc:
            class_ids = list(range(nc))
        else:
            # try infer from keys in per_class arrays
            arr = per_class.get('ap') or per_class.get('ap50') or per_class.get('precision')
            if arr:
                class_ids = list(range(len(arr)))
            else:
                class_ids = []
    else:
        class_ids = per_class['classes']

    # 统计 GT counts（若传入 labels_root）
    gt_counts = {}
    if args.labels_root:
        print("统计 ground-truth bbox 数量（从 labels_root）...")
        gt_counts = count_gt_from_labels(args.labels_root)
    else:
        gt_counts = {}

    # prepare class name mapping
    names_map = class_names or {}

    # Print summary overall (if available)
    print("\n==== Overall results summary (if available) ====")
    try:
        # results may have metrics.results_dict or results.metrics.results_dict
        rdict = None
        if hasattr(results, "metrics") and hasattr(results.metrics, "results_dict"):
            rdict = results.metrics.results_dict
        elif hasattr(results, "results_dict"):
            rdict = results.results_dict
        if rdict:
            print("总体指标：")
            for k, v in rdict.items():
                print(f"  {k}: {v}")
    except Exception:
        pass

    # 打印 per-class 表格
    if not class_ids:
        print("未能从验证返回中识别到 per-class 指标数组。你可以查看保存的 JSON（project/name）或升级 ultralytics 版本以支持 per-class 输出。")
        return

    pretty_print_table(class_ids, names_map, gt_counts, per_class)
    print("验证并打印完成。验证保存路径（含 save_json）在：", os.path.join(args.project, args.name))

if __name__ == "__main__":
    main()
