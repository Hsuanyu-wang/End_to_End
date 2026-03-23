#!/usr/bin/env python3
"""
圖譜品質批量分析工具

用法:
  # 分析單一 graphml
  python scripts/analyze_graphs.py --graphml storage/lightrag/DI_lightrag_default_hybrid/graph_chunk_entity_relation.graphml

  # 掃描 storage/ 下所有 graphml，批量比較
  python scripts/analyze_graphs.py --scan_all --output results/graph_quality_comparison.csv

  # 指定目錄掃描
  python scripts/analyze_graphs.py --scan_dir storage/lightrag --output results/lightrag_quality.csv
"""

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
from src.evaluation.metrics.graph_quality import GraphQualityMetrics


def find_graphml_files(root_dir: str) -> list:
    """遞迴搜尋目錄下所有 .graphml 檔案"""
    results = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".graphml"):
                results.append(os.path.join(dirpath, f))
    return sorted(results)


def analyze_single(path: str) -> dict:
    """分析單一 graphml，印出並回傳結果"""
    metrics = GraphQualityMetrics.compute_from_graphml(path)
    rel_path = os.path.relpath(path, _PROJECT_ROOT)
    metrics["graphml_path"] = rel_path
    print(f"\n{'='*60}")
    print(f"  {rel_path}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if k in ("entity_type_distribution", "relation_type_distribution"):
            print(f"  {k}:")
            if isinstance(v, dict):
                for tk, tv in list(v.items())[:15]:
                    print(f"    {tk}: {tv}")
                if len(v) > 15:
                    print(f"    ... ({len(v) - 15} more)")
        elif k != "graphml_path":
            print(f"  {k}: {v}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="圖譜品質批量分析工具")
    parser.add_argument("--graphml", type=str, help="分析單一 GraphML 檔案")
    parser.add_argument("--scan_all", action="store_true", help="掃描 storage/ 下所有 .graphml")
    parser.add_argument("--scan_dir", type=str, help="掃描指定目錄下所有 .graphml")
    parser.add_argument("--output", type=str, default="", help="輸出 CSV 路徑（批量模式）")
    args = parser.parse_args()

    if args.graphml:
        full_path = os.path.join(_PROJECT_ROOT, args.graphml) if not os.path.isabs(args.graphml) else args.graphml
        if not os.path.exists(full_path):
            print(f"❌ 檔案不存在: {full_path}")
            sys.exit(1)
        metrics = analyze_single(full_path)
        report_path = full_path.replace(".graphml", "_quality.json")
        GraphQualityMetrics.save_report(metrics, report_path)
        return

    scan_root = None
    if args.scan_all:
        scan_root = os.path.join(_PROJECT_ROOT, "storage")
    elif args.scan_dir:
        scan_root = os.path.join(_PROJECT_ROOT, args.scan_dir) if not os.path.isabs(args.scan_dir) else args.scan_dir

    if scan_root is None:
        parser.print_help()
        return

    if not os.path.isdir(scan_root):
        print(f"❌ 目錄不存在: {scan_root}")
        sys.exit(1)

    files = find_graphml_files(scan_root)
    if not files:
        print(f"⚠️  未找到任何 .graphml 檔案於 {scan_root}")
        return

    print(f"📂 找到 {len(files)} 個 GraphML 檔案")
    records = []
    for f in files:
        try:
            metrics = analyze_single(f)
            flat = {k: v for k, v in metrics.items()
                    if k not in ("entity_type_distribution", "relation_type_distribution")}
            flat["num_entity_types"] = len(metrics.get("entity_type_distribution", {}))
            flat["num_relation_types"] = len(metrics.get("relation_type_distribution", {}))
            records.append(flat)
        except Exception as e:
            print(f"⚠️  分析失敗 {f}: {e}")

    if records:
        df = pd.DataFrame(records)
        df = df.sort_values("node_count", ascending=False)
        output_path = args.output or os.path.join(_PROJECT_ROOT, "results", "graph_quality_comparison.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n📊 品質比較表已儲存: {output_path}")
        print(f"\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()
