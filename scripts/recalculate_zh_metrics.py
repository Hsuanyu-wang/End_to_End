"""
回溯重算中文友善指標（ROUGE / BLEU / METEOR）

掃描 results/exp/ 下所有 detailed_results.csv，
以 jieba 分詞版本重算 ROUGE / BLEU / METEOR，
將結果 append 為新欄位 (rouge1_zh, rouge2_zh, rougeL_zh, rougeLsum_zh, bleu_zh, meteor_zh)。
同時更新對應 run 內的 global_summary_report 以及頂層 global_summary.xlsx。

用法:
    python scripts/recalculate_zh_metrics.py [--dry-run] [--results-dir results/exp]
"""

import argparse
import os
import sys
import glob
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.metrics.generation import ROUGEMetric, BLEUMetric, METEORMetric

ZH_COLUMNS = [
    "rouge1_zh", "rouge2_zh", "rougeL_zh", "rougeLsum_zh",
    "bleu_zh", "meteor_zh",
]


def recalculate_single_csv(csv_path: str, rouge: ROUGEMetric, bleu: BLEUMetric, meteor: METEORMetric, dry_run: bool) -> bool:
    """重算單一 detailed_results.csv，回傳是否有修改。"""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(csv_path, encoding="utf-8")

    if "generated_answer" not in df.columns or "ground_truth_answer" not in df.columns:
        return False

    if all(col in df.columns for col in ZH_COLUMNS):
        non_null = df[ZH_COLUMNS[0]].notna().sum()
        if non_null > 0:
            return False

    r1_list, r2_list, rL_list, rLs_list = [], [], [], []
    bleu_list, meteor_list = [], []

    for _, row in df.iterrows():
        gen = str(row.get("generated_answer", "") or "")
        gt = str(row.get("ground_truth_answer", "") or "")

        idx_val = row.get("idx", "")
        if str(idx_val) == "平均":
            r1_list.append(None)
            r2_list.append(None)
            rL_list.append(None)
            rLs_list.append(None)
            bleu_list.append(None)
            meteor_list.append(None)
            continue

        r1, r2, rL, rLs = rouge.compute(gen, gt)
        b = bleu.compute(gen, gt)
        m = meteor.compute(gen, gt)

        r1_list.append(r1)
        r2_list.append(r2)
        rL_list.append(rL)
        rLs_list.append(rLs)
        bleu_list.append(b)
        meteor_list.append(m)

    df["rouge1_zh"] = r1_list
    df["rouge2_zh"] = r2_list
    df["rougeL_zh"] = rL_list
    df["rougeLsum_zh"] = rLs_list
    df["bleu_zh"] = bleu_list
    df["meteor_zh"] = meteor_list

    avg_mask = df["idx"].astype(str) == "平均"
    if avg_mask.any():
        for col in ZH_COLUMNS:
            df.loc[avg_mask, col] = df.loc[~avg_mask, col].mean()

    if not dry_run:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return True


def update_run_summary(run_dir: str, dry_run: bool):
    """更新 run 目錄內的 global_summary_report.csv/.xlsx。"""
    csv_path = os.path.join(run_dir, "global_summary_report.csv")
    xlsx_path = os.path.join(run_dir, "global_summary_report.xlsx")

    if not os.path.exists(csv_path):
        return

    try:
        summary_df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        summary_df = pd.read_csv(csv_path, encoding="utf-8")

    pipeline_dirs = [
        d for d in os.listdir(run_dir)
        if os.path.isdir(os.path.join(run_dir, d))
    ]

    updated = False
    for pname in pipeline_dirs:
        detail_csv = os.path.join(run_dir, pname, "detailed_results.csv")
        if not os.path.exists(detail_csv):
            continue
        try:
            detail_df = pd.read_csv(detail_csv, encoding="utf-8-sig")
        except Exception:
            detail_df = pd.read_csv(detail_csv, encoding="utf-8")

        data_rows = detail_df[detail_df["idx"].astype(str) != "平均"]

        mask = summary_df["pipeline_name"] == pname
        if not mask.any():
            continue

        for col in ZH_COLUMNS:
            if col in data_rows.columns:
                summary_df.loc[mask, f"avg_{col}"] = data_rows[col].mean()
                updated = True

    if updated and not dry_run:
        summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        summary_df.to_excel(xlsx_path, index=False)


def update_master_summary(master_xlsx_path: str, dry_run: bool):
    """更新頂層 global_summary.xlsx：對每一列的 run_dir 找到 detailed_results 重新計算平均。"""
    if not os.path.exists(master_xlsx_path):
        return

    master_df = pd.read_excel(master_xlsx_path)
    parent_dir = os.path.dirname(master_xlsx_path)

    updated = False
    for i, row in master_df.iterrows():
        run_dir_name = str(row.get("run_dir", ""))
        pipeline_name = str(row.get("pipeline_name", ""))
        if not run_dir_name or not pipeline_name:
            continue

        detail_csv = os.path.join(parent_dir, run_dir_name, pipeline_name, "detailed_results.csv")
        if not os.path.exists(detail_csv):
            continue
        try:
            detail_df = pd.read_csv(detail_csv, encoding="utf-8-sig")
        except Exception:
            detail_df = pd.read_csv(detail_csv, encoding="utf-8")

        data_rows = detail_df[detail_df["idx"].astype(str) != "平均"]

        for col in ZH_COLUMNS:
            if col in data_rows.columns:
                master_df.loc[i, f"avg_{col}"] = data_rows[col].mean()
                updated = True

    if updated and not dry_run:
        master_df.to_excel(master_xlsx_path, index=False)
        print(f"  📚 已更新累積總表: {master_xlsx_path}")


def main():
    parser = argparse.ArgumentParser(description="回溯重算中文友善指標")
    parser.add_argument("--results-dir", default="/home/End_to_End_RAG/results/exp",
                        help="結果根目錄（預設 results/exp）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只計算不寫入")
    args = parser.parse_args()

    rouge = ROUGEMetric()
    bleu = BLEUMetric()
    meteor = METEORMetric()

    csv_files = sorted(glob.glob(os.path.join(args.results_dir, "**", "detailed_results.csv"), recursive=True))
    print(f"找到 {len(csv_files)} 個 detailed_results.csv")

    if args.dry_run:
        print("⚠️  dry-run 模式：只計算不寫入")

    modified = 0
    for csv_path in tqdm(csv_files, desc="重算中文指標"):
        if recalculate_single_csv(csv_path, rouge, bleu, meteor, args.dry_run):
            modified += 1

    print(f"\n✅ 重算完成：{modified}/{len(csv_files)} 個檔案已更新")

    run_dirs = set()
    for csv_path in csv_files:
        parts = csv_path.split(os.sep)
        for j, part in enumerate(parts):
            if part.startswith("evaluation_results_"):
                run_dirs.add(os.sep.join(parts[:j + 1]))
                break

    print(f"更新 {len(run_dirs)} 個 run 目錄的 global_summary_report...")
    for rd in tqdm(sorted(run_dirs), desc="更新 run summary"):
        update_run_summary(rd, args.dry_run)

    master_files = glob.glob(os.path.join(args.results_dir, "*/global_summary.xlsx"))
    master_files += glob.glob(os.path.join(args.results_dir, "global_summary.xlsx"))
    master_files = list(set(master_files))
    print(f"更新 {len(master_files)} 個 global_summary.xlsx...")
    for mf in master_files:
        update_master_summary(mf, args.dry_run)

    print("\n🎉 全部完成！")


if __name__ == "__main__":
    main()
