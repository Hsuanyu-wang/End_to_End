"""
LightRAG 答案中 References 區塊解析，以及與 retrieved_ids 筆數/型態的對照稽核。

不驗證 Title 與 UUID 的字面相等（通常無法對齊），僅提供結構化摘要供人工或後續規則使用。
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# LightRAG prompt 規範：- [n] Document Title 或 * [n] Document Title
_REFERENCE_LINE = re.compile(
    r"^\s*[-*]\s*\[(\d+)\]\s*(.+?)\s*$",
    re.MULTILINE,
)
_REF_SECTION = re.compile(
    r"(?m)^###\s*References\s*\n(?P<body>.*?)(?=\Z|(?=^#{1,6}\s))",
    re.DOTALL | re.IGNORECASE,
)


def extract_references_block(answer: str) -> str:
    """擷取 `### References` 標題以下區塊（若無則回傳空字串）。"""
    if not answer:
        return ""
    m = _REF_SECTION.search(answer)
    if m:
        return (m.group("body") or "").strip()
    # 無標題時：若全文仍含清單式引用，可傳整段供 parse 嘗試
    return ""


def parse_reference_list(reference_block: str) -> List[Tuple[int, str]]:
    """
    解析引用行為 (序號, Document Title)。

    Returns:
        依出現順序的 (引用編號, 標題文字) 列表。
    """
    if not reference_block:
        return []
    out: List[Tuple[int, str]] = []
    for m in _REFERENCE_LINE.finditer(reference_block):
        idx = int(m.group(1))
        title = (m.group(2) or "").strip()
        out.append((idx, title))
    return out


def normalize_retrieved_ids(retrieved_ids: Union[str, List[str], None]) -> List[str]:
    """將 detailed_results 的 retrieved_ids 欄位（字串或列表）正規化為去空白後的列表。"""
    if retrieved_ids is None:
        return []
    if isinstance(retrieved_ids, list):
        return [str(x).strip() for x in retrieved_ids if x is not None and str(x).strip()]
    s = str(retrieved_ids).strip()
    if not s:
        return []
    parts = re.split(r"[\n,]+", s)
    return [p.strip() for p in parts if p.strip()]


def audit_sample(
    generated_answer: str,
    retrieved_ids: Union[str, List[str], None],
) -> Dict[str, Any]:
    """
    單筆樣本稽核：引用筆數、retrieved_ids 筆數、序號是否由 1 連續遞增。

    Returns:
        字典，含 ref_count、retrieved_id_count、reference_indices、index_ok 等。
    """
    ref_body = extract_references_block(generated_answer)
    parsed = parse_reference_list(ref_body) if ref_body else parse_reference_list(generated_answer)
    ids = normalize_retrieved_ids(retrieved_ids)
    indices = [p[0] for p in parsed]

    index_ok = False
    if indices:
        expected = list(range(1, len(indices) + 1))
        index_ok = indices == expected

    return {
        "reference_count": len(parsed),
        "retrieved_id_count": len(ids),
        "reference_indices": indices,
        "reference_titles": [p[1] for p in parsed],
        "index_sequential_from_one": index_ok,
        "count_match": len(parsed) == len(ids) if parsed or ids else True,
        "has_references_heading": bool(_REF_SECTION.search(generated_answer or "")),
    }


def audit_csv_file(
    csv_path: Union[str, Path],
    answer_col: str = "generated_answer",
    ids_col: str = "retrieved_ids",
    max_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    逐列讀取評估結果 CSV，對每筆執行 audit_sample 並附加 idx / query（若存在）。

    Args:
        csv_path: detailed_results.csv 路徑
        answer_col: 生成答案欄位名
        ids_col: retrieved_ids 欄位名
        max_rows: 僅處理前 N 列（除錯用）
    """
    path = Path(csv_path)
    rows_out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            ans = row.get(answer_col, "") or ""
            rid = row.get(ids_col, "") or ""
            summary = audit_sample(ans, rid)
            summary["row_index"] = i + 1
            if row.get("idx"):
                summary["idx"] = row["idx"]
            if row.get("query"):
                summary["query"] = row["query"][:200]
            rows_out.append(summary)
    return rows_out


def summarize_audit(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """聚合多筆稽核結果的簡要統計。"""
    n = len(rows)
    if not n:
        return {"samples": 0}
    with_heading = sum(1 for r in rows if r.get("has_references_heading"))
    seq_ok = sum(1 for r in rows if r.get("index_sequential_from_one"))
    count_match = sum(1 for r in rows if r.get("count_match"))
    return {
        "samples": n,
        "has_references_heading_ratio": round(with_heading / n, 4),
        "index_sequential_from_one_ratio": round(seq_ok / n, 4),
        "reference_retrieved_count_match_ratio": round(count_match / n, 4),
    }


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="稽核 LightRAG 答案引用與 retrieved_ids 對照")
    parser.add_argument("csv_path", help="detailed_results.csv 路徑")
    parser.add_argument("--max-rows", type=int, default=None, help="僅處理前 N 筆")
    parser.add_argument("--json", action="store_true", help="輸出 JSON")
    args = parser.parse_args()

    audited = audit_csv_file(args.csv_path, max_rows=args.max_rows)
    agg = summarize_audit(audited)
    if args.json:
        print(json.dumps({"summary": agg, "rows": audited}, ensure_ascii=False, indent=2))
    else:
        print("摘要:", json.dumps(agg, ensure_ascii=False))
        for r in audited[:20]:
            print(
                f"  idx={r.get('idx', r.get('row_index'))} "
                f"refs={r['reference_count']} ids={r['retrieved_id_count']} "
                f"seq={r['index_sequential_from_one']} count_match={r['count_match']}"
            )
        if len(audited) > 20:
            print(f"  ... 共 {len(audited)} 筆，僅顯示前 20 筆")
    sys.exit(0)
