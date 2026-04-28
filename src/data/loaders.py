"""
QA 資料載入模組

此模組負責載入與正規化不同格式的 QA 資料集
"""

import ast
import hashlib
import json
from typing import List, Dict, Any
import pandas as pd
from src.config.settings import get_settings


DataType = str


def build_context_doc_id(context: str) -> str:
    """依 context 內容建立穩定文件 ID。"""
    return hashlib.md5(str(context).strip().encode("utf-8")).hexdigest()


def _normalize_answer(data: Dict[str, Any]) -> str:
    """統一不同資料格式的答案欄位。"""
    answer = data.get("answer")
    if isinstance(answer, str):
        return answer.strip()
    if isinstance(answer, list):
        return "\n".join(str(item).strip() for item in answer if str(item).strip())

    answers = data.get("answers")
    if isinstance(answers, list):
        return "\n".join(str(item).strip() for item in answers if str(item).strip())
    if isinstance(answers, str):
        return answers.strip()
    return ""


def _parse_context_list(raw_contexts: Any) -> List[str]:
    """將 CSV 中的 reference_contexts 解析為 context 字串列表。"""
    if raw_contexts is None or (isinstance(raw_contexts, float) and pd.isna(raw_contexts)):
        return []

    if isinstance(raw_contexts, list):
        return [str(item).strip() for item in raw_contexts if str(item).strip()]

    text = str(raw_contexts).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]

    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if isinstance(parsed, str) and parsed.strip():
        return [parsed.strip()]
    return []


def _normalize_qa_scope(value: Any) -> str:
    """將題型值標準化為 local/global。"""
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text or text == "nan":
        return ""

    normalized = text.replace("_", "-").replace(" ", "")
    local_aliases = {"local", "single-hop", "singlehop", "1-hop", "1hop", "hop1", "one-hop"}
    global_aliases = {"global", "multi-hop", "multihop", "2-hop", "2hop", "hop2", "two-hop"}
    if normalized in local_aliases:
        return "local"
    if normalized in global_aliases:
        return "global"
    if "single" in normalized or "local" in normalized or normalized.startswith("1-") or normalized.startswith("1hop"):
        return "local"
    if "multi" in normalized or "global" in normalized or normalized.startswith("2-") or normalized.startswith("2hop"):
        return "global"
    return ""


def _resolve_question_type(data: Dict[str, Any]) -> Dict[str, str]:
    """統一不同來源題型欄位，回傳 Q_type/Hop_Level/qa_scope。"""
    q_type_candidates = [
        data.get("Q_type"),
        data.get("question_type"),
        data.get("Question_Type"),
    ]
    hop_candidates = [
        data.get("Hop_Level"),
        data.get("hop-level"),
        data.get("hop_level"),
    ]

    q_type = ""
    for candidate in q_type_candidates:
        if candidate is not None and str(candidate).strip():
            q_type = str(candidate).strip()
            break

    hop_level = ""
    for candidate in hop_candidates:
        if candidate is not None and str(candidate).strip():
            hop_level = str(candidate).strip()
            break

    scope = _normalize_qa_scope(q_type) or _normalize_qa_scope(hop_level)
    return {
        "Q_type": q_type,
        "Hop_Level": hop_level,
        "qa_scope": scope,
    }


class QADataLoader:
    """
    QA 資料載入器類別
    
    負責將不同格式的 QA 資料集統整為標準格式
    
    標準格式：
    {
        "source": str,              # 資料來源檔案
        "query": str,               # 問題
        "ground_truth_answer": str, # 標準答案
        "ground_truth_doc_ids": List[str]  # 相關文件 ID
    }
    """
    
    def __init__(self, data_type: DataType = "DI"):
        """
        初始化 QA 資料載入器
        
        Args:
            data_type: 資料類型，可選 "DI" 或 "GEN"
        """
        self.data_type = data_type
        self.settings = get_settings()
        self.qa_file_path = self._get_qa_file_path()
        self.qa_loader = self.settings.data_config.get_qa_loader(data_type)
    
    def _get_qa_file_path(self) -> str:
        """取得 QA 資料檔案路徑"""
        qa_file_path = self.settings.data_config.get_qa_file_path(self.data_type)
        if qa_file_path:
            return qa_file_path
        raise ValueError(f"不支援的資料類型或未設定 QA 路徑: {self.data_type}")
    
    def load_and_normalize(self) -> List[Dict[str, Any]]:
        """
        載入並正規化 QA 資料
        
        Returns:
            正規化後的 QA 資料列表
        """
        return load_and_normalize_qa_by_type(self.data_type, self.qa_file_path)
    
    def _load_csv_format(self) -> List[Dict[str, Any]]:
        """
        載入 CSV 格式的 QA 資料（DI 資料集）
        
        欄位格式：Q, GT, REF
        
        Returns:
            正規化後的 QA 資料列表
        """
        normalized_data = []
        df_csv = pd.read_csv(self.qa_file_path)
        
        for _, row in df_csv.iterrows():
            ref_id = str(row["REF"]).strip() if pd.notna(row["REF"]) else ""
            
            normalized_data.append({
                "source": self.qa_file_path.split("/")[-1],  # 檔案名稱
                "query": row["Q"],
                "ground_truth_answer": row["GT"],
                "ground_truth_doc_ids": [ref_id] if ref_id else []
            })
        
        return normalized_data
    
    def _load_jsonl_format(self) -> List[Dict[str, Any]]:
        """
        載入 JSONL 格式的 QA 資料（GEN 資料集）
        
        欄位格式：question, answer, source_doc_ids
        
        Returns:
            正規化後的 QA 資料列表
        """
        normalized_data = []
        
        with open(self.qa_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                normalized_data.append({
                    "source": self.qa_file_path.split("/")[-1],  # 檔案名稱
                    "query": data.get("question", ""),
                    "ground_truth_answer": data.get("answer", ""),
                    "ground_truth_doc_ids": data.get("source_doc_ids", [])
                })
        
        return normalized_data


def load_and_normalize_qa_CSR_DI(csv_path: str) -> List[Dict[str, Any]]:
    """
    載入並正規化 CSV 格式的 QA 資料（向後兼容函數）
    
    Args:
        csv_path: CSV 檔案路徑
    
    Returns:
        正規化後的 QA 資料列表
    """
    normalized_data = []
    df_csv = pd.read_csv(csv_path)
    
    for _, row in df_csv.iterrows():
        ref_id = str(row["REF"]).strip() if pd.notna(row["REF"]) else ""
        q_meta = _resolve_question_type(row.to_dict())
        normalized_data.append({
            "source": "QA_43_clean.csv",
            "query": row["Q"],
            "ground_truth_answer": row["GT"],
            "ground_truth_doc_ids": [ref_id] if ref_id else [],
            "Q_type": q_meta["Q_type"],
            "Hop_Level": q_meta["Hop_Level"],
            "qa_scope": q_meta["qa_scope"],
        })
    
    return normalized_data


def load_and_normalize_qa_CSR_full(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    載入並正規化 JSONL 格式的 QA 資料（向後兼容函數）
    
    Args:
        jsonl_path: JSONL 檔案路徑
    
    Returns:
        正規化後的 QA 資料列表
    """
    normalized_data = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            q_meta = _resolve_question_type(data)
            normalized_data.append({
                "source": "qa_global_group_100.jsonl",
                "query": data.get("question", ""),
                "ground_truth_answer": data.get("answer", ""),
                "ground_truth_doc_ids": data.get("source_doc_ids", []),
                "Q_type": q_meta["Q_type"],
                "Hop_Level": q_meta["Hop_Level"],
                "qa_scope": q_meta["qa_scope"],
            })
    
    return normalized_data


def load_and_normalize_qa_ultradomain(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    載入並正規化 UltraDomain QA 資料。

    支援原始欄位：
    - input / question
    - answers / answer
    - context
    - source_doc_ids（若已預先生成）
    """
    normalized_data = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            query = data.get("input") or data.get("question") or ""
            answer = _normalize_answer(data)
            source_doc_ids = data.get("source_doc_ids")
            q_meta = _resolve_question_type(data)

            if not source_doc_ids:
                context = data.get("context", "")
                source_doc_ids = [build_context_doc_id(context)] if context else []

            normalized_data.append({
                "source": jsonl_path.split("/")[-1],
                "query": query,
                "ground_truth_answer": answer,
                "ground_truth_doc_ids": source_doc_ids,
                "Q_type": q_meta["Q_type"],
                "Hop_Level": q_meta["Hop_Level"],
                "qa_scope": q_meta["qa_scope"],
            })

    return normalized_data


def load_and_normalize_qa_ragas_metadata(csv_path: str) -> List[Dict[str, Any]]:
    """
    載入並正規化 RAGAS metadata CSV。

    支援欄位：
    - user_input
    - reference
    - reference_contexts
    """
    normalized_data = []
    df_csv = pd.read_csv(csv_path)
    source_name = csv_path.split("/")[-1]

    for _, row in df_csv.iterrows():
        contexts = _parse_context_list(row.get("reference_contexts"))
        doc_ids = []
        seen_doc_ids = set()

        for context in contexts:
            doc_id = build_context_doc_id(context)
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                doc_ids.append(doc_id)

        q_meta = _resolve_question_type(row.to_dict())
        normalized_data.append({
            "source": source_name,
            "query": str(row.get("user_input", "")).strip(),
            "ground_truth_answer": str(row.get("reference", "")).strip(),
            "ground_truth_doc_ids": doc_ids,
            "Q_type": q_meta["Q_type"],
            "Hop_Level": q_meta["Hop_Level"],
            "qa_scope": q_meta["qa_scope"],
        })

    return normalized_data


def load_and_normalize_qa_by_type(data_type: str, qa_file_path: str = "") -> List[Dict[str, Any]]:
    """依 data_type 與 config 中的 qa_loader 載入對應 QA。"""
    settings = get_settings()
    actual_qa_file_path = qa_file_path or settings.data_config.get_qa_file_path(data_type)
    qa_loader = settings.data_config.get_qa_loader(data_type)

    if qa_loader == "csr_csv":
        return load_and_normalize_qa_CSR_DI(csv_path=actual_qa_file_path)
    if qa_loader == "csr_jsonl":
        return load_and_normalize_qa_CSR_full(jsonl_path=actual_qa_file_path)
    if qa_loader == "ultradomain_jsonl":
        return load_and_normalize_qa_ultradomain(jsonl_path=actual_qa_file_path)
    if qa_loader == "ragas_metadata_csv":
        return load_and_normalize_qa_ragas_metadata(csv_path=actual_qa_file_path)

    raise ValueError(f"不支援的 QA loader: {qa_loader} (data_type={data_type})")
