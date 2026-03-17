"""
QA 資料載入模組

此模組負責載入與正規化不同格式的 QA 資料集
"""

import json
from typing import List, Dict, Any, Literal
import pandas as pd
from src.config.settings import get_settings


DataType = Literal["DI", "GEN"]


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
    
    def _get_qa_file_path(self) -> str:
        """取得 QA 資料檔案路徑"""
        if self.data_type == "DI":
            return self.settings.qa_file_path_DI
        elif self.data_type == "GEN":
            return self.settings.qa_file_path_GEN
        else:
            raise ValueError(f"不支援的資料類型: {self.data_type}")
    
    def load_and_normalize(self) -> List[Dict[str, Any]]:
        """
        載入並正規化 QA 資料
        
        Returns:
            正規化後的 QA 資料列表
        """
        if self.data_type == "DI":
            return self._load_csv_format()
        elif self.data_type == "GEN":
            return self._load_jsonl_format()
    
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
        normalized_data.append({
            "source": "QA_43_clean.csv",
            "query": row["Q"],
            "ground_truth_answer": row["GT"],
            "ground_truth_doc_ids": [ref_id] if ref_id else []
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
            normalized_data.append({
                "source": "qa_global_group_100.jsonl",
                "query": data.get("question", ""),
                "ground_truth_answer": data.get("answer", ""),
                "ground_truth_doc_ids": data.get("source_doc_ids", [])
            })
    
    return normalized_data
