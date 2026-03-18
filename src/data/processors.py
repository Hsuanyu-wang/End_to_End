"""
資料處理模組

此模組負責處理原始資料並轉換為 LlamaIndex Document 物件，
支援多種資料模式以適應不同的 RAG 需求。
"""

import json
import os
from typing import List, Literal
from llama_index.core import Document
from src.config.settings import get_settings


DataMode = Literal["natural_text", "markdown", "key_value_text", "unstructured_text"]
DataType = Literal["DI", "GEN"]


class DataProcessor:
    """
    資料處理器類別
    
    負責將原始 JSONL 格式的維護紀錄轉換為不同模式的 LlamaIndex Document
    
    Attributes:
        mode: 資料模式，決定文本格式化方式
        data_type: 資料類型（DI 或 GEN）
        settings: 模型設定物件
    """
    
    def __init__(self, mode: DataMode = "natural_text", data_type: DataType = "DI"):
        """
        初始化資料處理器
        
        Args:
            mode: 資料模式，可選 "natural_text", "markdown", "key_value_text", "unstructured_text"
            data_type: 資料類型，可選 "DI" 或 "GEN"
        """
        self.mode = mode
        self.data_type = data_type
        self.settings = get_settings()
        self.raw_file_path = self._get_raw_file_path()
    
    def _get_raw_file_path(self) -> str:
        """取得原始資料檔案路徑"""
        if self.data_type == "DI":
            return self.settings.raw_file_path_DI
        elif self.data_type == "GEN":
            return self.settings.raw_file_path_GEN
        else:
            raise ValueError(f"不支援的資料類型: {self.data_type}")
    
    def _extract_metadata(self, data: dict) -> dict:
        """
        從原始資料中提取 metadata
        
        Args:
            data: 原始資料字典
        
        Returns:
            metadata 字典
        """
        metadata_keys = ["NO", "Customer", "Engineers", "Service Start", "Service End"]
        metadata = {}
        
        for key in metadata_keys:
            if key in data:
                metadata[key] = data[key]
        
        return metadata
    
    def _format_natural_text(self, data: dict, metadata: dict) -> str:
        """
        自然語言格式
        
        特點：乾淨的自然語意描述，沒有指令雜訊，最適合 Embedding
        """
        return (
            f"這是一筆關於 {data['Customer']} 的維護紀錄。\n"
            f"處理工程師為 {data['Engineers']}\n"
            f"進行時間為 {data['Service Start']} 至 {data['Service End']}。\n"
            f"狀態描述 (Description): {data['Description']}\n"
            f"處置動作 (Action): {data['Action']}"
        )
    
    def _format_markdown(self, data: dict) -> str:
        """
        Markdown 格式
        
        特點：利用 Markdown 原生層級與標題提示，區分「背景資訊」與「抽取目標」
        """
        return (
            f"# Maintenance Record (維護紀錄)\n\n"
            f"## Metadata (Context Only)\n"
            f"- **Record NO**: {data.get('NO', 'N/A')}\n"
            f"- **Customer**: {data.get('Customer', 'N/A')}\n"
            f"- **Engineers**: {data.get('Engineers', 'N/A')}\n"
            f"- **Service Time**: {data.get('Service Start', 'N/A')} to {data.get('Service End', 'N/A')}\n"
            f"- **Maintenance Type**: {data.get('維護類別', 'N/A')}\n\n"
            f"## Target Details (For Entity and Relation Extraction)\n"
            f"### Description (狀態描述)\n"
            f"{data.get('Description', '無描述')}\n\n"
            f"### Action (處置動作)\n"
            f"{data.get('Action', '無動作')}\n"
        )
    
    def _format_key_value_text(self, data: dict) -> str:
        """
        Key-Value 格式
        
        特點：簡單的鍵值對格式
        """
        return (
            f"NO: {data['NO']}\n"
            f"Customer: {data['Customer']}\n"
            f"Engineers: {data['Engineers']}\n"
            f"Start Time: {data['Service Start']}\n"
            f"End Time: {data['Service End']}\n"
            f"Maintenance Type: {data.get('維護類別', 'N/A')}\n"
            f"Description: {data['Description']}\n"
            f"Action: {data['Action']}"
        )
    
    def _format_unstructured_text(self, data: dict) -> str:
        """
        非結構化文本格式
        
        特點：帶有明確邊界與防呆指令，專給 LLM 抽取器使用
        """
        return (
            "=== BACKGROUND CONTEXT (DO NOT EXTRACT ENTITIES FROM THIS SECTION) ===\n"
            f"Record NO: {data['NO']}\n"
            f"Customer: {data['Customer']}\n"
            f"Engineer: {data['Engineers']}\n"
            "======================================================================\n\n"
            "=== TARGET TEXT FOR EXTRACTION (ONLY EXTRACT ENTITIES AND RELATIONS FROM HERE) ===\n"
            f"Description: {data['Description']}\n"
            f"Action: {data['Action']}\n"
            "=================================================================================="
        )
    
    def _create_document(self, data: dict) -> Document:
        """
        根據模式建立 Document 物件
        
        Args:
            data: 原始資料字典
        
        Returns:
            LlamaIndex Document 物件
        """
        metadata = self._extract_metadata(data)
        
        # 使用 NO 作為 document ID，確保可追溯
        doc_id = data.get("NO", None)
        
        if self.mode == "natural_text":
            text = self._format_natural_text(data, metadata)
            return Document(
                text=text,
                metadata=metadata,
                doc_id=doc_id,
                excluded_embed_metadata_keys=list(metadata.keys()),
                excluded_llm_metadata_keys=list(metadata.keys())
            )
        
        elif self.mode == "markdown":
            text = self._format_markdown(data)
            return Document(
                text=text,
                metadata=metadata,
                doc_id=doc_id,
                excluded_llm_metadata_keys=list(metadata.keys()),
                excluded_embed_metadata_keys=list(metadata.keys())
            )
        
        elif self.mode == "key_value_text":
            text = self._format_key_value_text(data)
            return Document(
                text=text,
                metadata=metadata,
                doc_id=doc_id
            )
        
        elif self.mode == "unstructured_text":
            text = self._format_unstructured_text(data)
            return Document(
                text=text,
                metadata=metadata,
                doc_id=doc_id,
                excluded_llm_metadata_keys=list(metadata.keys())
            )
        
        else:
            raise ValueError(f"不支援的資料模式: {self.mode}")
    
    def process(self) -> List[Document]:
        """
        處理原始資料並返回 Document 列表
        
        Returns:
            Document 物件列表
        
        Raises:
            FileNotFoundError: 當原始資料檔案不存在時
        """
        if not os.path.exists(self.raw_file_path):
            raise FileNotFoundError(f"找不到原始資料檔案: {self.raw_file_path}")
        
        documents = []
        
        with open(self.raw_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                doc = self._create_document(data)
                documents.append(doc)
        
        return documents
    
    def save_processed_data(self, output_dir: str = None):
        """
        儲存處理後的資料
        
        Args:
            output_dir: 輸出目錄，預設為 Data/{mode}_{data_type}/
        """
        documents = self.process()
        
        if output_dir is None:
            output_dir = f"/home/End_to_End_RAG/Data/{self.mode}_{self.data_type}"
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "documents.jsonl")
        
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(json.dumps(doc.model_dump(), ensure_ascii=False) + "\n")
        
        print(f"已儲存處理後的資料至: {output_path}")


def data_processing(mode: DataMode = "natural_text", data_type: DataType = "DI") -> List[Document]:
    """
    處理原始資料並轉換為 LlamaIndex Document 物件（向後兼容函數）
    
    Args:
        mode: 資料模式，可選 "key_value_text", "natural_text", "unstructured_text", "markdown"
        data_type: 資料類型，可選 "DI" 或 "GEN"
    
    Returns:
        Document 物件列表
    """
    processor = DataProcessor(mode=mode, data_type=data_type)
    return processor.process()


if __name__ == "__main__":
    mode = "natural_text"
    data_type = "DI"
    
    processor = DataProcessor(mode=mode, data_type=data_type)
    processor.save_processed_data()
