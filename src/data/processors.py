"""
資料處理模組

此模組負責處理原始資料並轉換為 LlamaIndex Document 物件，
支援多種資料模式以適應不同的 RAG 需求。
"""

import hashlib
import json
import os
from typing import List, Literal, Any
from llama_index.core import Document
from src.config.settings import get_settings


DataMode = Literal["natural_text", "markdown", "key_value_text", "unstructured_text"]
DataType = str


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
        self.data_config = self.settings.data_config
        self.document_loader = self.data_config.get_document_loader(data_type)
        self.raw_file_path = self._get_raw_file_path()
    
    def _get_raw_file_path(self) -> str:
        """取得原始資料檔案路徑"""
        raw_file_path = self.data_config.get_document_file_path(self.data_type)
        if raw_file_path:
            return raw_file_path
        raise ValueError(f"不支援的資料類型或未設定文件路徑: {self.data_type}")
    
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

    @staticmethod
    def _build_context_hash(context: str) -> str:
        """建立穩定的文件 ID，讓 QA 與檢索評測可對齊。"""
        return hashlib.md5(context.encode("utf-8")).hexdigest()
    
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
            # f"- **Maintenance Type**: {data.get('維護類別', 'N/A')}\n\n"
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
            # f"Maintenance Type: {data.get('維護類別', 'N/A')}\n"
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
        # doc_id = data.get("NO", None)
        doc_id = str(data.get("NO", "")) if data.get("NO") is not None else None
        
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

    # ------------------------------------------------------------------
    # Bio 透析病歷 JSONL loader
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_bio_dates(data: dict) -> set:
        """收集患者所有透析日期（從四個欄位聯集）。"""
        dates = set()
        for ts, _ in data.get("nurse_record", []):
            dates.add(str(ts).split()[0])
        for item in data.get("order", []):
            if item:
                dates.add(str(item[0]))
        for item in data.get("a+p", []):
            if item:
                dates.add(str(item[0]))
        for item in data.get("fiber_list", []):
            if item:
                dates.add(str(item[0]))
        return dates

    def _format_bio_session_text(self, data: dict, date: str) -> str:
        """將單次透析日（session）的資料格式化為指定 data_mode 的文本。"""
        case_number = data.get("case_number", "")
        age = data.get("age", "")
        blood_type = str(data.get("blood_type", "") or "")
        blood_rh = str(data.get("blood_rh", "") or "")
        hbv = data.get("hbv", 0)
        hcv = data.get("hcv", 0)
        hiv = data.get("hiv", 0)
        patient_header = (
            f"患者病歷號 {case_number}（{age} 歲，血型 {blood_type}{blood_rh}，"
            f"HBV/HCV/HIV: {hbv}/{hcv}/{hiv}）"
        )

        # 篩選當日護理紀錄（比較日期前綴）
        day_nurses = [
            f"[{ts.split(' ', 1)[1] if ' ' in ts else ts}] {note}"
            for ts, note in data.get("nurse_record", [])
            if str(ts).startswith(date) and note
        ]
        nurse_text = "\n".join(day_nurses) if day_nurses else "（無）"

        # 篩選當日醫囑
        day_orders = []
        for item in data.get("order", []):
            if len(item) >= 2 and str(item[0]) == date:
                drug = item[1]
                extra = " ".join(str(x) for x in item[2:] if x)
                day_orders.append(f"{drug} {extra}".strip())
        order_text = "\n".join(day_orders) if day_orders else "（無）"

        # 篩選當日評估計畫
        day_ap = [item[1] for item in data.get("a+p", []) if len(item) >= 2 and str(item[0]) == date]
        ap_text = "\n".join(day_ap) if day_ap else "（無）"

        # 篩選當日過濾器狀態
        day_fiber = []
        for item in data.get("fiber_list", []):
            if item and str(item[0]) == date:
                status = item[1]
                day_fiber.append(", ".join(status) if isinstance(status, list) else str(status))
        fiber_text = "\n".join(day_fiber) if day_fiber else "（無）"

        if self.mode == "markdown":
            return (
                f"# 透析紀錄（{date}）\n\n"
                f"## 患者資訊\n"
                f"- 病歷號: {case_number}\n"
                f"- 年齡: {age} 歲\n"
                f"- 血型: {blood_type}{blood_rh}\n"
                f"- HBV: {hbv}, HCV: {hcv}, HIV: {hiv}\n\n"
                f"## 護理紀錄\n{nurse_text}\n\n"
                f"## 醫囑\n{order_text}\n\n"
                f"## 評估計畫 (A+P)\n{ap_text}\n\n"
                f"## 過濾器狀態\n{fiber_text}\n"
            )

        if self.mode == "key_value_text":
            return (
                f"病歷號: {case_number}\n"
                f"年齡: {age}\n"
                f"血型: {blood_type}{blood_rh}\n"
                f"HBV/HCV/HIV: {hbv}/{hcv}/{hiv}\n"
                f"透析日期: {date}\n"
                f"護理紀錄:\n{nurse_text}\n"
                f"醫囑:\n{order_text}\n"
                f"評估計畫:\n{ap_text}\n"
                f"過濾器狀態:\n{fiber_text}\n"
            )

        if self.mode == "unstructured_text":
            return (
                "=== TARGET TEXT FOR EXTRACTION ===\n"
                f"{patient_header}\n"
                f"透析日期：{date}\n\n"
                f"護理紀錄:\n{nurse_text}\n\n"
                f"醫囑:\n{order_text}\n\n"
                f"評估計畫:\n{ap_text}\n\n"
                f"過濾器狀態:\n{fiber_text}\n"
                "=================================="
            )

        # natural_text（預設）
        return (
            f"{patient_header}\n"
            f"透析日期：{date}\n"
            f"--- 護理紀錄 ---\n{nurse_text}\n"
            f"--- 醫囑 ---\n{order_text}\n"
            f"--- 評估計畫 ---\n{ap_text}\n"
            f"--- 過濾器狀態 ---\n{fiber_text}\n"
        )

    def _create_bio_session_document(self, data: dict, date: str) -> Document:
        """將單次透析日資料轉為 LlamaIndex Document，doc_id = hash(case_number + date)。"""
        case_number = str(data.get("case_number", ""))
        doc_id = self._build_context_hash(case_number + date)
        metadata = {
            "case_number": case_number,
            "session_date": date,
            "age": data.get("age", ""),
            "blood_type": str(data.get("blood_type", "") or "") + str(data.get("blood_rh", "") or ""),
            "hbv": data.get("hbv", 0),
            "hcv": data.get("hcv", 0),
            "hiv": data.get("hiv", 0),
            "source": os.path.basename(self.raw_file_path),
            "data_type": self.data_type,
        }
        text = self._format_bio_session_text(data, date)
        hidden_keys = list(metadata.keys())
        return Document(
            text=text,
            metadata=metadata,
            doc_id=doc_id,
            excluded_embed_metadata_keys=hidden_keys,
            excluded_llm_metadata_keys=hidden_keys,
        )

    def _create_bio_document(self, data: dict) -> Document:
        """（向後相容）整患者級 Document，不再由 _process_bio_jsonl 呼叫。"""
        case_number = str(data.get("case_number", ""))
        doc_id = self._build_context_hash(case_number) if case_number else None
        metadata = {
            "case_number": case_number,
            "age": data.get("age", ""),
            "blood_type": str(data.get("blood_type", "") or "") + str(data.get("blood_rh", "") or ""),
            "hbv": data.get("hbv", 0),
            "hcv": data.get("hcv", 0),
            "hiv": data.get("hiv", 0),
            "source": os.path.basename(self.raw_file_path),
            "data_type": self.data_type,
        }
        # 沿用舊格式（整患者所有日期合併）
        all_dates = sorted(self._collect_bio_dates(data))
        texts = [self._format_bio_session_text(data, d) for d in all_dates]
        text = "\n\n".join(texts)
        hidden_keys = list(metadata.keys())
        return Document(
            text=text,
            metadata=metadata,
            doc_id=doc_id,
            excluded_embed_metadata_keys=hidden_keys,
            excluded_llm_metadata_keys=hidden_keys,
        )

    def _process_bio_jsonl(self) -> List[Document]:
        """載入 Bio 透析病歷 JSONL，以透析日為單位建立 Document（每患者每日一筆）。"""
        documents = []
        with open(self.raw_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                dates = self._collect_bio_dates(data)
                for date in sorted(dates):
                    documents.append(self._create_bio_session_document(data, date))
        return documents

    def _format_ultradomain_text(self, context: str, metadata: dict) -> str:
        """UltraDomain 文本預設保留原始 context，減少對既有實驗的干擾。"""
        if self.mode == "markdown":
            domain = metadata.get("domain", "unknown")
            return f"# UltraDomain Document\n\n- Domain: {domain}\n- Doc ID: {metadata['NO']}\n\n{context}"

        if self.mode == "key_value_text":
            domain = metadata.get("domain", "unknown")
            return f"NO: {metadata['NO']}\nDomain: {domain}\nContext:\n{context}"

        if self.mode == "unstructured_text":
            return (
                "=== TARGET TEXT FOR EXTRACTION ===\n"
                f"{context}\n"
                "=================================="
            )

        return context

    def _create_ultradomain_document(self, context: str, index: int) -> Document:
        """將 Step 0 unique contexts 轉為統一 Document。"""
        context = str(context).strip()
        if not context:
            raise ValueError(f"第 {index} 筆 context 為空字串")

        doc_id = self._build_context_hash(context)
        dataset_config = self.data_config.get_dataset_config(self.data_type)
        metadata = {
            "NO": doc_id,
            "doc_id": doc_id,
            "original_doc_id": doc_id,
            "domain": dataset_config.get("domain", self.data_type),
            "source": os.path.basename(self.raw_file_path),
            "data_type": self.data_type,
        }
        text = self._format_ultradomain_text(context, metadata)
        hidden_metadata_keys = list(metadata.keys())
        return Document(
            text=text,
            metadata=metadata,
            doc_id=doc_id,
            excluded_embed_metadata_keys=hidden_metadata_keys,
            excluded_llm_metadata_keys=hidden_metadata_keys,
        )

    def _process_ultradomain_contexts(self) -> List[Document]:
        """處理 UltraDomain unique contexts JSON。"""
        with open(self.raw_file_path, "r", encoding="utf-8") as f:
            contexts = json.load(f)

        if not isinstance(contexts, list):
            raise ValueError(f"UltraDomain 文件格式錯誤，應為 list: {self.raw_file_path}")

        documents = []
        for idx, context in enumerate(contexts, start=1):
            if not isinstance(context, str):
                raise ValueError(f"第 {idx} 筆 context 不是字串: {type(context)}")
            documents.append(self._create_ultradomain_document(context, idx))
        return documents
    
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

        if self.document_loader == "ultradomain_contexts_json":
            return self._process_ultradomain_contexts()

        if self.document_loader == "bio_jsonl":
            return self._process_bio_jsonl()

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
