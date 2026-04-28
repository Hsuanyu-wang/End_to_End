"""
配置管理模組

此模組負責載入與管理所有專案配置，包括：
- LLM 模型設定（主模型、評估模型、建圖模型）
- Embedding 模型設定
- 資料路徑配置
- LightRAG 特定配置

注意：本模組使用命名空間方式區分配置來源：
- LlamaSettings：LlamaIndex 的全域 Settings 物件
- ModelSettings/DataConfig/LightRAGConfig：我們的自訂配置類別
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import yaml
from llama_index.core import Settings as LlamaSettings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


class DataConfig:
    """
    資料路徑配置管理類別
    
    管理所有與資料檔案路徑相關的配置
    
    Attributes:
        raw_file_path_DI: DI 原始資料路徑
        raw_file_path_GEN: GEN 原始資料路徑
        qa_file_path_DI: DI 問答資料路徑
        qa_file_path_GEN: GEN 問答資料路徑
    """
    
    def __init__(self, config: dict):
        """
        初始化資料配置
        
        Args:
            config: 從 YAML 載入的配置字典
        """
        data_config = config.get("data", {})
        self.raw_file_path_DI = data_config.get("raw_file_path_DI", "")
        self.raw_file_path_GEN = data_config.get("raw_file_path_GEN", "")
        self.qa_file_path_DI = data_config.get("qa_file_path_DI", "")
        self.qa_file_path_GEN = data_config.get("qa_file_path_GEN", "")
        self.datasets = data_config.get("datasets", {})

    def get_dataset_config(self, data_type: str) -> Dict[str, Any]:
        """取得指定資料類型的完整設定，保留 DI/GEN 舊欄位相容。"""
        legacy_defaults = {
            "DI": {
                "document_file_path": self.raw_file_path_DI,
                "qa_file_path": self.qa_file_path_DI,
                "document_loader": "csr_raw_jsonl",
                "qa_loader": "csr_csv",
            },
            "GEN": {
                "document_file_path": self.raw_file_path_GEN,
                "qa_file_path": self.qa_file_path_GEN,
                "document_loader": "csr_raw_jsonl",
                "qa_loader": "csr_jsonl",
            },
        }

        merged: Dict[str, Any] = {}
        if data_type in legacy_defaults:
            merged.update(legacy_defaults[data_type])

        custom_config = self.datasets.get(data_type, {})
        if not isinstance(custom_config, dict):
            raise ValueError(f"data.datasets.{data_type} 必須為 dict")
        merged.update(custom_config)
        return merged

    def get_document_file_path(self, data_type: str) -> str:
        """取得文件語料路徑。"""
        dataset_config = self.get_dataset_config(data_type)
        return (
            dataset_config.get("document_file_path")
            or dataset_config.get("raw_file_path")
            or dataset_config.get("contexts_file_path")
            or ""
        )

    def get_qa_file_path(self, data_type: str) -> str:
        """取得 QA 檔路徑。"""
        dataset_config = self.get_dataset_config(data_type)
        return dataset_config.get("qa_file_path", "")

    def get_document_loader(self, data_type: str) -> str:
        """取得文件載入器類型。"""
        dataset_config = self.get_dataset_config(data_type)
        return dataset_config.get("document_loader", "csr_raw_jsonl")

    def get_qa_loader(self, data_type: str) -> str:
        """取得 QA 載入器類型。"""
        dataset_config = self.get_dataset_config(data_type)
        return dataset_config.get("qa_loader", "csr_jsonl")

    def get_source_raw_file_path(self, data_type: str) -> str:
        """取得原始來源資料路徑，供額外資料生成腳本使用。"""
        dataset_config = self.get_dataset_config(data_type)
        return (
            dataset_config.get("source_raw_file_path")
            or dataset_config.get("raw_file_path")
            or dataset_config.get("document_file_path")
            or ""
        )


class LightRAGConfig:
    """
    LightRAG 專用配置管理類別
    
    管理所有與 LightRAG 相關的配置
    
    Attributes:
        storage_path_DIR: LightRAG 儲存路徑
        language: LightRAG 語言設定
        entity_types: LightRAG 實體類型列表
    """
    
    def __init__(self, config: dict):
        """
        初始化 LightRAG 配置
        
        Args:
            config: 從 YAML 載入的配置字典
        """
        lightrag_config = config.get("lightrag", {})
        self.storage_path_DIR = lightrag_config.get("lightrag_storage_path_DIR", "")
        self.language = lightrag_config.get("lightrag_language", "Chinese")
        self.entity_types = lightrag_config.get("lightrag_entity_types", [])
        # 送入 Ollama / 向量庫嵌入前之字元截斷上限（中文 token 密度高，預設偏保守）
        self.embed_max_input_chars = int(lightrag_config.get("embed_max_input_chars", 2560))
        # 相似實體合併 plugin（見 docs/LIGHTRAG_ENTITY_MERGE.md）
        _plugins = lightrag_config.get("plugins") or {}
        _sem = _plugins.get("similar_entity_merge") or {}
        self.similar_entity_merge_threshold = float(_sem.get("threshold", 0.8))
        _raw_tmax = _sem.get("threshold_max")
        self.similar_entity_merge_threshold_max = (
            None if _raw_tmax is None else float(_raw_tmax)
        )
        self.similar_entity_merge_text_mode = str(_sem.get("text_mode", "name"))
        self.similar_entity_merge_force_recopy = bool(_sem.get("force_recopy", False))
        self.similar_entity_merge_dry_run = bool(_sem.get("dry_run", False))
        if self.similar_entity_merge_threshold_max is not None:
            if self.similar_entity_merge_threshold_max <= self.similar_entity_merge_threshold:
                raise ValueError(
                    "config.yml lightrag.plugins.similar_entity_merge："
                    "threshold_max 必須大於 threshold（上界不含）"
                )


class ModelSettings:
    """
    模型設定管理類別
    
    封裝 LlamaIndex Settings 並提供額外的配置管理功能
    
    Attributes:
        llm: 主要 LLM 模型
        eval_llm: 評估用 LLM 模型
        builder_llm: 圖譜建構用 LLM 模型
        embed_model: Embedding 模型
        data_config: 資料路徑配置物件
        lightrag_config: LightRAG 配置物件
    """
    
    def __init__(
        self,
        config_path: str = "/home/End_to_End_RAG/config.yml",
        ollama_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        eval_llm_model: Optional[str] = None,
        builder_llm_model: Optional[str] = None,
    ):
        """
        初始化模型設定
        
        Args:
            config_path: 配置檔案路徑，預設為專案根目錄的 config.yml
            ollama_url: （可選）覆蓋三個角色的 base_url
            llm_model: （可選）覆蓋 model.llm_model
            eval_llm_model: （可選）覆蓋 eval_model.llm_model
            builder_llm_model: （可選）覆蓋 builder_model.llm_model
        """
        self.config_path = Path(config_path)
        self._override_ollama_url = ollama_url
        self._override_llm_model = llm_model
        self._override_eval_llm_model = eval_llm_model
        self._override_builder_llm_model = builder_llm_model
        self._config = self._load_config()
        
        self.data_config = DataConfig(self._config)
        self.lightrag_config = LightRAGConfig(self._config)
        
        self._initialize_llama_settings()
    
    def _load_config(self) -> dict:
        """載入 YAML 配置檔案"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"找不到配置檔案: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _initialize_llama_settings(self):
        """
        初始化 LlamaIndex Settings 全域物件的標準屬性
        
        注意：只設定 LlamaIndex 官方支援的屬性，不再污染全域物件
        """
        # 決定 base_url / model（支援 CLI override）
        main_base_url = (
            self._override_ollama_url
            if self._override_ollama_url is not None
            else self._config["model"]["ollama_url"]
        )
        eval_base_url = (
            self._override_ollama_url
            if self._override_ollama_url is not None
            else self._config["eval_model"]["ollama_url"]
        )
        builder_base_url = (
            self._override_ollama_url
            if self._override_ollama_url is not None
            else self._config["builder_model"]["ollama_url"]
        )

        main_model_name = (
            self._override_llm_model
            if self._override_llm_model is not None
            else self._config["model"]["llm_model"]
        )
        eval_model_name = (
            self._override_eval_llm_model
            if self._override_eval_llm_model is not None
            else self._config["eval_model"]["llm_model"]
        )
        builder_model_name = (
            self._override_builder_llm_model
            if self._override_builder_llm_model is not None
            else self._config["builder_model"]["llm_model"]
        )

        # 主要 LLM 模型
        LlamaSettings.llm = Ollama(
            model=main_model_name,
            base_url=main_base_url,
            request_timeout=300.0
        )
        
        # 評估用 LLM 模型（自訂擴充）
        LlamaSettings.eval_llm = Ollama(
            model=eval_model_name,
            base_url=eval_base_url,
            request_timeout=300.0
        )
        
        # 建圖用 LLM 模型（自訂擴充）
        LlamaSettings.builder_llm = Ollama(
            model=builder_model_name,
            base_url=builder_base_url,
            request_timeout=300.0
        )
        
        # Embedding 模型
        LlamaSettings.embed_model = OllamaEmbedding(
            model_name=self._config["model"]["embed_model"],
            # embedding model 名稱沿用 config.yml，base_url 受 ollama_url override 影響
            base_url=main_base_url,
            embed_batch_size=1,
            embed_batch_timeout=300.0,
        )
    
    @property
    def llm(self):
        """取得主要 LLM 模型"""
        return LlamaSettings.llm
    
    @property
    def eval_llm(self):
        """取得評估用 LLM 模型"""
        return LlamaSettings.eval_llm
    
    @property
    def builder_llm(self):
        """取得建圖用 LLM 模型"""
        return LlamaSettings.builder_llm
    
    @property
    def embed_model(self):
        """取得 Embedding 模型"""
        return LlamaSettings.embed_model
    
    def get_llama_settings(self):
        """
        取得 LlamaIndex Settings 物件（向後兼容）
        
        Returns:
            LlamaIndex Settings 物件
        """
        return LlamaSettings


# 全域單例實例
_settings_instance: Optional[ModelSettings] = None


def get_settings(
    model_type: str = "small",
    config_path: Optional[str] = None,
    ollama_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    eval_llm_model: Optional[str] = None,
    builder_llm_model: Optional[str] = None,
) -> ModelSettings:
    """
    取得模型設定（單例模式）
    
    Args:
        model_type: 模型類型（'small' 或 'big'），目前未使用但保留介面
        config_path: 配置檔案路徑，預設為 None 使用預設路徑
        ollama_url: （可選）覆蓋 model/eval_model/builder_model 的 base_url
        llm_model: （可選）覆蓋 model.llm_model
        eval_llm_model: （可選）覆蓋 eval_model.llm_model
        builder_llm_model: （可選）覆蓋 builder_model.llm_model
    
    Returns:
        ModelSettings 實例（包含 llm, eval_llm, builder_llm, embed_model, data_config, lightrag_config）
    
    Raises:
        ValueError: 當 model_type 不在允許範圍時
    """
    global _settings_instance
    
    if model_type not in ["small", "big"]:
        raise ValueError("model_type must be either 'small' or 'big'")

    if config_path is None:
        config_path = "/home/End_to_End_RAG/config.yml"

    need_override = any(
        v is not None for v in (ollama_url, llm_model, eval_llm_model, builder_llm_model)
    )

    # 有覆蓋時重建 singleton；確保後續 wrapper 呼叫 get_settings() 時拿到相同 LlamaIndex Settings
    if need_override:
        _settings_instance = ModelSettings(
            config_path=config_path,
            ollama_url=ollama_url,
            llm_model=llm_model,
            eval_llm_model=eval_llm_model,
            builder_llm_model=builder_llm_model,
        )
        return _settings_instance

    if _settings_instance is None:
        _settings_instance = ModelSettings(config_path=config_path)

    return _settings_instance

