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

from typing import Optional, List
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
    
    def __init__(self, config_path: str = "/home/End_to_End_RAG/config.yml"):
        """
        初始化模型設定
        
        Args:
            config_path: 配置檔案路徑，預設為專案根目錄的 config.yml
        """
        self.config_path = Path(config_path)
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
        # 主要 LLM 模型
        LlamaSettings.llm = Ollama(
            model=self._config["model"]["llm_model"], 
            base_url=self._config["model"]["ollama_url"],
            request_timeout=300.0
        )
        
        # 評估用 LLM 模型（自訂擴充）
        LlamaSettings.eval_llm = Ollama(
            model=self._config["eval_model"]["llm_model"], 
            base_url=self._config["eval_model"]["ollama_url"],
            request_timeout=300.0
        )
        
        # 建圖用 LLM 模型（自訂擴充）
        LlamaSettings.builder_llm = Ollama(
            model=self._config["builder_model"]["llm_model"], 
            base_url=self._config["builder_model"]["ollama_url"],
            request_timeout=300.0
        )
        
        # Embedding 模型
        LlamaSettings.embed_model = OllamaEmbedding(
            model_name=self._config["model"]["embed_model"],
            base_url=self._config["model"]["ollama_url"],
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


def get_settings(model_type: str = "small", config_path: Optional[str] = None) -> ModelSettings:
    """
    取得模型設定（單例模式）
    
    Args:
        model_type: 模型類型（'small' 或 'big'），目前未使用但保留介面
        config_path: 配置檔案路徑，預設為 None 使用預設路徑
    
    Returns:
        ModelSettings 實例（包含 llm, eval_llm, builder_llm, embed_model, data_config, lightrag_config）
    
    Raises:
        ValueError: 當 model_type 不在允許範圍時
    """
    global _settings_instance
    
    if model_type not in ["small", "big"]:
        raise ValueError("model_type must be either 'small' or 'big'")
    
    if _settings_instance is None:
        if config_path is None:
            config_path = "/home/End_to_End_RAG/config.yml"
        _settings_instance = ModelSettings(config_path=config_path)
    
    return _settings_instance
