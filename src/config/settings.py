"""
配置管理模組

此模組負責載入與管理所有專案配置，包括：
- LLM 模型設定（主模型、評估模型、建圖模型）
- Embedding 模型設定
- 資料路徑配置
- LightRAG 特定配置
"""

from typing import Optional
from pathlib import Path
import yaml
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


class ModelSettings:
    """
    模型設定管理類別
    
    封裝 LlamaIndex Settings 並提供額外的配置管理功能
    
    Attributes:
        llm: 主要 LLM 模型
        eval_llm: 評估用 LLM 模型
        builder_llm: 圖譜建構用 LLM 模型
        embed_model: Embedding 模型
        lightrag_storage_path_DIR: LightRAG 儲存路徑
        lightrag_language: LightRAG 語言設定
        lightrag_entity_types: LightRAG 實體類型列表
        raw_file_path_DI: DI 原始資料路徑
        raw_file_path_GEN: GEN 原始資料路徑
        qa_file_path_DI: DI 問答資料路徑
        qa_file_path_GEN: GEN 問答資料路徑
    """
    
    def __init__(self, config_path: str = "/home/End_to_End_RAG/config.yml"):
        """
        初始化模型設定
        
        Args:
            config_path: 配置檔案路徑，預設為專案根目錄的 config.yml
        """
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._initialize_settings()
    
    def _load_config(self) -> dict:
        """載入 YAML 配置檔案"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"找不到配置檔案: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _initialize_settings(self):
        """初始化所有模型與配置"""
        # 主要 LLM 模型
        Settings.llm = Ollama(
            model=self._config["model"]["llm_model"], 
            base_url=self._config["model"]["ollama_url"],
            request_timeout=300.0
        )
        
        # 評估用 LLM 模型
        Settings.eval_llm = Ollama(
            model=self._config["eval_model"]["llm_model"], 
            base_url=self._config["eval_model"]["ollama_url"],
            request_timeout=300.0
        )
        
        # 建圖用 LLM 模型
        Settings.builder_llm = Ollama(
            model=self._config["builder_model"]["llm_model"], 
            base_url=self._config["builder_model"]["ollama_url"],
            request_timeout=300.0
        )
        
        # Embedding 模型
        Settings.embed_model = OllamaEmbedding(
            model_name=self._config["model"]["embed_model"],
            base_url=self._config["model"]["ollama_url"],
            embed_batch_size=1,
            embed_batch_timeout=300.0,
        )
        
        # LightRAG 配置
        Settings.lightrag_storage_path_DIR = self._config["lightrag"]["lightrag_storage_path_DIR"]
        Settings.lightrag_language = self._config["lightrag"]["lightrag_language"]
        Settings.lightrag_entity_types = self._config["lightrag"]["lightrag_entity_types"]
        
        # 資料路徑配置
        Settings.raw_file_path_DI = self._config["data"]["raw_file_path_DI"]
        Settings.raw_file_path_GEN = self._config["data"]["raw_file_path_GEN"]
        Settings.qa_file_path_DI = self._config["data"]["qa_file_path_DI"]
        Settings.qa_file_path_GEN = self._config["data"]["qa_file_path_GEN"]
    
    @property
    def llm(self):
        """取得主要 LLM 模型"""
        return Settings.llm
    
    @property
    def eval_llm(self):
        """取得評估用 LLM 模型"""
        return Settings.eval_llm
    
    @property
    def builder_llm(self):
        """取得建圖用 LLM 模型"""
        return Settings.builder_llm
    
    @property
    def embed_model(self):
        """取得 Embedding 模型"""
        return Settings.embed_model
    
    def get_settings_object(self):
        """
        取得 LlamaIndex Settings 物件（向後兼容）
        
        Returns:
            Settings 物件
        """
        return Settings


def get_settings(model_type: str = "small", config_path: Optional[str] = None) -> Settings:
    """
    取得模型設定（向後兼容函數）
    
    Args:
        model_type: 模型類型（'small' 或 'big'），目前未使用但保留介面
        config_path: 配置檔案路徑，預設為 None 使用預設路徑
    
    Returns:
        LlamaIndex Settings 物件
    
    Raises:
        ValueError: 當 model_type 不在允許範圍時
    """
    if model_type not in ["small", "big"]:
        raise ValueError("model_type must be either 'small' or 'big'")
    
    if config_path is None:
        config_path = "/home/End_to_End_RAG/config.yml"
    
    settings_manager = ModelSettings(config_path=config_path)
    return settings_manager.get_settings_object()
