"""
配置管理套件

提供專案所需的所有配置管理功能

使用方式：
    from src.config import my_settings
    
    # 訪問模型
    llm = my_settings.llm
    
    # 訪問資料路徑
    qa_path = my_settings.data_config.qa_file_path_DI
    
    # 訪問 LightRAG 配置
    entity_types = my_settings.lightrag_config.entity_types
"""

from .settings import ModelSettings, DataConfig, LightRAGConfig, get_settings

# 建立全域單例實例
my_settings = get_settings()

__all__ = ["ModelSettings", "DataConfig", "LightRAGConfig", "get_settings", "my_settings"]
