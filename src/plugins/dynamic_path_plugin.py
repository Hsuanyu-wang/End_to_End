"""
DynamicLLMPathExtractor 插件

基於 LlamaIndex DynamicLLMPathExtractor 的動態實體類型檢測插件
"""

from typing import List, Dict, Any
from src.plugins.base import BaseKGPlugin
from src.plugins.registry import register_plugin


@register_plugin("dynamic_path")
class DynamicPathPlugin(BaseKGPlugin):
    """
    DynamicLLMPathExtractor 插件
    
    核心功能：
    - 動態實體類型檢測（不限於預定義類型）
    - LLM 推斷實體和關係
    - 靈活 Schema（可擴展初始本體）
    
    參考：LlamaIndex 官方文檔
    """
    
    def get_name(self) -> str:
        return "DynamicPath"
    
    def get_description(self) -> str:
        return "動態實體類型檢測與靈活 Schema 擴展插件"
    
    def enhance_schema(
        self,
        text_corpus: List[str],
        base_schema: List[str],
        **kwargs
    ) -> List[str]:
        """
        從初始 Schema 開始，允許擴展
        
        TODO: 實作 optional ontology 機制
        """
        print(f"🔍 [DynamicPath] 啟用動態 Schema 擴展...")
        return base_schema
    
    def enhance_extraction(
        self,
        text: str,
        base_triples: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        使用 LLM 動態推斷三元組，不受 Schema 限制
        
        TODO: 整合 LlamaIndex DynamicLLMPathExtractor
        """
        return base_triples
    
    def post_process_graph(
        self,
        graph_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        合併推斷的新類型到 Schema
        """
        return graph_data
