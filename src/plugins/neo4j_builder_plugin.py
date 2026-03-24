"""
Neo4j LLM Graph Builder 插件（未實作 Stub）

DEPRECATED: 此插件尚未實作（enabled=False），僅保留檔案避免 import 錯誤。
"""

from src.plugins.base import BaseKGPlugin
from src.plugins.registry import register_plugin


@register_plugin("neo4j", enabled=False)
class Neo4jBuilderPlugin(BaseKGPlugin):
    """Neo4j 圖資料庫與圖演算法插件（未實作）"""

    def get_name(self) -> str:
        return "Neo4j"

    def get_description(self) -> str:
        return "Neo4j 圖資料庫與圖演算法插件（未實作）"
