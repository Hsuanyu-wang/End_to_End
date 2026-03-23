"""
相似實體合併 plugin 註冊（實際合併邏輯在 src.rag.plugins.lightrag_similar_entity_merge）。
"""

from src.plugins.base import BaseKGPlugin
from src.plugins.registry import register_plugin


@register_plugin("similar_entity_merge")
class SimilarEntityMergePlugin(BaseKGPlugin):
    """佔位註冊：供 --lightrag_plugins 列出；執行由 run_evaluation 呼叫合併模組。"""

    def get_name(self) -> str:
        return "similar_entity_merge"

    def get_description(self) -> str:
        return (
            "LightRAG 建圖後依 embedding 相似度合併實體（獨立 storage，"
            "見 docs/LIGHTRAG_ENTITY_MERGE.md）"
        )
