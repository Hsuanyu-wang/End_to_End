"""LightRAG 相關可選插件（後處理、實驗用）。"""

from src.rag.plugins.lightrag_similar_entity_merge import (
    build_simmerge_custom_tag,
    ensure_similar_merged_lightrag_index,
)

__all__ = [
    "build_simmerge_custom_tag",
    "ensure_similar_merged_lightrag_index",
]
