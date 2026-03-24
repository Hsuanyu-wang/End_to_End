"""
Graphiti 時序知識圖譜插件（未實作 Stub）

DEPRECATED: 此插件尚未實作（enabled=False），僅保留檔案避免 import 錯誤。
時序功能請改用 --plugin_temporal。
"""

from src.plugins.base import BaseKGPlugin
from src.plugins.registry import register_plugin


@register_plugin("graphiti", enabled=False)
class GraphitiPlugin(BaseKGPlugin):
    """Graphiti 時序知識圖譜插件（未實作）"""

    def get_name(self) -> str:
        return "Graphiti"

    def get_description(self) -> str:
        return "Graphiti 時序知識圖譜插件（未實作）"
