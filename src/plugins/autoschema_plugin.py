"""
AutoSchemaKG 插件（未實作 Stub）

DEPRECATED: 此插件尚未實作（enabled=False），僅保留檔案避免 import 錯誤。
AutoSchema 功能請改用 --graph_type autoschema。
"""

from src.plugins.base import BaseKGPlugin
from src.plugins.registry import register_plugin


@register_plugin("autoschema", enabled=False)
class AutoSchemaKGPlugin(BaseKGPlugin):
    """AutoSchemaKG 插件（未實作）"""

    def get_name(self) -> str:
        return "AutoSchemaKG"

    def get_description(self) -> str:
        return "AutoSchemaKG 插件（未實作）"
