"""
CQ-Driven（Competency Question）插件（未實作 Stub）

DEPRECATED: 此插件尚未實作（enabled=False），僅保留檔案避免 import 錯誤。
"""

from src.plugins.base import BaseKGPlugin
from src.plugins.registry import register_plugin


@register_plugin("cq_driven", enabled=False)
class CQDrivenPlugin(BaseKGPlugin):
    """CQ-Driven 知識圖譜插件（未實作）"""

    def get_name(self) -> str:
        return "CQ-Driven"

    def get_description(self) -> str:
        return "CQ-Driven 知識圖譜插件（未實作）"
