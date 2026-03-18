"""
插件系統模組

提供可擴展的 KG 建構增強插件機制
"""

from .base import BaseKGPlugin
from .registry import PluginRegistry, register_plugin, get_plugin, list_available_plugins

__all__ = [
    "BaseKGPlugin",
    "PluginRegistry",
    "register_plugin",
    "get_plugin",
    "list_available_plugins",
]
