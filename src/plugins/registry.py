"""
插件註冊機制

提供插件的註冊、查詢和管理功能
"""

from typing import Dict, List, Type, Optional
from .base import BaseKGPlugin


class PluginRegistry:
    """
    插件註冊器
    
    管理所有可用的插件，支援 enabled/disabled 狀態
    """
    
    _plugins: Dict[str, Type[BaseKGPlugin]] = {}
    _instances: Dict[str, BaseKGPlugin] = {}
    _enabled: Dict[str, bool] = {}
    
    @classmethod
    def register(cls, name: str, plugin_class: Type[BaseKGPlugin], enabled: bool = True):
        """
        註冊插件
        
        Args:
            name: 插件名稱
            plugin_class: 插件類別
            enabled: 是否啟用（未實作的 stub plugin 應設為 False）
        """
        if not issubclass(plugin_class, BaseKGPlugin):
            raise TypeError(f"{plugin_class} 必須繼承 BaseKGPlugin")
        
        cls._plugins[name] = plugin_class
        cls._enabled[name] = enabled
        status = "啟用" if enabled else "停用（未實作）"
        print(f"✅ 已註冊插件: {name} [{status}]")
    
    @classmethod
    def get(cls, name: str, config: Optional[Dict] = None) -> Optional[BaseKGPlugin]:
        """
        取得插件實例
        
        Args:
            name: 插件名稱
            config: 插件配置
        
        Returns:
            插件實例，若不存在或未啟用則返回 None
        """
        if name not in cls._plugins:
            print(f"⚠️  插件不存在: {name}")
            return None
        
        if not cls._enabled.get(name, True):
            print(f"⚠️  插件 '{name}' 已註冊但尚未實作（disabled），略過")
            return None
        
        if name in cls._instances:
            return cls._instances[name]
        
        plugin_class = cls._plugins[name]
        instance = plugin_class()
        instance.initialize(config)
        cls._instances[name] = instance
        
        return instance
    
    @classmethod
    def is_enabled(cls, name: str) -> bool:
        """檢查插件是否已啟用"""
        return cls._enabled.get(name, False)
    
    @classmethod
    def list_all(cls) -> List[str]:
        """列出所有已註冊的插件名稱"""
        return list(cls._plugins.keys())
    
    @classmethod
    def list_enabled(cls) -> List[str]:
        """列出所有已啟用的插件名稱"""
        return [name for name, enabled in cls._enabled.items() if enabled]
    
    @classmethod
    def clear(cls):
        """清空所有註冊的插件"""
        for instance in cls._instances.values():
            instance.cleanup()
        
        cls._plugins = {}
        cls._instances = {}
        cls._enabled = {}
    
    @classmethod
    def unregister(cls, name: str):
        """
        取消註冊插件
        
        Args:
            name: 插件名稱
        """
        if name in cls._instances:
            cls._instances[name].cleanup()
            del cls._instances[name]
        
        if name in cls._plugins:
            del cls._plugins[name]
            cls._enabled.pop(name, None)
            print(f"✅ 已取消註冊插件: {name}")


# 便捷函數
def register_plugin(name: str, enabled: bool = True):
    """
    插件註冊裝飾器
    
    使用方式:
    ```python
    @register_plugin("my_plugin")
    class MyPlugin(BaseKGPlugin):
        ...

    @register_plugin("stub_plugin", enabled=False)
    class StubPlugin(BaseKGPlugin):
        ...
    ```
    
    Args:
        name: 插件名稱
        enabled: 是否啟用（未實作的 stub 設為 False）
    """
    def decorator(plugin_class: Type[BaseKGPlugin]):
        PluginRegistry.register(name, plugin_class, enabled=enabled)
        return plugin_class
    return decorator


def get_plugin(name: str, config: Optional[Dict] = None) -> Optional[BaseKGPlugin]:
    """
    取得插件實例的便捷函數
    
    Args:
        name: 插件名稱
        config: 插件配置
    
    Returns:
        插件實例
    """
    return PluginRegistry.get(name, config)


def list_available_plugins() -> List[str]:
    """
    列出所有可用插件的便捷函數
    
    Returns:
        插件名稱列表
    """
    return PluginRegistry.list_all()
