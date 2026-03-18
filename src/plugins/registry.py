"""
插件註冊機制

提供插件的註冊、查詢和管理功能
"""

from typing import Dict, List, Type, Optional
from .base import BaseKGPlugin


class PluginRegistry:
    """
    插件註冊器
    
    管理所有可用的插件
    """
    
    _plugins: Dict[str, Type[BaseKGPlugin]] = {}
    _instances: Dict[str, BaseKGPlugin] = {}
    
    @classmethod
    def register(cls, name: str, plugin_class: Type[BaseKGPlugin]):
        """
        註冊插件
        
        Args:
            name: 插件名稱
            plugin_class: 插件類別
        """
        if not issubclass(plugin_class, BaseKGPlugin):
            raise TypeError(f"{plugin_class} 必須繼承 BaseKGPlugin")
        
        cls._plugins[name] = plugin_class
        print(f"✅ 已註冊插件: {name}")
    
    @classmethod
    def get(cls, name: str, config: Optional[Dict] = None) -> Optional[BaseKGPlugin]:
        """
        取得插件實例
        
        Args:
            name: 插件名稱
            config: 插件配置
        
        Returns:
            插件實例，若不存在則返回 None
        """
        if name not in cls._plugins:
            print(f"⚠️  插件不存在: {name}")
            return None
        
        # 檢查是否已有實例
        if name in cls._instances:
            return cls._instances[name]
        
        # 建立新實例
        plugin_class = cls._plugins[name]
        instance = plugin_class()
        instance.initialize(config)
        cls._instances[name] = instance
        
        return instance
    
    @classmethod
    def list_all(cls) -> List[str]:
        """
        列出所有已註冊的插件名稱
        
        Returns:
            插件名稱列表
        """
        return list(cls._plugins.keys())
    
    @classmethod
    def clear(cls):
        """清空所有註冊的插件"""
        # 清理所有實例
        for instance in cls._instances.values():
            instance.cleanup()
        
        cls._plugins = {}
        cls._instances = {}
    
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
            print(f"✅ 已取消註冊插件: {name}")


# 便捷函數
def register_plugin(name: str):
    """
    插件註冊裝飾器
    
    使用方式:
    ```python
    @register_plugin("my_plugin")
    class MyPlugin(BaseKGPlugin):
        ...
    ```
    
    Args:
        name: 插件名稱
    """
    def decorator(plugin_class: Type[BaseKGPlugin]):
        PluginRegistry.register(name, plugin_class)
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
