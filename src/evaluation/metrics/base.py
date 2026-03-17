"""
評估指標基底類別

定義所有評估指標的共用介面與註冊機制
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union


class BaseMetric(ABC):
    """
    評估指標抽象基底類別
    
    所有評估指標都應繼承此類別並實作 compute 方法
    
    Attributes:
        name: 指標名稱
        description: 指標描述
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        初始化指標
        
        Args:
            name: 指標名稱
            description: 指標描述
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def compute(self, **kwargs) -> Union[float, Dict[str, float]]:
        """
        計算指標值
        
        Args:
            **kwargs: 計算所需的參數
        
        Returns:
            指標值（單一數值或字典）
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class MetricRegistry:
    """
    指標註冊表
    
    提供指標註冊、查詢與管理功能
    """
    
    _registry: Dict[str, BaseMetric] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        註冊指標的裝飾器
        
        Args:
            name: 指標名稱
        
        Returns:
            裝飾器函數
        
        Example:
            @MetricRegistry.register("custom_f1")
            class CustomF1Metric(BaseMetric):
                def compute(self, **kwargs):
                    # 實作邏輯
                    pass
        """
        def decorator(metric_class):
            if not issubclass(metric_class, BaseMetric):
                raise TypeError(f"{metric_class} 必須繼承 BaseMetric")
            
            cls._registry[name] = metric_class
            return metric_class
        
        return decorator
    
    @classmethod
    def get(cls, name: str) -> BaseMetric:
        """
        取得指定名稱的指標
        
        Args:
            name: 指標名稱
        
        Returns:
            指標實例
        
        Raises:
            KeyError: 當指標不存在時
        """
        if name not in cls._registry:
            raise KeyError(f"找不到指標: {name}")
        
        metric_class = cls._registry[name]
        return metric_class(name=name)
    
    @classmethod
    def get_all(cls) -> Dict[str, BaseMetric]:
        """
        取得所有已註冊的指標
        
        Returns:
            指標字典 {name: metric_instance}
        """
        return {name: cls.get(name) for name in cls._registry.keys()}
    
    @classmethod
    def list_names(cls) -> List[str]:
        """
        列出所有已註冊的指標名稱
        
        Returns:
            指標名稱列表
        """
        return list(cls._registry.keys())
    
    @classmethod
    def clear(cls):
        """清空註冊表（主要用於測試）"""
        cls._registry.clear()
