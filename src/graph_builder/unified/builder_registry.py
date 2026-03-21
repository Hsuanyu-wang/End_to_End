"""
Graph Builder 註冊器

提供 Plugin Registry System 讓 Builder 可動態註冊和發現
"""

from typing import Dict, Type, List
from src.graph_builder.base_builder import BaseGraphBuilder


class GraphBuilderRegistry:
    """
    Graph Builder 註冊器
    
    支援動態註冊和發現不同的建圖方法
    """
    
    _builders: Dict[str, Type[BaseGraphBuilder]] = {}
    
    @classmethod
    def register(cls, name: str, builder_class: Type[BaseGraphBuilder]):
        """
        註冊 builder
        
        Args:
            name: Builder 名稱（用於識別）
            builder_class: Builder 類別（須繼承 BaseGraphBuilder）
        """
        if not issubclass(builder_class, BaseGraphBuilder):
            raise ValueError(f"Builder 類別必須繼承 BaseGraphBuilder，收到: {builder_class}")
        
        cls._builders[name] = builder_class
        print(f"✅ 註冊 Builder: {name} -> {builder_class.__name__}")
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseGraphBuilder:
        """
        根據名稱建立 builder 實例
        
        Args:
            name: Builder 名稱
            **kwargs: Builder 初始化參數
        
        Returns:
            Builder 實例
        """
        if name not in cls._builders:
            available = list(cls._builders.keys())
            raise ValueError(f"未知的 Builder: {name}。可用: {available}")
        
        builder_class = cls._builders[name]
        return builder_class(**kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """列出所有可用的 builder"""
        return list(cls._builders.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """檢查 builder 是否已註冊"""
        return name in cls._builders


# 自動註冊內建 builders
def _register_builtin_builders():
    """註冊內建的 builders"""
    try:
        # PropertyGraph Builder（特殊處理，需要 ExtractorFactory）
        # 暫時不註冊，等 PropertyGraphBuilder 完成後再註冊
        pass
    except ImportError:
        print("⚠️  PropertyGraph Builder 註冊失敗（可能缺少依賴）")
    
    try:
        from src.graph_builder.lightrag_builder import LightRAGBuilder
        GraphBuilderRegistry.register("lightrag", LightRAGBuilder)
    except ImportError:
        print("⚠️  LightRAG Builder 註冊失敗（可能缺少依賴）")
    
    try:
        from src.graph_builder.autoschema_builder import AutoSchemaKGBuilder
        GraphBuilderRegistry.register("autoschema", AutoSchemaKGBuilder)
    except ImportError:
        print("⚠️  AutoSchemaKG Builder 註冊失敗（可能缺少依賴）")
    
    try:
        from src.graph_builder.ontology_builder import OntologyGraphBuilder
        GraphBuilderRegistry.register("ontology", OntologyGraphBuilder)
    except ImportError:
        print("⚠️  Ontology Builder 註冊失敗（可能缺少依賴）")
    
    try:
        from src.graph_builder.baseline_builder import BaselineGraphBuilder
        GraphBuilderRegistry.register("baseline", BaselineGraphBuilder)
    except ImportError:
        print("⚠️  Baseline Builder 註冊失敗（可能缺少依賴）")


# 初始化時自動註冊
_register_builtin_builders()
