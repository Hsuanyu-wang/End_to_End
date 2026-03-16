# graph_builder/__init__.py
from .baseline_builder import BaselineGraphBuilder
from .ontology_builder import OntologyGraphBuilder

# 註冊表：未來有新方法，只需在此新增一行
BUILDER_REGISTRY = {
    "baseline": BaselineGraphBuilder,
    "ontology_learning": OntologyGraphBuilder
}

def get_graph_builder(method_name: str, graph_store, settings):
    if method_name not in BUILDER_REGISTRY:
        raise ValueError(f"未知的建圖方法: {method_name}。支援的方法: {list(BUILDER_REGISTRY.keys())}")
    
    # 實例化並注入相依元件
    builder_class = BUILDER_REGISTRY[method_name]
    return builder_class(graph_store, settings)