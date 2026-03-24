"""
Graph Traversal Strategies

可插拔的 graph traversal 策略：k_hop, ppr, pcst, tog, anchor_hybrid_khop。
import 此 package 時自動註冊所有內建策略。
"""

from src.graph_retriever.strategies.base import (
    BaseTraversalStrategy,
    TraversalResult,
    TraversalStrategyRegistry,
    register_strategy,
)

from src.graph_retriever.strategies.one_hop import KHopStrategy
from src.graph_retriever.strategies.ppr import PPRStrategy
from src.graph_retriever.strategies.pcst import PCSTStrategy
from src.graph_retriever.strategies.tog import ToGStrategy
from src.graph_retriever.strategies.tog_refine import ToGRefineStrategy
from src.graph_retriever.strategies.anchor_hybrid_khop import AnchorHybridKHopStrategy

__all__ = [
    "BaseTraversalStrategy",
    "TraversalResult",
    "TraversalStrategyRegistry",
    "register_strategy",
    "KHopStrategy",
    "PPRStrategy",
    "PCSTStrategy",
    "ToGStrategy",
    "ToGRefineStrategy",
    "AnchorHybridKHopStrategy",
]
