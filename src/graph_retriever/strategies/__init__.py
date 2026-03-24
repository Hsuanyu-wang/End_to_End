"""
Graph Traversal Strategies

可插拔的 graph traversal 策略：one_hop, ppr, pcst, tog。
import 此 package 時自動註冊所有內建策略。
"""

from src.graph_retriever.strategies.base import (
    BaseTraversalStrategy,
    TraversalResult,
    TraversalStrategyRegistry,
    register_strategy,
)

from src.graph_retriever.strategies.one_hop import OneHopStrategy
from src.graph_retriever.strategies.ppr import PPRStrategy
from src.graph_retriever.strategies.pcst import PCSTStrategy
from src.graph_retriever.strategies.tog import ToGStrategy

__all__ = [
    "BaseTraversalStrategy",
    "TraversalResult",
    "TraversalStrategyRegistry",
    "register_strategy",
    "OneHopStrategy",
    "PPRStrategy",
    "PCSTStrategy",
    "ToGStrategy",
]
