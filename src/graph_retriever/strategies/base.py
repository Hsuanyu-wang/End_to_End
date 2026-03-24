"""
Graph Traversal Strategy 基類

定義所有 graph traversal 策略的統一介面。
Seed entities 由 EntityLinker 提供，strategy 只負責在 NetworkX graph 上做子圖擴展。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import networkx as nx


@dataclass
class TraversalResult:
    """Graph traversal 策略的回傳結果。"""

    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTraversalStrategy(ABC):
    """
    Graph traversal 策略基類。

    所有策略接受同一組輸入（NetworkX graph + seed entities），
    回傳 TraversalResult（nodes + edges + metadata）。
    """

    @abstractmethod
    def get_name(self) -> str:
        ...

    @abstractmethod
    def traverse(
        self,
        nx_graph: nx.DiGraph,
        seed_entities: List[Dict[str, Any]],
        query: str,
        top_k: int = 20,
        **kwargs,
    ) -> TraversalResult:
        """
        在 nx_graph 上從 seed_entities 出發執行 graph traversal。

        Args:
            nx_graph: LightRAG 的 NetworkX DiGraph（含 weight 等 edge 屬性）
            seed_entities: Entity Linker 回傳的 seed nodes，
                           每個 dict 至少含 entity_name, vdb_score
            query: 原始使用者查詢（部分策略如 ToG 需要）
            top_k: 最多回傳的節點數
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.get_name()}')"


class TraversalStrategyRegistry:
    """Graph traversal strategy 的註冊與工廠。"""

    _strategies: Dict[str, Type[BaseTraversalStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_class: Type[BaseTraversalStrategy]):
        cls._strategies[name] = strategy_class

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseTraversalStrategy:
        if name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(
                f"未知的 traversal strategy: {name}。可用: {available}"
            )
        return cls._strategies[name](**kwargs)

    @classmethod
    def list_available(cls) -> List[str]:
        return list(cls._strategies.keys())


def register_strategy(name: str):
    """Decorator：自動註冊 strategy。"""
    def decorator(cls: Type[BaseTraversalStrategy]):
        TraversalStrategyRegistry.register(name, cls)
        return cls
    return decorator
