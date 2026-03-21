"""
圖譜標準格式定義

定義了統一的圖譜表示格式，支援不同圖譜建構方法的輸出。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from .entity import Entity, EntityList
from .relation import Relation, RelationList


@dataclass
class Graph:
    """
    標準圖譜格式
    
    統一的圖譜表示，包含實體、關係和相關元資料。
    
    Attributes:
        entities: 實體列表
        relations: 關係列表
        metadata: 圖譜元資料（如建構時間、來源等）
        statistics: 圖譜統計資訊
    
    Examples:
        >>> entities = EntityList([entity1, entity2])
        >>> relations = RelationList([relation1])
        >>> graph = Graph(entities=entities, relations=relations)
    """
    entities: EntityList = field(default_factory=EntityList)
    relations: RelationList = field(default_factory=RelationList)
    metadata: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """計算圖譜統計資訊"""
        self._update_statistics()
    
    def _update_statistics(self):
        """更新圖譜統計資訊"""
        self.statistics = {
            "num_entities": len(self.entities),
            "num_relations": len(self.relations),
            "entity_types": self._count_entity_types(),
            "relation_types": self._count_relation_types(),
            "avg_entity_confidence": self._avg_entity_confidence(),
            "avg_relation_confidence": self._avg_relation_confidence(),
        }
    
    def _count_entity_types(self) -> Dict[str, int]:
        """統計實體類型分佈"""
        type_counts = {}
        for entity in self.entities:
            type_counts[entity.type] = type_counts.get(entity.type, 0) + 1
        return type_counts
    
    def _count_relation_types(self) -> Dict[str, int]:
        """統計關係類型分佈"""
        type_counts = {}
        for relation in self.relations:
            type_counts[relation.relation] = type_counts.get(relation.relation, 0) + 1
        return type_counts
    
    def _avg_entity_confidence(self) -> float:
        """計算平均實體信心度"""
        if len(self.entities) == 0:
            return 0.0
        return sum(e.confidence for e in self.entities) / len(self.entities)
    
    def _avg_relation_confidence(self) -> float:
        """計算平均關係信心度"""
        if len(self.relations) == 0:
            return 0.0
        return sum(r.confidence for r in self.relations) / len(self.relations)
    
    def add_entity(self, entity: Entity):
        """新增實體"""
        self.entities.add(entity)
        self._update_statistics()
    
    def add_relation(self, relation: Relation):
        """新增關係"""
        self.relations.add(relation)
        self._update_statistics()
    
    def get_neighbors(self, entity_id: str, direction: str = "both") -> Set[str]:
        """
        獲取指定實體的鄰居節點
        
        Args:
            entity_id: 實體 ID
            direction: 方向 ("in", "out", "both")
        
        Returns:
            鄰居節點 ID 集合
        """
        neighbors = set()
        
        if direction in ["out", "both"]:
            for relation in self.relations.get_by_head(entity_id):
                neighbors.add(relation.tail)
        
        if direction in ["in", "both"]:
            for relation in self.relations.get_by_tail(entity_id):
                neighbors.add(relation.head)
        
        return neighbors
    
    def get_subgraph(self, entity_ids: List[str], k_hop: int = 1) -> "Graph":
        """
        提取子圖
        
        Args:
            entity_ids: 種子實體 ID 列表
            k_hop: 跳數（1 表示只包含直接鄰居）
        
        Returns:
            子圖
        """
        # 收集 k-hop 內的所有實體
        current_entities = set(entity_ids)
        all_entities = set(entity_ids)
        
        for _ in range(k_hop):
            next_entities = set()
            for entity_id in current_entities:
                neighbors = self.get_neighbors(entity_id)
                next_entities.update(neighbors)
            all_entities.update(next_entities)
            current_entities = next_entities
        
        # 提取相關實體和關係
        subgraph_entities = [e for e in self.entities if e.entity_id in all_entities]
        subgraph_relations = [
            r for r in self.relations
            if r.head in all_entities and r.tail in all_entities
        ]
        
        return Graph(
            entities=EntityList(entities=subgraph_entities),
            relations=RelationList(relations=subgraph_relations),
            metadata={**self.metadata, "is_subgraph": True, "k_hop": k_hop},
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "entities": self.entities.to_dict_list(),
            "relations": self.relations.to_dict_list(),
            "metadata": self.metadata,
            "statistics": self.statistics,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Graph":
        """從字典創建圖譜"""
        entities = EntityList.from_dict_list(data.get("entities", []))
        relations = RelationList.from_dict_list(data.get("relations", []))
        metadata = data.get("metadata", {})
        
        return cls(entities=entities, relations=relations, metadata=metadata)
    
    def merge(self, other: "Graph") -> "Graph":
        """
        合併兩個圖譜
        
        去重並合併實體和關係。
        """
        merged_entities = EntityList(entities=self.entities.entities + other.entities.entities)
        merged_entities = merged_entities.deduplicate(merge=True)
        
        merged_relations = RelationList(relations=self.relations.relations + other.relations.relations)
        merged_relations = merged_relations.deduplicate()
        
        merged_metadata = {**self.metadata, **other.metadata, "is_merged": True}
        
        return Graph(
            entities=merged_entities,
            relations=merged_relations,
            metadata=merged_metadata,
        )


@dataclass
class GraphData:
    """
    圖譜資料容器
    
    用於在 Builder 和 Retriever 之間傳遞圖譜資訊的標準格式。
    這是計畫中提到的統一圖譜格式介面。
    
    Attributes:
        graph: Graph 物件
        storage_path: 圖譜儲存路徑（可選）
        format_type: 圖譜格式類型（如 "networkx", "neo4j", "lightrag"）
        raw_data: 原始格式的資料（用於需要保留原始格式的場景）
    """
    graph: Graph
    storage_path: Optional[str] = None
    format_type: str = "standard"
    raw_data: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "graph": self.graph.to_dict(),
            "storage_path": self.storage_path,
            "format_type": self.format_type,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphData":
        """從字典創建"""
        graph = Graph.from_dict(data["graph"])
        return cls(
            graph=graph,
            storage_path=data.get("storage_path"),
            format_type=data.get("format_type", "standard"),
        )
