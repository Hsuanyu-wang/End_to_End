"""
關係標準格式定義

定義了統一的關係表示格式，支援從不同來源的關係抽取。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class Relation:
    """
    標準關係格式
    
    所有組件產出的關係都應轉換為此格式，確保互操作性。
    
    Attributes:
        relation_id: 關係唯一識別碼
        head: 頭實體 ID
        relation: 關係類型（如 "works_at", "located_in"）
        tail: 尾實體 ID
        properties: 關係屬性字典
        confidence: 信心分數 (0.0-1.0)
        source: 來源標識
        metadata: 元資料
    
    Examples:
        >>> relation = Relation(
        ...     relation_id="r1",
        ...     head="e1",
        ...     relation="works_at",
        ...     tail="e2",
        ...     properties={"since": "2020"},
        ...     confidence=0.9,
        ...     source="lightrag"
        ... )
    """
    relation_id: str
    head: str  # entity_id
    relation: str
    tail: str  # entity_id
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """驗證關係資料的有效性"""
        if not self.relation_id:
            raise ValueError("relation_id cannot be empty")
        if not self.head:
            raise ValueError("head entity_id cannot be empty")
        if not self.relation:
            raise ValueError("relation type cannot be empty")
        if not self.tail:
            raise ValueError("tail entity_id cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "relation_id": self.relation_id,
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "properties": self.properties,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        """從字典創建關係"""
        return cls(
            relation_id=data["relation_id"],
            head=data["head"],
            relation=data["relation"],
            tail=data["tail"],
            properties=data.get("properties", {}),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", ""),
            metadata=data.get("metadata", {}),
        )
    
    def to_triple(self) -> tuple:
        """轉換為三元組格式 (head, relation, tail)"""
        return (self.head, self.relation, self.tail)
    
    def reverse(self) -> "Relation":
        """
        創建反向關係
        
        例如：(A, works_at, B) -> (B, has_employee, A)
        """
        reverse_relation = f"inverse_{self.relation}"
        return Relation(
            relation_id=f"{self.relation_id}_inverse",
            head=self.tail,
            relation=reverse_relation,
            tail=self.head,
            properties=self.properties.copy(),
            confidence=self.confidence,
            source=self.source,
            metadata={**self.metadata, "is_inverse": True},
        )


@dataclass
class RelationList:
    """
    關係列表容器
    
    提供關係列表的便利操作方法。
    """
    relations: List[Relation] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.relations)
    
    def __iter__(self):
        return iter(self.relations)
    
    def __getitem__(self, index):
        return self.relations[index]
    
    def add(self, relation: Relation):
        """新增關係"""
        self.relations.append(relation)
    
    def get_by_id(self, relation_id: str) -> Optional[Relation]:
        """根據 ID 獲取關係"""
        for relation in self.relations:
            if relation.relation_id == relation_id:
                return relation
        return None
    
    def get_by_entity(self, entity_id: str) -> List[Relation]:
        """獲取與指定實體相關的所有關係"""
        return [r for r in self.relations if r.head == entity_id or r.tail == entity_id]
    
    def get_by_head(self, entity_id: str) -> List[Relation]:
        """獲取以指定實體為頭的所有關係"""
        return [r for r in self.relations if r.head == entity_id]
    
    def get_by_tail(self, entity_id: str) -> List[Relation]:
        """獲取以指定實體為尾的所有關係"""
        return [r for r in self.relations if r.tail == entity_id]
    
    def get_by_type(self, relation_type: str) -> List[Relation]:
        """根據關係類型獲取關係列表"""
        return [r for r in self.relations if r.relation == relation_type]
    
    def get_by_source(self, source: str) -> List[Relation]:
        """根據來源獲取關係列表"""
        return [r for r in self.relations if source in r.source]
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """轉換為字典列表"""
        return [r.to_dict() for r in self.relations]
    
    @classmethod
    def from_dict_list(cls, data_list: List[Dict[str, Any]]) -> "RelationList":
        """從字典列表創建"""
        relations = [Relation.from_dict(d) for d in data_list]
        return cls(relations=relations)
    
    def to_triples(self) -> List[tuple]:
        """轉換為三元組列表"""
        return [r.to_triple() for r in self.relations]
    
    def deduplicate(self) -> "RelationList":
        """
        去除重複關係
        
        基於 (head, relation, tail) 三元組去重，保留信心度最高的。
        """
        seen = {}
        result = []
        
        for relation in self.relations:
            triple = relation.to_triple()
            if triple in seen:
                # 保留信心度較高的
                existing = seen[triple]
                if relation.confidence > existing.confidence:
                    # 替換為信心度更高的
                    for i, r in enumerate(result):
                        if r.to_triple() == triple:
                            result[i] = relation
                            break
                    seen[triple] = relation
            else:
                seen[triple] = relation
                result.append(relation)
        
        return RelationList(relations=result)
    
    def filter_by_confidence(self, min_confidence: float = 0.5) -> "RelationList":
        """根據信心度過濾關係"""
        filtered = [r for r in self.relations if r.confidence >= min_confidence]
        return RelationList(relations=filtered)
