"""
統一格式定義模組

此模組定義了所有組件之間交換資料的標準格式，確保不同實作的組件可以互相操作。

主要格式：
- Entity: 實體標準格式
- Relation: 關係標準格式  
- Graph: 圖譜標準格式
- Schema: Schema 標準格式
"""

from .entity import Entity, EntityList
from .relation import Relation, RelationList
from .graph import Graph, GraphData
from .schema import Schema, EntityType, RelationType, SchemaType

__all__ = [
    "Entity",
    "EntityList",
    "Relation",
    "RelationList",
    "Graph",
    "GraphData",
    "Schema",
    "EntityType",
    "RelationType",
    "SchemaType",
]
