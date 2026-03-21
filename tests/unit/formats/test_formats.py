"""
格式單元測試
"""

import unittest
import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from formats import Entity, EntityList, Relation, RelationList, Graph, Schema, EntityType, RelationType, SchemaType


class TestEntity(unittest.TestCase):
    """測試 Entity 類別"""
    
    def test_create_entity(self):
        """測試創建實體"""
        entity = Entity(
            entity_id="e1",
            name="張三",
            type="Person",
            confidence=0.9
        )
        self.assertEqual(entity.entity_id, "e1")
        self.assertEqual(entity.name, "張三")
        self.assertEqual(entity.type, "Person")
        self.assertEqual(entity.confidence, 0.9)
    
    def test_entity_validation(self):
        """測試實體驗證"""
        with self.assertRaises(ValueError):
            Entity(entity_id="", name="test", type="Person")
        
        with self.assertRaises(ValueError):
            Entity(entity_id="e1", name="", type="Person")
        
        with self.assertRaises(ValueError):
            Entity(entity_id="e1", name="test", type="", confidence=1.5)
    
    def test_entity_merge(self):
        """測試實體合併"""
        e1 = Entity(entity_id="e1", name="張三", type="Person", confidence=0.9)
        e2 = Entity(entity_id="e1", name="Zhang San", type="Person", confidence=0.7, aliases=["小張"])
        
        merged = e1.merge_with(e2)
        self.assertEqual(merged.name, "張三")  # 保留較高信心度的名稱
        self.assertEqual(merged.confidence, 0.9)
        self.assertIn("小張", merged.aliases)


class TestRelation(unittest.TestCase):
    """測試 Relation 類別"""
    
    def test_create_relation(self):
        """測試創建關係"""
        relation = Relation(
            relation_id="r1",
            head="e1",
            relation="works_at",
            tail="e2",
            confidence=0.8
        )
        self.assertEqual(relation.relation_id, "r1")
        self.assertEqual(relation.head, "e1")
        self.assertEqual(relation.relation, "works_at")
        self.assertEqual(relation.tail, "e2")
    
    def test_relation_to_triple(self):
        """測試轉換為三元組"""
        relation = Relation(
            relation_id="r1",
            head="e1",
            relation="works_at",
            tail="e2"
        )
        triple = relation.to_triple()
        self.assertEqual(triple, ("e1", "works_at", "e2"))


class TestGraph(unittest.TestCase):
    """測試 Graph 類別"""
    
    def test_create_graph(self):
        """測試創建圖譜"""
        entities = EntityList([
            Entity(entity_id="e1", name="張三", type="Person"),
            Entity(entity_id="e2", name="公司A", type="Organization")
        ])
        relations = RelationList([
            Relation(relation_id="r1", head="e1", relation="works_at", tail="e2")
        ])
        
        graph = Graph(entities=entities, relations=relations)
        self.assertEqual(len(graph.entities), 2)
        self.assertEqual(len(graph.relations), 1)
        self.assertEqual(graph.statistics["num_entities"], 2)
        self.assertEqual(graph.statistics["num_relations"], 1)
    
    def test_get_neighbors(self):
        """測試獲取鄰居"""
        entities = EntityList([
            Entity(entity_id="e1", name="A", type="Person"),
            Entity(entity_id="e2", name="B", type="Person"),
            Entity(entity_id="e3", name="C", type="Organization")
        ])
        relations = RelationList([
            Relation(relation_id="r1", head="e1", relation="knows", tail="e2"),
            Relation(relation_id="r2", head="e1", relation="works_at", tail="e3")
        ])
        
        graph = Graph(entities=entities, relations=relations)
        neighbors = graph.get_neighbors("e1", direction="out")
        self.assertEqual(neighbors, {"e2", "e3"})


class TestSchema(unittest.TestCase):
    """測試 Schema 類別"""
    
    def test_create_schema(self):
        """測試創建 Schema"""
        entity_types = [
            EntityType(name="Person"),
            EntityType(name="Organization")
        ]
        relation_types = [
            RelationType(name="works_at", head_types=["Person"], tail_types=["Organization"])
        ]
        
        schema = Schema(entity_types=entity_types, relation_types=relation_types)
        self.assertEqual(len(schema.entity_types), 2)
        self.assertEqual(len(schema.relation_types), 1)
        self.assertTrue(schema.has_entity_type("Person"))
        self.assertTrue(schema.has_relation_type("works_at"))
    
    def test_schema_validation(self):
        """測試 Schema 驗證"""
        schema = Schema(
            entity_types=[EntityType(name="Person")],
            relation_types=[
                RelationType(name="knows", head_types=["Person"], tail_types=["Person"])
            ]
        )
        
        self.assertTrue(schema.validate_entity("Person"))
        self.assertFalse(schema.validate_entity("Organization"))
        self.assertTrue(schema.validate_relation("knows", "Person", "Person"))
        self.assertFalse(schema.validate_relation("works_at", "Person", "Organization"))


if __name__ == '__main__':
    unittest.main()
