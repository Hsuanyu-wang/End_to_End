"""
核心組件基類測試

測試抽象基類的基本功能和介面定義。
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from core.entity_extraction import BaseEntityExtractor
from core.relation_extraction import BaseRelationExtractor
from core.schema import BaseSchemaManager
from core.graph_construction import BaseGraphConstructor
from core.retrieval import BaseRetriever, RetrievalResult
from formats import Entity, EntityList, Relation, RelationList, Schema, GraphData, Graph


class ConcreteEntityExtractor(BaseEntityExtractor):
    """測試用的具體實體抽取器"""
    
    def extract(self, text, schema=None, **kwargs):
        # 簡單實作：返回固定實體
        return EntityList([
            Entity(entity_id="e1", name="測試實體", type="Test")
        ])


class ConcreteRelationExtractor(BaseRelationExtractor):
    """測試用的具體關係抽取器"""
    
    def extract(self, text, entities, schema=None, **kwargs):
        # 簡單實作：返回空關係列表
        return RelationList()


class ConcreteSchemaManager(BaseSchemaManager):
    """測試用的具體 Schema 管理器"""
    
    def learn(self, entities, relations, **kwargs):
        # 簡單實作：從實體類型創建 Schema
        from formats import EntityType
        entity_types = list(set(e.type for e in entities))
        return Schema(entity_types=[EntityType(name=et) for et in entity_types])
    
    def align(self, entities, relations, schema, **kwargs):
        # 簡單實作：返回原始實體和關係
        return entities, relations


class ConcreteGraphConstructor(BaseGraphConstructor):
    """測試用的具體圖譜建構器"""
    
    def build(self, entities, relations, schema=None, documents=None, **kwargs):
        graph = Graph(entities=entities, relations=relations)
        return GraphData(graph=graph)


class ConcreteRetriever(BaseRetriever):
    """測試用的具體檢索器"""
    
    def retrieve(self, query, graph_data, top_k=5, **kwargs):
        # 簡單實作：返回固定上下文
        contexts = ["測試上下文1", "測試上下文2"][:top_k]
        return RetrievalResult(contexts=contexts)


class TestEntityExtractor(unittest.TestCase):
    """測試實體抽取器基類"""
    
    def setUp(self):
        self.extractor = ConcreteEntityExtractor()
    
    def test_extract(self):
        """測試抽取實體"""
        entities = self.extractor.extract("測試文本")
        self.assertIsInstance(entities, EntityList)
        self.assertGreater(len(entities), 0)
    
    def test_extract_batch(self):
        """測試批次抽取"""
        texts = ["文本1", "文本2"]
        results = self.extractor.extract_batch(texts)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], EntityList)
    
    def test_validate_input(self):
        """測試輸入驗證"""
        self.assertTrue(self.extractor.validate_input("有效文本"))
        self.assertFalse(self.extractor.validate_input(""))
        self.assertFalse(self.extractor.validate_input(None))


class TestRelationExtractor(unittest.TestCase):
    """測試關係抽取器基類"""
    
    def setUp(self):
        self.extractor = ConcreteRelationExtractor()
        self.entities = EntityList([
            Entity(entity_id="e1", name="A", type="Person"),
            Entity(entity_id="e2", name="B", type="Person")
        ])
    
    def test_extract(self):
        """測試抽取關係"""
        relations = self.extractor.extract("測試文本", self.entities)
        self.assertIsInstance(relations, RelationList)
    
    def test_validate_input(self):
        """測試輸入驗證"""
        self.assertTrue(self.extractor.validate_input("有效文本", self.entities))
        self.assertFalse(self.extractor.validate_input("", self.entities))
        self.assertFalse(self.extractor.validate_input("文本", EntityList()))


class TestSchemaManager(unittest.TestCase):
    """測試 Schema 管理器基類"""
    
    def setUp(self):
        self.manager = ConcreteSchemaManager()
        self.entities = EntityList([
            Entity(entity_id="e1", name="A", type="Person"),
            Entity(entity_id="e2", name="B", type="Organization")
        ])
        self.relations = RelationList([
            Relation(relation_id="r1", head="e1", relation="works_at", tail="e2")
        ])
    
    def test_learn(self):
        """測試學習 Schema"""
        schema = self.manager.learn(self.entities, self.relations)
        self.assertIsInstance(schema, Schema)
        self.assertTrue(schema.has_entity_type("Person"))
        self.assertTrue(schema.has_entity_type("Organization"))
    
    def test_align(self):
        """測試對齊"""
        schema = self.manager.learn(self.entities, self.relations)
        aligned_entities, aligned_relations = self.manager.align(
            self.entities, self.relations, schema
        )
        self.assertIsInstance(aligned_entities, EntityList)
        self.assertIsInstance(aligned_relations, RelationList)


class TestGraphConstructor(unittest.TestCase):
    """測試圖譜建構器基類"""
    
    def setUp(self):
        self.constructor = ConcreteGraphConstructor()
        self.entities = EntityList([
            Entity(entity_id="e1", name="A", type="Person"),
            Entity(entity_id="e2", name="B", type="Person")
        ])
        self.relations = RelationList([
            Relation(relation_id="r1", head="e1", relation="knows", tail="e2")
        ])
    
    def test_build(self):
        """測試建構圖譜"""
        graph_data = self.constructor.build(self.entities, self.relations)
        self.assertIsInstance(graph_data, GraphData)
        self.assertIsInstance(graph_data.graph, Graph)
    
    def test_validate_graph(self):
        """測試圖譜驗證"""
        errors = self.constructor.validate_graph(self.entities, self.relations)
        self.assertEqual(len(errors), 0)
        
        # 測試無效關係
        invalid_relations = RelationList([
            Relation(relation_id="r1", head="e1", relation="knows", tail="e999")  # 不存在的實體
        ])
        errors = self.constructor.validate_graph(self.entities, invalid_relations)
        self.assertGreater(len(errors), 0)


class TestRetriever(unittest.TestCase):
    """測試檢索器基類"""
    
    def setUp(self):
        self.retriever = ConcreteRetriever()
        graph = Graph(
            entities=EntityList([Entity(entity_id="e1", name="A", type="Person")]),
            relations=RelationList()
        )
        self.graph_data = GraphData(graph=graph)
    
    def test_retrieve(self):
        """測試檢索"""
        result = self.retriever.retrieve("測試查詢", self.graph_data, top_k=2)
        self.assertIsInstance(result, RetrievalResult)
        self.assertGreater(len(result.contexts), 0)
        self.assertLessEqual(len(result.contexts), 2)
    
    def test_retrieve_batch(self):
        """測試批次檢索"""
        queries = ["查詢1", "查詢2"]
        results = self.retriever.retrieve_batch(queries, self.graph_data)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], RetrievalResult)


if __name__ == '__main__':
    unittest.main()
