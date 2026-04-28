"""
OntoLearner（entity-only）單元測試
"""

import unittest
import sys
from pathlib import Path


# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


from rag.schema.onto_learner import learn_entity_types


class _FakeCompletion:
    def __init__(self, text: str):
        self.text = text


class _FakeLLM:
    def __init__(self, text: str):
        self._text = text

    def complete(self, prompt: str):
        return _FakeCompletion(self._text)


class _Doc:
    def __init__(self, text: str):
        self.text = text


class TestOntoLearnerEntities(unittest.TestCase):
    def test_learn_entity_types_json_list_and_normalization(self):
        llm = _FakeLLM('["Server","Customer","Server","Server_001","A","IP_10_0_0_1","Issue"]')
        corpus = [_Doc("任意文本 1"), _Doc("任意文本 2")]

        entities = learn_entity_types(corpus, llm, max_entity_types=30, sample_docs=2, max_chars_per_doc=50)

        self.assertIsInstance(entities, list)
        self.assertTrue(all(isinstance(x, str) for x in entities))
        self.assertIn("Server", entities)
        self.assertIn("Customer", entities)
        self.assertIn("Issue", entities)
        self.assertNotIn("Server_001", entities)
        self.assertNotIn("IP_10_0_0_1", entities)
        self.assertNotIn("A", entities)  # 太短

    def test_learn_entity_types_handles_no_text(self):
        llm = _FakeLLM('["Server"]')
        entities = learn_entity_types([_Doc("   ")], llm)
        self.assertEqual(entities, [])


if __name__ == "__main__":
    unittest.main()

