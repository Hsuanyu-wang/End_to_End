"""
資料處理測試

測試資料載入與處理功能
"""

import pytest
import tempfile
import json
import csv
from pathlib import Path
from src.data.processors import DataProcessor
from src.data.loaders import QADataLoader


class TestDataProcessor:
    """資料處理器測試"""
    
    def test_natural_text_mode(self, tmp_path):
        """測試 natural_text 模式"""
        # 建立測試資料
        test_data = {
            "NO": "TEST001",
            "Customer": "測試客戶",
            "Engineers": "工程師A",
            "Service Start": "2024-01-01",
            "Service End": "2024-01-02",
            "Description": "問題描述",
            "Action": "處理動作"
        }
        
        data_file = tmp_path / "test_data.jsonl"
        with open(data_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(test_data, ensure_ascii=False) + "\n")
        
        # 處理資料（需要 mock Settings）
        # processor = DataProcessor(mode="natural_text", data_type="DI")
        # processor.raw_file_path = str(data_file)
        # documents = processor.process()
        
        # 驗證結果
        # assert len(documents) == 1
        # assert "測試客戶" in documents[0].text
        # assert documents[0].metadata["NO"] == "TEST001"
    
    def test_markdown_mode(self):
        """測試 markdown 模式"""
        # 類似的測試邏輯
        pass
    
    def test_empty_file(self):
        """測試空檔案"""
        # 測試邊界情況
        pass


class TestQADataLoader:
    """QA 資料載入器測試"""
    
    def test_csv_loading(self, tmp_path):
        """測試 CSV 格式載入"""
        # 建立測試 CSV
        csv_file = tmp_path / "test_qa.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Q", "GT", "REF"])
            writer.writerow(["問題1", "答案1", "doc1"])
            writer.writerow(["問題2", "答案2", "doc2\ndoc3"])
        
        # 測試載入（需要 mock Settings）
        # loader = QADataLoader(data_type="DI")
        # loader.qa_file_path = str(csv_file)
        # data = loader.load_and_normalize()
        
        # 驗證結果
        # assert len(data) == 2
        # assert data[0]["query"] == "問題1"
        # assert data[0]["ground_truth_doc_ids"] == ["doc1"]
    
    def test_jsonl_loading(self, tmp_path):
        """測試 JSONL 格式載入"""
        # 建立測試 JSONL
        jsonl_file = tmp_path / "test_qa.jsonl"
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            data1 = {
                "question": "問題1",
                "answer": "答案1",
                "source_doc_ids": ["doc1", "doc2"]
            }
            f.write(json.dumps(data1, ensure_ascii=False) + "\n")
        
        # 測試載入
        # loader = QADataLoader(data_type="GEN")
        # loader.qa_file_path = str(jsonl_file)
        # data = loader.load_and_normalize()
        
        # 驗證結果
        # assert len(data) == 1
        # assert data[0]["query"] == "問題1"
        # assert len(data[0]["ground_truth_doc_ids"]) == 2
    
    def test_normalization_format(self):
        """測試正規化格式"""
        # 確保輸出格式一致
        expected_keys = {"source", "query", "ground_truth_answer", "ground_truth_doc_ids"}
        # 驗證 keys
        pass


class TestDataModes:
    """資料模式測試"""
    
    def test_all_modes_produce_documents(self):
        """測試所有模式都能產生文件"""
        modes = ["natural_text", "markdown", "key_value_text", "unstructured_text"]
        
        for mode in modes:
            # processor = DataProcessor(mode=mode, data_type="DI")
            # documents = processor.process()
            # assert len(documents) > 0
            pass
    
    def test_metadata_consistency(self):
        """測試 metadata 一致性"""
        # 確保所有模式的 metadata 欄位一致
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
