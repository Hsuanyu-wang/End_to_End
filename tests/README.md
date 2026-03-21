# Phase 1 測試運行指南

## 執行所有測試

```bash
cd /home/End_to_End_RAG
python -m pytest tests/
```

## 執行特定測試

```bash
# 測試格式定義
python -m pytest tests/unit/formats/test_formats.py -v

# 測試核心組件基類
python -m pytest tests/unit/core/test_base_classes.py -v

# 運行單一測試
python -m pytest tests/unit/formats/test_formats.py::TestEntity::test_create_entity -v
```

## 使用 unittest 運行

```bash
# 測試格式
python tests/unit/formats/test_formats.py

# 測試基類
python tests/unit/core/test_base_classes.py
```

## 測試覆蓋範圍

Phase 1 的測試涵蓋：

1. **格式定義測試** (`test_formats.py`)
   - Entity 創建、驗證、合併
   - Relation 創建、轉換
   - Graph 建構、鄰居查詢
   - Schema 驗證

2. **基類測試** (`test_base_classes.py`)
   - BaseEntityExtractor 介面
   - BaseRelationExtractor 介面
   - BaseSchemaManager 介面
   - BaseGraphConstructor 介面
   - BaseRetriever 介面

## 下一步

Phase 2 將建立：
- LightRAG 組件的具體實作測試
- 整合測試
- 效能測試
