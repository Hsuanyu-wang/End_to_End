# 模組化 Graph Pipeline 架構說明

本文檔說明 End-to-End RAG 系統的模組化 Graph Pipeline 設計。

## 概述

系統支援兩種使用方式:

1. **端到端方法**:完整的建圖+檢索 Pipeline,開箱即用
2. **模組化方法**:自由組合 Builder + Retriever,靈活實驗

## 架構設計

### 核心組件

```
Graph Pipeline = Graph Builder + Graph Retriever + LLM Generator

Builder 階段:處理文檔 → 建立知識圖譜
Retriever 階段:查詢問題 → 檢索相關上下文
Generator 階段:上下文 + 問題 → 生成答案
```

### 基類介面

**BaseGraphBuilder** ([src/graph_builder/base_builder.py](../src/graph_builder/base_builder.py))
- `build(documents)`: 建立圖譜,返回標準化的 graph_data
- `get_name()`: 取得 Builder 名稱

**BaseGraphRetriever** ([src/graph_retriever/base_retriever.py](../src/graph_retriever/base_retriever.py))
- `retrieve(query, graph_data, top_k)`: 從圖譜檢索,返回 contexts
- `get_name()`: 取得 Retriever 名稱

**ModularGraphWrapper** ([src/rag/wrappers/modular_graph_wrapper.py](../src/rag/wrappers/modular_graph_wrapper.py))
- 組合 Builder + Retriever
- 統一的查詢介面:`query_and_log()`

## 支援的方法

### 端到端方法

| 方法 | 說明 | 狀態 | 命令範例 |
|------|------|------|---------|
| **LightRAG** | 完整的 LightRAG 建圖+檢索 | ✅ 完整實作 | `--graph_rag_method lightrag --lightrag_mode hybrid` |
| **Vector RAG** | 向量檢索基準線 | ✅ 完整實作 | `--vector_method hybrid` |
| **AutoSchemaKG** | Schema-free 建圖+概念層級檢索 | 🔨 可整合 | `--graph_rag_method autoschema` |
| **DynamicSchema** | LlamaIndex 動態 Schema | ✅ 完整實作 | `--graph_rag_method dynamic_schema` |
| **Graphiti** | 時序感知圖譜 | 📋 架構預留 | `--graph_rag_method graphiti` |
| **Neo4j** | 圖資料庫+圖演算法 | 📋 架構預留 | `--graph_rag_method neo4j` |
| **CQ-Driven** | 能力問題驅動本體 | 📋 架構預留 | `--graph_rag_method cq_driven` |

### 模組化組合

#### 可用的 Builders

| Builder | 說明 | 檔案 |
|---------|------|------|
| `autoschema` | AutoSchemaKG 三元組抽取+概念化 | [autoschema_builder.py](../src/graph_builder/autoschema_builder.py) |
| `lightrag` | LightRAG 建圖 | [lightrag_builder.py](../src/graph_builder/lightrag_builder.py) |
| `dynamic` | LlamaIndex DynamicLLMPathExtractor | [dynamic_schema_builder.py](../src/graph_builder/dynamic_schema_builder.py) |
| `property` | LlamaIndex PropertyGraph | [baseline_builder.py](../src/graph_builder/baseline_builder.py) |

#### 可用的 Retrievers

| Retriever | 說明 | 檔案 |
|-----------|------|------|
| `lightrag` | LightRAG 檢索(local/global/hybrid/mix) | [lightrag_retriever.py](../src/graph_retriever/lightrag_retriever.py) |
| `csr` | CSR Graph 檢索 | [csr_graph_query_engine.py](../src/graph_retriever/csr_graph_query_engine.py) |
| `neo4j` | Neo4j Cypher 查詢(待實作) | 待建立 |

#### 預設組合

| 組合名稱 | Builder | Retriever | 說明 |
|---------|---------|-----------|------|
| `autoschema_lightrag` | AutoSchemaKG | LightRAG | 概念層級建圖+混合檢索 |
| `lightrag_csr` | LightRAG | CSR | LightRAG 建圖+圖遍歷檢索 |
| `dynamic_csr` | DynamicSchema | CSR | 動態 Schema+圖遍歷檢索 |
| `dynamic_lightrag` | DynamicSchema | LightRAG | 動態 Schema+混合檢索 |

## 使用範例

### 端到端方法

```bash
# LightRAG 完整
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode hybrid \
    --data_type DI

# AutoSchemaKG 完整
python scripts/run_evaluation.py \
    --graph_rag_method autoschema \
    --data_type DI

# DynamicSchema 完整
python scripts/run_evaluation.py \
    --graph_rag_method dynamic_schema \
    --data_type DI

# Vector RAG 基準線
python scripts/run_evaluation.py \
    --vector_method hybrid \
    --data_type DI
```

### 模組化組合

```bash
# 使用預設組合:AutoSchemaKG Builder + LightRAG Retriever
python scripts/run_evaluation.py \
    --graph_preset autoschema_lightrag \
    --lightrag_mode hybrid \
    --data_type DI

# 自訂組合:LightRAG Builder + CSR Retriever
python scripts/run_evaluation.py \
    --graph_builder lightrag \
    --graph_retriever csr \
    --data_type DI

# 自訂組合:DynamicSchema Builder + LightRAG Retriever
python scripts/run_evaluation.py \
    --graph_builder dynamic \
    --graph_retriever lightrag \
    --lightrag_mode hybrid \
    --data_type DI
```

### 快速測試

```bash
# 快速測試模式(前 2 筆文檔,前 2 題 QA)
python scripts/run_evaluation.py \
    --graph_preset autoschema_lightrag \
    --qa_dataset_fast_test \
    --data_type DI
```

## 擴展指南

### 新增 Builder

1. 繼承 `BaseGraphBuilder`
2. 實作 `build(documents)` 方法
3. 返回標準化的 graph_data 字典
4. 在 `PipelineFactory` 註冊

```python
from src.graph_builder.base_builder import BaseGraphBuilder

class MyCustomBuilder(BaseGraphBuilder):
    def get_name(self) -> str:
        return "MyCustom"
    
    def build(self, documents):
        # 你的建圖邏輯
        return {
            "nodes": [...],
            "edges": [...],
            "metadata": {...},
            "schema_info": {...},
            "storage_path": "...",
            "graph_format": "custom"
        }
```

### 新增 Retriever

1. 繼承 `BaseGraphRetriever`
2. 實作 `retrieve(query, graph_data, top_k)` 方法
3. 返回包含 contexts 的字典
4. 在 `PipelineFactory` 註冊

```python
from src.graph_retriever.base_retriever import BaseGraphRetriever

class MyCustomRetriever(BaseGraphRetriever):
    def get_name(self) -> str:
        return "MyCustom"
    
    def retrieve(self, query, graph_data, top_k):
        # 你的檢索邏輯
        return {
            "contexts": [...],
            "nodes": [...],
            "metadata": {...}
        }
```

### 新增預設組合

在 `src/rag/pipeline_factory.py` 的 `PRESET_PIPELINES` 中新增:

```python
PRESET_PIPELINES = {
    "my_combination": {
        "builder": "my_builder",
        "retriever": "my_retriever",
        "description": "My custom combination"
    }
}
```

## graph_data 標準格式

Builder 返回的 graph_data 應符合以下格式:

```python
{
    "nodes": [
        {"id": "node1", "label": "Person", "properties": {...}},
        ...
    ],
    "edges": [
        {"source": "node1", "target": "node2", "type": "knows"},
        ...
    ],
    "metadata": {
        "num_documents": 100,
        "build_time": 123.45,
        ...
    },
    "schema_info": {
        "entities": ["Person", "Organization", ...],
        "relations": ["works_at", "knows", ...],
        "method": "autoschema"
    },
    "storage_path": "/path/to/graph/storage",
    "graph_format": "networkx"  # or "lightrag", "neo4j", "custom"
}
```

## 常見問題

### Q: 如何選擇 Builder 和 Retriever?

A: 根據你的需求:
- **需要概念層級**:使用 `autoschema` Builder
- **需要時序資訊**:使用 `lightrag` Builder(未來支援 temporal)
- **不想預定義 Schema**:使用 `dynamic` Builder
- **需要圖遍歷**:使用 `csr` Retriever
- **需要混合檢索**:使用 `lightrag` Retriever

### Q: 端到端方法和模組化方法有什麼區別?

A: 
- **端到端**:Builder 和 Retriever 緊密整合,通常效能更好,但靈活性較低
- **模組化**:可自由組合,方便進行消融實驗,評估各階段貢獻

### Q: 如何新增架構預留方法(Graphiti/Neo4j/CQ-Driven)?

A: 參考已實作的 Wrapper:
1. 建立 Wrapper 檔案(如 `graphiti_wrapper.py`)
2. 繼承 `BaseRAGWrapper`
3. 實作 `_execute_query()` 方法
4. 在 `setup_graph_pipelines()` 中添加判斷

## 相關文檔

- [API 參考](API.md)
- [測試指南](TESTING_GUIDE.md)
- [使用範例](EXAMPLES.md)
- [評估指標](METRICS.md)
