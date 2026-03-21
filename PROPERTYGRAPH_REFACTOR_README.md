# PropertyGraph 模組化重構 - 完整實作

## 概述

本重構專案實現了一個**插件化、模組化的 Graph RAG 架構**，核心特點：

1. **NetworkX Hub 模式**：所有圖譜格式透過 NetworkX 作為中轉站互轉
2. **Plugin Registry System**：Builder 和 Retriever 可動態註冊和擴展
3. **多方法並行**：PropertyGraph 支援多 extractor/retriever 組合
4. **格式無關**：任意 Builder × Retriever 組合（自動格式轉換）

## 架構圖

```
┌─────────────────┐
│  User Request   │
└────────┬────────┘
         │
         v
┌─────────────────────────────────────────┐
│      UnifiedGraphBuilder                │
│  (Plugin Registry: property_graph,      │
│   lightrag, autoschema, ontology, ...)  │
└──────────────┬──────────────────────────┘
               │
               v
        ┌──────────────┐
        │ NetworkX Hub │  ← 統一中轉格式
        │  (GraphML)   │
        └──────┬───────┘
               │
               v
┌─────────────────────────────────────────┐
│     UnifiedGraphRetriever               │
│  (Plugin Registry: property_graph,      │
│   lightrag, tog, csr, ...)              │
└─────────────────────────────────────────┘
```

## 快速開始

### 1. 使用 PropertyGraph（多 extractor）

```python
from src.graph_builder.unified import UnifiedGraphBuilder
from src.graph_adapter import GraphFormatAdapter

# 建立 Builder，組合多個 extractors
builder = UnifiedGraphBuilder(
    settings=Settings,
    builder_type="property_graph",
    builder_config={
        "extractors": {
            "implicit": {"enabled": True},
            "schema": {
                "enabled": True,
                "entities": ["Person", "Org", "Location"],
                "relations": ["WORKS_AT", "LOCATED_IN"],
                "strict": False
            },
            "simple": {
                "enabled": True,
                "max_paths_per_chunk": 10
            }
        }
    },
    data_type="DI",
    fast_test=False
)

# 建圖（統一輸出 NetworkX）
result = builder.build(documents)
nx_graph = result["graph_data"]

# 儲存為 GraphML（可持久化）
GraphFormatAdapter.save_graphml(nx_graph, "my_graph.graphml")
```

### 2. 查看可用的 Builder 和 Retriever

```python
from src.graph_builder.unified import GraphBuilderRegistry
from src.graph_retriever.unified import GraphRetrieverRegistry

print("可用 Builders:", GraphBuilderRegistry.list_available())
# 輸出: ['lightrag', 'autoschema', 'ontology', 'baseline', 'property_graph']

print("可用 Retrievers:", GraphRetrieverRegistry.list_available())
# 輸出: ['lightrag', 'tog', 'csr', 'property_graph']
```

### 3. 跨格式組合（AutoSchemaKG + PropertyGraph Retriever）

```python
# 使用 AutoSchemaKG 建圖
builder = UnifiedGraphBuilder(
    settings=Settings,
    builder_type="autoschema",
    builder_config={
        "output_dir": "/path/to/output",
        "batch_size_triple": 3
    }
)

result = builder.build(documents)
nx_graph = result["graph_data"]  # NetworkX 作為中轉

# 使用 PropertyGraph Retriever 檢索（自動格式轉換）
from src.graph_retriever.unified.property_graph import PropertyGraphRetriever

retriever = PropertyGraphRetriever(
    graph_source=nx_graph,  # 自動轉為 PropertyGraphIndex
    settings=Settings,
    retriever_config={
        "vector": {"enabled": True, "similarity_top_k": 5},
        "synonym": {"enabled": True}
    },
    combination_mode="ensemble"
)

result = retriever.retrieve(query="測試問題", top_k=3)
```

### 4. 新增自定義 Builder

```python
from src.graph_builder.base_builder import BaseGraphBuilder
from src.graph_builder.unified import GraphBuilderRegistry

# 1. 實作自定義 Builder
class MyCustomBuilder(BaseGraphBuilder):
    def get_name(self) -> str:
        return "MyCustom"
    
    def build(self, documents):
        # 自定義建圖邏輯
        nodes = [...]
        edges = [...]
        
        return {
            "nodes": nodes,
            "edges": edges,
            "graph_format": "custom",
            "metadata": {...},
            "schema_info": {...}
        }

# 2. 註冊到 Registry
GraphBuilderRegistry.register("my_custom", MyCustomBuilder)

# 3. 直接使用
builder = UnifiedGraphBuilder(
    settings=Settings,
    builder_type="my_custom"
)
```

## 已實作功能

### Phase 1: 圖譜格式轉換層 ✅
- ✅ `GraphFormatAdapter` 基類
- ✅ PropertyGraph ⟷ NetworkX 轉換
- ✅ Neo4j ⟷ NetworkX 轉換
- ✅ LightRAG ⟷ NetworkX 轉換
- ✅ GraphML 持久化

### Phase 2: Builder System ✅
- ✅ `GraphBuilderRegistry` - Plugin Registry
- ✅ `PropertyGraphExtractorFactory` - 4 種 extractors
  - ImplicitPathExtractor
  - SchemaLLMPathExtractor  
  - SimpleLLMPathExtractor
  - DynamicLLMPathExtractor
- ✅ `PropertyGraphBuilder` - 多 extractor 並行
- ✅ `UnifiedGraphBuilder` - 統一 Wrapper

### Phase 3: Retriever System ✅
- ✅ `GraphRetrieverRegistry` - Plugin Registry
- ✅ `PropertyGraphRetrieverFactory` - 3 種 retrievers
  - VectorContextRetriever
  - LLMSynonymRetriever
  - TextToCypherRetriever
- ✅ `PropertyGraphRetriever` - ensemble/cascade/single 模式

## 核心組件

### 1. GraphFormatAdapter

統一的格式轉換器，使用 NetworkX 作為 Hub：

```python
# 任意格式 → NetworkX
nx_graph = GraphFormatAdapter.to_networkx(source, source_format="property_graph")

# NetworkX → 任意格式
pg_index = GraphFormatAdapter.from_networkx(nx_graph, target_format="property_graph", settings=Settings)

# GraphML 持久化
GraphFormatAdapter.save_graphml(nx_graph, "graph.graphml")
nx_graph = GraphFormatAdapter.load_graphml("graph.graphml")
```

### 2. GraphBuilderRegistry

Plugin Registry System for Builders：

```python
# 註冊新 Builder
GraphBuilderRegistry.register("my_method", MyBuilder)

# 建立 Builder
builder = GraphBuilderRegistry.create("my_method", settings=Settings, ...)

# 列出可用 Builders
builders = GraphBuilderRegistry.list_available()
```

### 3. PropertyGraphExtractorFactory

支援多 extractor 並行：

```python
extractors = PropertyGraphExtractorFactory.create_extractors(
    settings,
    extractor_config={
        "implicit": {"enabled": True},
        "schema": {
            "enabled": True,
            "entities": ["Person", "Org"],
            "relations": ["WORKS_AT"]
        },
        "simple": {"enabled": True, "max_paths_per_chunk": 10},
        "dynamic": {"enabled": True, "max_triplets_per_chunk": 20}
    }
)
```

### 4. PropertyGraphRetrieverFactory

支援多 retriever 組合：

```python
sub_retrievers = PropertyGraphRetrieverFactory.create_retrievers(
    pg_index,
    settings,
    retriever_config={
        "vector": {"enabled": True, "similarity_top_k": 5},
        "synonym": {"enabled": True, "include_text": True},
        "text2cypher": {"enabled": False}
    }
)
```

## 檔案結構

```
End_to_End_RAG/
├── src/
│   ├── graph_adapter/              # 格式轉換層
│   │   ├── __init__.py
│   │   ├── base_adapter.py
│   │   └── converters/
│   │       ├── pg_converter.py
│   │       ├── neo4j_converter.py
│   │       ├── lightrag_converter.py
│   │       └── graphml_handler.py
│   │
│   ├── graph_builder/
│   │   ├── unified/                # Unified Builder
│   │   │   ├── __init__.py
│   │   │   ├── builder_registry.py
│   │   │   ├── unified_builder.py
│   │   │   └── property_graph/
│   │   │       ├── extractor_factory.py
│   │   │       └── pg_builder.py
│   │   └── ...（現有 builders）
│   │
│   └── graph_retriever/
│       ├── unified/                # Unified Retriever
│       │   ├── __init__.py
│       │   ├── retriever_registry.py
│       │   └── property_graph/
│       │       ├── retriever_factory.py
│       │       └── pg_retriever.py
│       └── ...（現有 retrievers）
```

## 設計決策

1. **NetworkX 作為中轉站**：避免 N×(N-1) 的轉換組合爆炸
2. **Plugin Registry**：無需修改核心程式碼即可擴展
3. **統一輸出格式**：所有 Builder 統一輸出 NetworkX
4. **格式自動轉換**：Retriever 自動轉換輸入格式
5. **向後相容**：現有 Builder/Retriever 仍可直接使用
6. **GraphML 持久化**：避免重複建圖

## 擴展性範例

### 新增 GraphRAG 2.0

```python
class GraphRAG2Builder(BaseGraphBuilder):
    def get_name(self) -> str:
        return "GraphRAG2"
    
    def build(self, documents):
        # GraphRAG 2.0 建圖邏輯
        ...
        return {"nodes": nodes, "edges": edges, "graph_format": "graphrag2", ...}

# 註冊
GraphBuilderRegistry.register("graphrag2", GraphRAG2Builder)

# 立即可用
builder = UnifiedGraphBuilder(
    settings=Settings,
    builder_type="graphrag2"
)
```

### 新增 HybridRetriever

```python
class HybridRetriever(BaseGraphRetriever):
    def get_name(self) -> str:
        return "Hybrid"
    
    def retrieve(self, query, graph_data, top_k):
        # 混合檢索邏輯
        ...

# 註冊
GraphRetrieverRegistry.register("hybrid", HybridRetriever)
```

## 性能優化

1. **快取轉換結果**：避免重複格式轉換
2. **GraphML 持久化**：避免重複建圖
3. **並行 extractor**：LlamaIndex 原生支援
4. **批次處理**：大規模圖譜匯入 Neo4j

## 未來工作

1. ⏸️ 更新 `ModularGraphWrapper` 支援格式轉換
2. ⏸️ 整合到 `run_evaluation.py`
3. ⏸️ 更新 `config.yml` 配置段
4. ⏸️ 端到端測試與單元測試

## 技術棧

- **LlamaIndex**: PropertyGraphIndex, extractors, retrievers
- **NetworkX**: 圖譜中轉格式
- **Neo4j** (可選): 圖資料庫
- **LightRAG** (可選): 替代建圖方法

## 貢獻

歡迎貢獻新的 Builder/Retriever 實作！

1. 繼承 `BaseGraphBuilder` 或 `BaseGraphRetriever`
2. 實作必要方法
3. 註冊到對應的 Registry
4. 提交 PR

## License

MIT
