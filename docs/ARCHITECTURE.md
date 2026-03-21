# End_to_End_RAG 系統架構文檔

本文檔詳細說明 End_to_End_RAG 專案的系統架構、模組關係和資料流。

## 目錄

- [系統整體架構](#系統整體架構)
- [CLI 旗標與 Graph 執行路徑對照](#cli-旗標與-graph-執行路徑對照)
- [核心模組說明](#核心模組說明)
- [資料流圖](#資料流圖)
- [類別關係圖](#類別關係圖)
- [模組化 Pipeline 架構](#模組化-pipeline-架構)
- [儲存管理架構](#儲存管理架構)
- [評估系統架構](#評估系統架構)

---

## 系統整體架構

```mermaid
graph TB
    subgraph userInterface[使用者介面層]
        CLI[run_evaluation.py<br/>命令列介面]
        Interactive[run_comprehensive_tests.py<br/>互動式測試]
        Batch[run_all_experiments.sh<br/>批次執行]
        PGBatch[run_all_propertygraphindex.py<br/>PropertyGraph組合掃描]
    end
    
    subgraph configLayer[配置層]
        ConfigYML[config.yml<br/>全域配置]
        Settings[ModelSettings<br/>設定管理]
    end
    
    subgraph dataLayer[資料處理層]
        DataLoader[QADataLoader<br/>資料載入]
        DataProcessor[DataProcessor<br/>資料處理]
        Documents[Documents<br/>LlamaIndex格式]
    end
    
    subgraph ragLayer[RAG 處理層]
        VectorRAG[Vector RAG<br/>向量檢索]
        LightRAGE2E[LightRAG 端到端<br/>graph_rag_method]
        UnifiedPG[統一 PropertyGraph<br/>UnifiedGraphBuilder加Retriever]
        UnifiedLR[統一 LightRAG<br/>unified_graph_type lightrag]
        ModularPipeline[ModularGraphWrapper<br/>preset 或 builder 加 retriever]
        TemporalRAG[Temporal LightRAG<br/>可選旗標]
    end
    
    subgraph evaluationLayer[評估層]
        Evaluator[RAGEvaluator<br/>評估引擎]
        Metrics[Metrics<br/>評估指標]
        Reporter[Reporter<br/>報告生成]
    end
    
    subgraph storageLayer[儲存層]
        VectorIndex[Vector Index<br/>向量索引]
        GraphIndex[Graph Index<br/>圖譜索引]
        SchemaCache[Schema Cache<br/>Schema快取]
    end
    
    CLI --> Settings
    Interactive --> Settings
    Batch --> CLI
    PGBatch --> CLI
    
    Settings --> DataLoader
    ConfigYML --> Settings
    
    DataLoader --> DataProcessor
    DataProcessor --> Documents
    
    Documents --> VectorRAG
    Documents --> LightRAGE2E
    Documents --> UnifiedPG
    Documents --> UnifiedLR
    Documents --> ModularPipeline
    Documents --> TemporalRAG
    
    VectorRAG --> Evaluator
    LightRAGE2E --> Evaluator
    UnifiedPG --> Evaluator
    UnifiedLR --> Evaluator
    ModularPipeline --> Evaluator
    TemporalRAG --> Evaluator
    
    Evaluator --> Metrics
    Metrics --> Reporter
    
    VectorRAG -.持久化.-> VectorIndex
    LightRAGE2E -.持久化.-> GraphIndex
    UnifiedPG -.持久化.-> GraphIndex
    UnifiedLR -.持久化.-> GraphIndex
    ModularPipeline -.持久化.-> GraphIndex
    LightRAGE2E -.使用.-> SchemaCache
```

**說明**：舊版端到端 `--graph_rag_method propertyindex`／`dynamic_schema`／`autoschema` 已棄用（僅印遷移提示），評估主路徑為上圖之 `UnifiedPG` 或模組化組合。細節見下節與 [README.md](README.md)（文檔索引）。

---

## CLI 旗標與 Graph 執行路徑對照

| 路徑 | 主要旗標 | 實作入口（`run_evaluation.py`） |
|------|-----------|-------------------------------|
| LightRAG 端到端 | `--graph_rag_method lightrag`（及 `--lightrag_mode` 等） | `setup_lightrag_pipeline` |
| 統一 PropertyGraph | `--unified_graph_type property_graph`、`--pg_extractors`、`--pg_retrievers`、`--pg_combination_mode` | `setup_unified_graph_pipeline` → `UnifiedGraphBuilder` + `UnifiedGraphRetriever` → `ModularGraphWrapper` |
| 統一 LightRAG | `--unified_graph_type lightrag` | 同上，`builder_type=lightrag` |
| 模組化 Builder+Retriever | `--graph_preset` 或同時 `--graph_builder` + `--graph_retriever` | `setup_modular_graph_pipeline` → `PipelineFactory` |
| 時序 LightRAG | `--lightrag_temporal_graph` | `TemporalLightRAGWrapper`（別名 `TemporalWrapper`） |

**已棄用**：`--graph_rag_method autoschema`、`dynamic_schema`、`propertyindex`。

**未實作**：`PipelineFactory.create_retriever("neo4j")` 會拋出 `NotImplementedError`。

**Legacy**：舊版 `AutoSchemaWrapper`、`DynamicSchemaWrapper` 仍保留於 `legacy/wrappers/`，現行 `src/rag/wrappers/` 不再匯出。

**深入閱讀 PropertyGraph 統一架構**：[PROPERTYGRAPH_REFACTOR_README.md](../PROPERTYGRAPH_REFACTOR_README.md)。

---

## 核心模組說明

### 1. 配置管理模組 (`src/config/`)

**職責**: 管理全域配置和模型設定

```mermaid
classDiagram
    class ModelSettings {
        +ollama_url: str
        +embed_model: str
        +llm_model: str
        +from_yaml()
        +get_embed_model()
        +get_llm()
    }
    
    class get_settings {
        +返回全域設定實例
    }
    
    get_settings --> ModelSettings
```

**核心檔案**:
- `settings.py`: ModelSettings 類別，載入 config.yml

---

### 2. 資料處理模組 (`src/data/`)

**職責**: 載入和處理 QA 資料集

```mermaid
classDiagram
    class QADataLoader {
        +load_csv()
        +load_jsonl()
        +normalize_qa_format()
    }
    
    class DataProcessor {
        +process_data()
        +natural_text_mode()
        +markdown_mode()
        +key_value_mode()
    }
    
    QADataLoader --> DataProcessor : 提供原始資料
    DataProcessor --> LlamaIndexDocuments : 轉換為
```

**核心檔案**:
- `loaders.py`: 資料載入器
- `processors.py`: 資料處理器

**支援格式**:
- CSV (舊格式)
- JSONL (新格式)

**支援模式**:
- natural_text
- markdown
- key_value_text
- unstructured_text

---

### 3. RAG 核心模組 (`src/rag/`)

#### 3.1 Wrappers 子模組 (`src/rag/wrappers/`)

**職責**: 統一封裝所有 RAG 方法

```mermaid
classDiagram
    class BaseRAGWrapper {
        <<abstract>>
        +name: str
        +schema_info: dict
        +retrieval_max_tokens: int
        +_execute_query()* 
        +query()
        +_truncate_contexts_by_tokens()
    }
    
    class VectorRAGWrapper {
        +query_engine
        +_execute_query()
    }
    
    class LightRAGWrapper {
        +lightrag_instance
        +mode: str
        +_execute_query()
    }
    
    class ModularGraphWrapper {
        +builder
        +retriever
        +_execute_query()
    }
    
    class TemporalLightRAGWrapper {
        +rag_instance
        +mode: str
        +_execute_query()
    }
    
    BaseRAGWrapper <|-- VectorRAGWrapper
    BaseRAGWrapper <|-- LightRAGWrapper
    BaseRAGWrapper <|-- ModularGraphWrapper
    BaseRAGWrapper <|-- TemporalLightRAGWrapper
```

**Legacy（僅追溯，不從 `src.rag.wrappers` 匯出）**: `legacy/wrappers/autoschema_wrapper.py`、`legacy/wrappers/dynamic_schema_wrapper.py` 之 `AutoSchemaWrapper`、`DynamicSchemaWrapper`。

**統一功能**:
- ✅ 時間計算
- ✅ Token 統計
- ✅ 錯誤處理
- ✅ Context 截斷
- ✅ 標準化輸出

---

#### 3.2 Vector RAG 子模組 (`src/rag/vector/`)

**職責**: 向量檢索方法實作

```mermaid
graph LR
    Documents[Documents] --> VectorStore[Vector Store<br/>持久化]
    Documents --> BM25Index[BM25 Index]
    
    VectorStore --> HybridRetriever[Hybrid Retriever<br/>Vector + BM25]
    VectorStore --> VectorRetriever[Vector Retriever]
    BM25Index --> BM25Retriever[BM25 Retriever]
    BM25Index --> HybridRetriever
    
    HybridRetriever --> QueryEngine[Query Engine]
    VectorRetriever --> QueryEngine
    BM25Retriever --> QueryEngine
```

**實作方法**:
- `basic.py`: hybrid, vector, bm25
- `advanced.py`: self_query, parent_child

---

#### 3.3 Graph RAG 子模組 (`src/rag/graph/`)

**職責**: 圖譜檢索方法實作

```mermaid
graph TB
    Documents[Documents] --> PropertyGraph[Property Graph]
    Documents --> DynamicSchema[Dynamic Schema Graph]
    Documents --> LightRAGGraph[LightRAG Graph]
    
    PropertyGraph --> PropertyRetriever[Property Graph Retriever]
    DynamicSchema --> DynamicRetriever[Dynamic Retriever]
    LightRAGGraph --> LightRAGRetriever[LightRAG Retriever<br/>7種模式]
    
    LightRAGRetriever --> LocalMode[Local Mode]
    LightRAGRetriever --> GlobalMode[Global Mode]
    LightRAGRetriever --> HybridMode[Hybrid Mode]
    LightRAGRetriever --> MixMode[Mix Mode]
```

**實作檔案**:
- `property_graph.py`: PropertyGraph 方法
- `dynamic_schema.py`: DynamicSchema 方法
- `lightrag.py`: LightRAG 整合
- `lightrag_id_mapper.py`: Chunk ID 映射
- `autoschema_lightrag.py`: AutoSchemaKG
- `temporal_lightrag.py`: 時序 LightRAG

---

#### 3.4 Schema 管理子模組 (`src/rag/schema/`)

**職責**: Schema 生成和管理

```mermaid
graph TB
    SchemaFactory[Schema Factory] --> DefaultSchema[Default Schema<br/>預定義]
    SchemaFactory --> EvolutionSchema[Evolution Schema<br/>迭代演化]
    SchemaFactory --> LLMDynamicSchema[LLM Dynamic Schema<br/>LLM生成]
    SchemaFactory --> AutoSchema[Auto Schema<br/>自動學習]
    
    EvolutionSchema --> SchemaCache[Schema Cache<br/>快取管理]
    LLMDynamicSchema --> SchemaCache
    AutoSchema --> SchemaCache
    
    SchemaCache --> CacheFiles[(快取檔案<br/>JSON)]
```

**核心檔案**:
- `factory.py`: Schema 工廠
- `schema_cache.py`: 快取管理
- `evolution.py`: 演化式生成
- `convergence.py`: 收斂檢測
- `entity_disambiguation.py`: 實體消歧

---

### 4. 模組化 Pipeline (`src/graph_builder/` + `src/graph_retriever/`)

**架構**: Builder + Retriever 分離設計

```mermaid
graph LR
    subgraph builders[Builders建圖]
        AutoBuilder[AutoSchema Builder]
        LightBuilder[LightRAG Builder]
        PropertyBuilder[Property Builder]
        DynamicBuilder[Dynamic Builder]
    end
    
    subgraph graphData[Graph Data中間層]
        GraphDataObj[GraphData Object<br/>標準格式]
    end
    
    subgraph retrievers[Retrievers檢索]
        LightRetriever[LightRAG Retriever]
        CSRRetriever[CSR Retriever]
        Neo4jRetriever[Neo4j Retriever]
    end
    
    Documents[Documents] --> AutoBuilder
    Documents --> LightBuilder
    Documents --> PropertyBuilder
    Documents --> DynamicBuilder
    
    AutoBuilder --> GraphDataObj
    LightBuilder --> GraphDataObj
    PropertyBuilder --> GraphDataObj
    DynamicBuilder --> GraphDataObj
    
    GraphDataObj --> LightRetriever
    GraphDataObj --> CSRRetriever
    GraphDataObj --> Neo4jRetriever
    
    LightRetriever --> Results[Query Results]
    CSRRetriever --> Results
    Neo4jRetriever --> Results
```

**註**：`neo4j` Retriever 於 `PipelineFactory.create_retriever` 尚未實作（`NotImplementedError`），圖示僅表示規劃介面。

**預設組合**:
1. AutoSchema + LightRAG
2. LightRAG + CSR
3. DynamicSchema + CSR
4. DynamicSchema + LightRAG

---

### 5. 評估系統 (`src/evaluation/`)

**職責**: 完整的評估指標計算和報告生成

```mermaid
graph TB
    subgraph metricsModule[評估指標模組]
        RetrievalMetrics[Retrieval Metrics<br/>Hit Rate, MRR, F1]
        GenerationMetrics[Generation Metrics<br/>ROUGE, BLEU, BERTScore]
        LLMJudge[LLM Judge<br/>Correctness, Faithfulness]
    end
    
    subgraph evaluator[評估引擎]
        RAGEvaluator[RAGEvaluator<br/>主評估器]
        MetricRegistry[Metric Registry<br/>指標註冊]
    end
    
    subgraph reporter[報告生成]
        GlobalReporter[Global Summary Reporter]
        DetailedReporter[Detailed Results Reporter]
    end
    
    RAGEvaluator --> MetricRegistry
    MetricRegistry --> RetrievalMetrics
    MetricRegistry --> GenerationMetrics
    MetricRegistry --> LLMJudge
    
    RAGEvaluator --> GlobalReporter
    RAGEvaluator --> DetailedReporter
    
    GlobalReporter --> CSVReport[global_summary_report.csv]
    GlobalReporter --> XLSXReport[global_summary_report.xlsx]
    DetailedReporter --> DetailedCSV[detailed_results.csv]
    GlobalReporter -.可選彙總.-> MasterXLSX[results/exp/資料類別/global_summary.xlsx]
```

**指標類別**:
1. **檢索指標** (`metrics/retrieval.py`)
   - Hit Rate
   - MRR
   - Precision / Recall / F1

2. **生成指標** (`metrics/generation.py`)
   - ROUGE (1/2/L/Lsum)
   - BLEU
   - METEOR
   - BERTScore
   - Token F1 / Jieba F1

3. **LLM-as-Judge** (`metrics/llm_judge.py`)
   - Correctness (0-5)
   - Faithfulness (0/1)

---

### 6. 插件系統 (`src/plugins/`)

**職責**: 可擴展的 KG 增強功能

```mermaid
classDiagram
    class BaseKGPlugin {
        <<abstract>>
        +name: str
        +version: str
        +apply()*
        +validate()
    }
    
    class PluginRegistry {
        +_plugins: dict
        +register()
        +get_plugin()
        +list_plugins()
    }
    
    class AutoSchemaPlugin {
        +apply()
    }
    
    class DynamicPathPlugin {
        +apply()
    }
    
    class GraphitiPlugin {
        +apply()
        +時序感知
    }
    
    BaseKGPlugin <|-- AutoSchemaPlugin
    BaseKGPlugin <|-- DynamicPathPlugin
    BaseKGPlugin <|-- GraphitiPlugin
    BaseKGPlugin <|-- CQDrivenPlugin
    BaseKGPlugin <|-- Neo4jPlugin
    
    PluginRegistry --> BaseKGPlugin : 管理
```

**已實作插件架構**:
- `autoschema_plugin.py`
- `dynamic_path_plugin.py`
- `graphiti_plugin.py`
- `cq_driven_plugin.py`
- `neo4j_builder_plugin.py`

**狀態**：`autoschema_plugin`、`dynamic_path_plugin` 可掛載於 LightRAG 流程；`graphiti_plugin`、`cq_driven_plugin`、`neo4j_builder_plugin` 多為占位或 TODO，與「已實作」模組分開看待。

---

### 7. 儲存管理 (`src/storage/`)

**職責**: 統一的 storage 目錄管理

```mermaid
graph TB
    StorageManager[Storage Manager<br/>統一管理器] --> VectorDir[vector_index/<br/>向量索引]
    StorageManager --> LightRAGDir[lightrag/<br/>LightRAG圖譜]
    StorageManager --> GraphDir[graph_index/<br/>其他圖索引]
    StorageManager --> CSRDir[csr_graph/<br/>CSR快取]
    StorageManager --> SchemaDir[schema_cache/<br/>Schema快取]
    StorageManager --> AutoSchemaDir[autoschema/<br/>AutoSchemaKG 輸出]
    
    VectorDir --> HybridIndex[DI_hybrid/]
    VectorDir --> VectorIndex[DI_vector/]
    VectorDir --> BM25Index[DI_bm25/]
    
    LightRAGDir --> DefaultLR[DI_lightrag_default/]
    LightRAGDir --> EvolutionLR[DI_iterative_evolution/]
    LightRAGDir --> FastTestLR[*_fast_test/]
    
    SchemaDir --> MethodCache[method_name/]
    MethodCache --> DatasetCache[dataset_type/<br/>JSON檔案]
```

**目錄結構**:
```
storage/
├── vector_index/       # Vector 索引
├── graph_index/        # Graph 索引
├── lightrag/          # LightRAG 圖譜
├── autoschema/        # AutoSchemaKG；子目錄 slug = data_type[_data_mode][_sup][_fast_test]
├── csr_graph/         # CSR Graph 快取
├── cache/             # 其他快取
├── lightrag_temporal/ # 可選：Temporal LightRAG（config lightrag_storage_path_DIR）
└── schema_cache/      # Schema 快取
```

---

## 資料流圖

### 完整評估流程

```mermaid
sequenceDiagram
    participant User
    participant CLI as run_evaluation.py
    participant Config as ModelSettings
    participant Data as DataLoader
    participant RAG as RAG Wrapper
    participant Eval as RAGEvaluator
    participant Report as Reporter
    
    User->>CLI: 執行命令
    CLI->>Config: 載入配置
    Config->>Data: 初始化資料載入器
    Data->>Data: 載入 QA 資料集
    Data->>Data: 處理為 Documents
    
    CLI->>RAG: 建立 RAG Pipeline
    RAG->>RAG: 建立索引（如需要）
    
    loop 每個問題
        CLI->>RAG: query(question)
        RAG->>RAG: 檢索 contexts
        RAG->>RAG: 生成 answer
        RAG-->>CLI: 返回結果
    end
    
    CLI->>Eval: 評估所有結果
    Eval->>Eval: 計算檢索指標
    Eval->>Eval: 計算生成指標
    Eval->>Eval: LLM-as-Judge
    
    Eval->>Report: 生成報告
    Report->>Report: 全域摘要
    Report->>Report: 詳細結果
    Report-->>User: 儲存 CSV 報告
```

---

### Vector RAG 資料流

```mermaid
graph LR
    RawData[原始資料<br/>JSONL/CSV] --> Processor[Data Processor<br/>轉換格式]
    Processor --> Documents[LlamaIndex<br/>Documents]
    
    Documents --> Embedding[Embedding Model<br/>nomic-embed-text]
    Embedding --> VectorStore[Vector Store<br/>FAISS/持久化]
    
    Documents --> BM25Builder[BM25 Builder]
    BM25Builder --> BM25Index[BM25 Index]
    
    Query[使用者查詢] --> Embedding
    Embedding --> VectorRetrieval[向量檢索]
    VectorStore --> VectorRetrieval
    
    Query --> BM25Retrieval[BM25檢索]
    BM25Index --> BM25Retrieval
    
    VectorRetrieval --> Fusion[Fusion<br/>混合檢索]
    BM25Retrieval --> Fusion
    
    Fusion --> Contexts[Retrieved Contexts]
    Contexts --> LLM[LLM生成答案<br/>qwen2.5:7b]
    LLM --> Answer[最終答案]
```

---

### LightRAG 資料流

```mermaid
graph TB
    subgraph indexing[索引建立階段]
        RawDocs[原始文檔] --> Chunking[文檔切塊]
        Chunking --> EntityExtraction[實體抽取<br/>使用Schema]
        EntityExtraction --> RelationExtraction[關係抽取]
        RelationExtraction --> GraphStorage[圖譜儲存<br/>JSON格式]
    end
    
    subgraph schemaGen[Schema生成可選]
        SchemaMethod{Schema方法}
        SchemaMethod -->|default| PredefinedSchema[預定義Schema<br/>136類型]
        SchemaMethod -->|evolution| EvolutionSchema[演化式Schema<br/>迭代優化]
        SchemaMethod -->|llm_dynamic| LLMSchema[LLM動態Schema<br/>自動生成]
        
        EvolutionSchema --> SchemaCache[(Schema Cache)]
        LLMSchema --> SchemaCache
    end
    
    subgraph retrieval[檢索階段]
        UserQuery[使用者查詢] --> ModeRouter{檢索模式}
        
        ModeRouter -->|local| LocalSearch[Local Search<br/>實體+關係檢索]
        ModeRouter -->|global| GlobalSearch[Global Search<br/>社群摘要檢索]
        ModeRouter -->|hybrid| HybridSearch[Hybrid Search<br/>混合檢索]
        ModeRouter -->|mix| MixSearch[Mix Search<br/>圖譜+向量]
        
        LocalSearch --> ContextAgg[Context Aggregation]
        GlobalSearch --> ContextAgg
        HybridSearch --> ContextAgg
        MixSearch --> ContextAgg
    end
    
    GraphStorage --> LocalSearch
    GraphStorage --> GlobalSearch
    GraphStorage --> HybridSearch
    GraphStorage --> MixSearch
    
    ContextAgg --> LLMGen[LLM生成最終答案]
    LLMGen --> FinalAnswer[最終答案]
    
    SchemaCache -.影響.-> EntityExtraction
```

---

## 類別關係圖

### RAG Wrapper 繼承關係

```mermaid
classDiagram
    class BaseRAGWrapper {
        <<abstract>>
        +name: str
        +schema_info: dict
        +retrieval_max_tokens: int
        +_execute_query()*
        +query()
        +set_retrieval_max_tokens()
        +_truncate_contexts_by_tokens()
        +_count_tokens()
    }
    
    class VectorRAGWrapper {
        +query_engine
        +mode: str
        +_execute_query()
    }
    
    class LightRAGWrapper {
        +lightrag_instance
        +mode: str
        +schema_method: str
        +chunk_id_mapper
        +_execute_query()
        +_extract_retrieved_ids()
    }
    
    class ModularGraphWrapper {
        +builder: BaseGraphBuilder
        +retriever: BaseGraphRetriever
        +graph_data
        +_execute_query()
        +build_graph()
    }
    
    class TemporalLightRAGWrapper {
        +rag_instance
        +mode: str
        +_execute_query()
    }
    
    BaseRAGWrapper <|-- VectorRAGWrapper
    BaseRAGWrapper <|-- LightRAGWrapper
    BaseRAGWrapper <|-- ModularGraphWrapper
    BaseRAGWrapper <|-- TemporalLightRAGWrapper
```

**Legacy**：`AutoSchemaWrapper`、`DynamicSchemaWrapper` 見 `legacy/wrappers/`。

---

### Builder-Retriever 關係

```mermaid
classDiagram
    class BaseGraphBuilder {
        <<abstract>>
        +name: str
        +build()*
    }
    
    class BaseGraphRetriever {
        <<abstract>>
        +name: str
        +retrieve()*
    }
    
    class AutoSchemaKGBuilder {
        +build()
        +run_extraction()
        +generate_concepts()
        +convert_to_graphml()
    }
    
    class LightRAGBuilder {
        +build()
        +build_lightrag_index()
    }
    
    class LightRAGRetriever {
        +retrieve()
        +supports multiple modes
    }
    
    class CSRRetriever {
        +retrieve()
        +graph_traversal()
    }
    
    BaseGraphBuilder <|-- AutoSchemaKGBuilder
    BaseGraphBuilder <|-- LightRAGBuilder
    BaseGraphBuilder <|-- PropertyBuilder
    BaseGraphBuilder <|-- DynamicSchemaBuilder
    
    BaseGraphRetriever <|-- LightRAGRetriever
    BaseGraphRetriever <|-- CSRRetriever
    BaseGraphRetriever <|-- Neo4jRetriever
    
    PipelineFactory ..> BaseGraphBuilder : 使用
    PipelineFactory ..> BaseGraphRetriever : 使用
    ModularGraphWrapper ..> BaseGraphBuilder : 組合
    ModularGraphWrapper ..> BaseGraphRetriever : 組合
```

---

### 評估指標繼承關係

```mermaid
classDiagram
    class BaseMetric {
        <<abstract>>
        +name: str
        +compute()*
        +aggregate()
    }
    
    class HitRate {
        +compute()
    }
    
    class MRR {
        +compute()
    }
    
    class RetrievalF1 {
        +compute()
    }
    
    class RougeScore {
        +compute()
        +rouge_1, rouge_2, rouge_L
    }
    
    class BLEUScore {
        +compute()
    }
    
    class BERTScore {
        +compute()
        +uses bert_score library
    }
    
    class Correctness {
        +compute()
        +uses LLM for judging
    }
    
    class Faithfulness {
        +compute()
        +uses LLM for judging
    }
    
    BaseMetric <|-- HitRate
    BaseMetric <|-- MRR
    BaseMetric <|-- RetrievalF1
    BaseMetric <|-- RougeScore
    BaseMetric <|-- BLEUScore
    BaseMetric <|-- BERTScore
    BaseMetric <|-- Correctness
    BaseMetric <|-- Faithfulness
    
    MetricRegistry ..> BaseMetric : 註冊與管理
    RAGEvaluator ..> MetricRegistry : 使用
```

---

## 模組化 Pipeline 架構

### 設計理念

**實務上的三種 Graph 入口**（與 CLI 對照表一致）:
1. **LightRAG 端到端**：`--graph_rag_method lightrag`
2. **統一 Graph**：`--unified_graph_type property_graph` 或 `lightrag`（`UnifiedGraphBuilder` + `UnifiedGraphRetriever`）
3. **模組化**：`--graph_preset` 或 `--graph_builder` + `--graph_retriever`（`PipelineFactory`）

```mermaid
graph TB
    subgraph endToEnd[端到端與統一Graph]
        E2E_Vector[Vector RAG<br/>vector_method]
        E2E_LightRAG[LightRAG<br/>graph_rag_method lightrag]
        UnifiedPG[統一 PropertyGraph<br/>unified_graph_type property_graph]
    end
    
    subgraph modular[模組化模式]
        SelectBuilder[選擇 Builder] --> BuildGraph[建立圖譜]
        BuildGraph --> GraphData[Graph Data<br/>標準格式]
        GraphData --> SelectRetriever[選擇 Retriever]
        SelectRetriever --> Retrieve[執行檢索]
    end
    
    User{使用者選擇}
    User -->|向量基線| E2E_Vector
    User -->|LightRAG| E2E_LightRAG
    User -->|多extractor組合| UnifiedPG
    User -->|Builder乘Retriever實驗| modular
    
    endToEnd --> Results[評估結果]
    Retrieve --> Results
```

---

### PipelineFactory 運作機制

```mermaid
sequenceDiagram
    participant User
    participant Factory as PipelineFactory
    participant Builder
    participant GraphData
    participant Retriever
    
    User->>Factory: create_modular_pipeline(builder, retriever)
    Factory->>Factory: 驗證組合有效性
    
    Factory->>Builder: initialize builder
    Builder->>Builder: build graph
    Builder->>GraphData: 產生 GraphData 物件
    
    Factory->>Retriever: initialize retriever
    Retriever->>GraphData: 載入 graph_data
    
    Factory-->>User: 返回 Wrapper
    
    User->>Retriever: query(question)
    Retriever->>Retriever: 檢索圖譜
    Retriever-->>User: 返回結果
```

**預設組合表**:

| 組合名稱 | Builder | Retriever | 特點 |
|---------|---------|-----------|------|
| autoschema_lightrag | AutoSchemaKG | LightRAG | 自動 Schema + 強檢索 |
| lightrag_csr | LightRAG | CSR | LightRAG建圖 + 圖遍歷 |
| dynamic_csr | DynamicSchema | CSR | 動態Schema + 圖遍歷 |
| dynamic_lightrag | DynamicSchema | LightRAG | 動態Schema + LightRAG檢索 |

---

## 儲存管理架構

### Storage Manager 設計

```mermaid
graph TB
    subgraph api[Storage Manager API]
        GetVectorPath[get_vector_index_path]
        GetLightRAGPath[get_lightrag_storage_path]
        GetGraphPath[get_graph_index_path]
        GetCSRPath[get_csr_graph_cache_path]
        GetSchemaPath[get_schema_cache_path]
    end
    
    subgraph structure[統一目錄結構]
        RootDir[storage/]
        RootDir --> VectorDir[vector_index/]
        RootDir --> LightRAGDir[lightrag/]
        RootDir --> GraphDir[graph_index/]
        RootDir --> CSRDir[csr_graph/]
        RootDir --> SchemaDir[schema_cache/]
    end
    
    GetVectorPath --> VectorDir
    GetLightRAGPath --> LightRAGDir
    GetGraphPath --> GraphDir
    GetCSRPath --> CSRDir
    GetSchemaPath --> SchemaDir
    
    VectorDir --> Isolation[fast_test 隔離]
    LightRAGDir --> Isolation
    GraphDir --> Isolation
```

**設計優勢**:
1. ✅ 集中管理，避免路徑分散
2. ✅ 完整的 fast_test 隔離
3. ✅ 支援索引持久化
4. ✅ 自動目錄創建

---

## 評估系統架構

### 評估流程詳細設計

```mermaid
graph TB
    subgraph preparation[準備階段]
        LoadQA[載入 QA 資料集] --> InitWrapper[初始化 RAG Wrapper]
        InitWrapper --> BuildIndex[建立索引如需要]
    end
    
    subgraph execution[執行階段]
        BuildIndex --> QueryLoop[逐題查詢]
        QueryLoop --> CollectResults[收集結果]
    end
    
    subgraph evaluation[評估階段]
        CollectResults --> CalcRetrieval[計算檢索指標<br/>Hit Rate, MRR, F1]
        CollectResults --> CalcGeneration[計算生成指標<br/>ROUGE, BLEU, BERTScore]
        CollectResults --> CalcLLMJudge[LLM-as-Judge<br/>Correctness, Faithfulness]
        
        CalcRetrieval --> Aggregate[聚合指標]
        CalcGeneration --> Aggregate
        CalcLLMJudge --> Aggregate
    end
    
    subgraph reporting[報告階段]
        Aggregate --> GlobalReport[全域摘要<br/>CSV與XLSX]
        Aggregate --> DetailedReport[詳細結果<br/>detailed_results.csv]
        Aggregate --> TokenAnalysis[Token分析<br/>可選]
    end
```

---

### 指標計算流程

```mermaid
sequenceDiagram
    participant Evaluator as RAGEvaluator
    participant Registry as MetricRegistry
    participant Metric
    participant Results
    
    Evaluator->>Registry: get_metric("hit_rate")
    Registry-->>Evaluator: HitRate instance
    
    loop 每個問題
        Evaluator->>Metric: compute(ground_truth, retrieved)
        Metric-->>Evaluator: score
        Evaluator->>Results: 記錄分數
    end
    
    Evaluator->>Metric: aggregate(all_scores)
    Metric-->>Evaluator: 平均分數
    
    Evaluator->>Results: 寫入 CSV
```

---

## 擴展性設計

### 新增 RAG 方法

```mermaid
graph LR
    Step1[1. 實作 Wrapper<br/>繼承 BaseRAGWrapper] --> Step2[2. 實作 _execute_query]
    Step2 --> Step3[3. 註冊到 Factory]
    Step3 --> Step4[4. 添加 CLI 參數]
    Step4 --> Step5[5. 撰寫文檔]
    Step5 --> Complete[完成✓]
```

### 新增評估指標

```mermaid
graph LR
    Step1[1. 實作 Metric<br/>繼承 BaseMetric] --> Step2[2. 實作 compute 方法]
    Step2 --> Step3[3. 註冊到 Registry]
    Step3 --> Step4[4. 更新 Evaluator]
    Step4 --> Complete[完成✓]
```

### 新增插件

```mermaid
graph LR
    Step1[1. 實作 Plugin<br/>繼承 BaseKGPlugin] --> Step2[2. 實作 apply 方法]
    Step2 --> Step3[3. 註冊到 Registry]
    Step3 --> Step4[4. 測試整合]
    Step4 --> Complete[完成✓]
```

---

## 配置管理流程

```mermaid
graph TB
    ConfigYML[config.yml<br/>YAML配置檔] --> Parser[YAML Parser]
    Parser --> ModelSettings[ModelSettings 物件]
    
    ModelSettings --> OllamaConfig[Ollama 配置<br/>URL, 模型名稱]
    ModelSettings --> DataConfig[資料配置<br/>檔案路徑]
    ModelSettings --> LightRAGConfig[LightRAG 配置<br/>實體類型, 語言]
    ModelSettings --> StorageConfig[Storage 配置<br/>根目錄, 子目錄]
    ModelSettings --> SchemaConfig[Schema 配置<br/>快取, 方法]
    
    OllamaConfig --> LLM[LLM 實例]
    OllamaConfig --> EmbedModel[Embedding 模型]
    
    DataConfig --> DataLoader[資料載入器]
    LightRAGConfig --> LightRAGEngine[LightRAG 引擎]
    StorageConfig --> StorageManager[Storage Manager]
    SchemaConfig --> SchemaFactory[Schema Factory]
```

---

## 總結

### 架構優勢

1. **模組化設計**: 清晰的分層，職責分離
2. **可擴展性**: 抽象類別 + 工廠模式
3. **統一介面**: BaseRAGWrapper 提供一致 API
4. **多種 Graph 入口**: LightRAG 端到端、統一 PropertyGraph、模組化組合
5. **完整評估**: 多層次指標系統
6. **儲存管理**: 集中化、隔離化
7. **插件架構**: 靈活的功能擴展

### 技術棧總覽

| 層次 | 主要技術 |
|------|---------|
| LLM 後端 | Ollama (qwen2.5:7b/14b) |
| Embedding | nomic-embed-text, bge-m3 |
| 向量檢索 | LlamaIndex VectorStore, FAISS |
| 圖譜檢索 | LightRAG, PropertyGraph |
| 評估指標 | rouge-score, bert-score, sacrebleu |
| 資料處理 | pandas, numpy |
| 測試框架 | pytest |
| 非同步 | nest-asyncio |

### 關鍵設計模式

- **工廠模式**: PipelineFactory, SchemaFactory
- **策略模式**: 不同的檢索模式
- **裝飾器模式**: RAG Wrappers
- **觀察者模式**: 評估指標註冊
- **建造者模式**: Graph Builders

---

**最後更新**: 2026-03-21  
**專案路徑**: `/home/End_to_End_RAG`
