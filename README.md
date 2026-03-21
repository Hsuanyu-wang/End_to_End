# End-to-End RAG 評估框架

一個完整、模組化的 RAG (Retrieval-Augmented Generation) Pipeline 評估框架，支援多種檢索方法與全面的評估指標。

## ✨ 最新更新 (2026-03-17)

### 📊 Schema 資訊記錄（NEW）
- 詳細記錄每個 RAG Pipeline 使用的 Schema 資訊
- 支援記錄 entity types、relations、validation schema
- 在評估結果的第一行記錄完整 schema 細節
- 文檔：[Schema 記錄功能說明](docs/SCHEMA_RECORDING.md)

### 🗂️ 集中化 Storage 管理
- 新增統一的 storage 目錄結構 (`/storage/`)
- 所有索引和快取集中管理，支援完整的 fast_test 隔離
- Vector Index 支援持久化，避免重複建立

### 🔍 Retrieval 指標修復
- 修復 LightRAG 的 retrieval 指標計算問題
- 實作 ChunkIDMapper 機制，正確映射 chunk IDs
- 所有 retrieval 指標（Hit Rate, MRR, Precision, Recall, F1）現可正常計算

### 🔌 插件系統（`src/plugins/`）
- 可擴展插件架構；**實際整合程度因插件而異**
- **可掛載於 LightRAG 流程**（`--lightrag_plugins`）：`autoschema`、`dynamic_path` 等（見下方「插件系統」一節）
- **多為 stub／TODO 或僅列印提示**：`graphiti`、`cq_driven`、`neo4j` 對應類別存在，**非**五項皆已端到端可用；端到端 `graphiti`／`neo4j`／`cq_driven` 於 `--graph_rag_method` 仍為架構預留

## 專案特色

- ✨ **模組化設計**：清晰的程式碼結構，易於維護與擴展
- 🎯 **多種 RAG 方法**：支援 Vector RAG、Graph RAG、LightRAG 等多種方法
- 📊 **全面評估**：涵蓋檢索指標、生成指標、LLM-as-Judge 評估
- ⚡ **高效能**：支援非同步執行與快取機制
- 🔧 **易於擴展**：基於抽象類別與工廠模式，方便新增新方法或指標
- 🗂️ **統一 Storage**：集中化的 storage 管理，完整隔離測試環境
- 🔌 **插件系統**：可擴展的 KG 增強插件架構

## 快速開始

### 安裝依賴

```bash
pip install -r requirements.txt
```

### 互動式測試（推薦）

使用新的互動式測試腳本，可靈活選擇要執行的測試：

```bash
python scripts/run_comprehensive_tests.py
```

提供多種測試選項：
- Vector RAG、Advanced Vector RAG、Graph RAG
- LightRAG Schema 方法、LightRAG 檢索模式
- 快速測試 vs 完整測試
- 單一或批次執行

詳細使用說明請參考 [測試指南](docs/TESTING_GUIDE.md)。

### 基本使用範例

#### 1. Vector RAG 評估

```bash
python scripts/run_evaluation.py \
    --vector_method hybrid \
    --data_type DI \
    --top_k 2
```

#### 2. LightRAG 評估

```bash
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode hybrid \
    --data_type DI
```

#### 3. 快速測試模式

```bash
python scripts/run_evaluation.py \
    --vector_method hybrid \
    --qa_dataset_fast_test
```

#### 4. 統一 PropertyGraph（多 extractor／retriever）

`--graph_rag_method autoschema`、`dynamic_schema`、`propertyindex` 已棄用，請改用：

```bash
python scripts/run_evaluation.py \
    --unified_graph_type property_graph \
    --pg_extractors implicit,schema,simple,dynamic \
    --pg_retrievers vector,synonym \
    --pg_combination_mode ensemble \
    --data_type DI
```

詳見 [文檔索引：Graph 對照](docs/README.md) 與 [PropertyGraph 重構說明](PROPERTYGRAPH_REFACTOR_README.md)。

#### 5. 模組化組合

```bash
# 使用預設組合
python scripts/run_evaluation.py \
    --graph_preset autoschema_lightrag \
    --lightrag_mode hybrid \
    --data_type DI

# 自訂組合
python scripts/run_evaluation.py \
    --graph_builder lightrag \
    --graph_retriever csr \
    --data_type DI
```

### 批次實驗

執行所有 RAG 方法的完整實驗：

```bash
bash scripts/run_all_experiments.sh
```

## 專案結構

```
End_to_End_RAG/
├── src/                      # 核心源碼
│   ├── config/               # 配置管理
│   ├── data/                 # 資料處理
│   ├── rag/                  # RAG Pipeline 實作
│   │   ├── wrappers/         # RAG 封裝器
│   │   ├── vector/           # Vector RAG
│   │   ├── graph/            # Graph RAG
│   │   └── schema/           # Schema 管理
│   ├── plugins/              # 插件系統 (NEW)
│   ├── storage/              # Storage 管理 (NEW)
│   ├── graph_builder/        # 圖譜建構
│   ├── graph_retriever/      # 圖譜檢索
│   ├── evaluation/           # 評估系統
│   │   └── metrics/          # 評估指標
│   └── utils/                # 工具函數
├── storage/                  # 統一 Storage 目錄 (NEW)
│   ├── vector_index/         # Vector 索引
│   ├── graph_index/          # Graph 索引
│   ├── lightrag/             # LightRAG 圖譜
│   ├── csr_graph/            # CSR Graph cache
│   └── cache/                # 其他快取
├── scripts/                  # 執行腳本
│   ├── run_evaluation.py     # 主執行入口
│   └── migrate_storage.py    # Storage 遷移工具 (NEW)
├── tests/                    # 測試
├── docs/                     # 文檔
├── Data/                     # 資料目錄
├── results/                  # 評估結果
└── config.yml                # 全域配置
```

## 支援的 RAG 方法

### Vector RAG

- **Hybrid**: Vector + BM25 混合檢索
- **Vector**: 純向量檢索
- **BM25**: 基於 BM25 的關鍵字檢索

### 進階 Vector RAG

- **Self-Query**: 帶有 metadata 過濾的自查詢
- **Parent-Child**: 多層級檢索（Auto Merging）

### Graph RAG（建議入口）

- **LightRAG 端到端**：`--graph_rag_method lightrag`（local/global/hybrid/mix/naive/bypass/original 等）
- **統一 PropertyGraph**：`--unified_graph_type property_graph` + `--pg_extractors` / `--pg_retrievers` / `--pg_combination_mode`（取代已棄用的 `--graph_rag_method propertyindex` / `dynamic_schema` / `autoschema`）
- **統一 LightRAG（模組化包裝）**：`--unified_graph_type lightrag`
- **Graphiti / Neo4j / CQ-Driven（`--graph_rag_method`）**：架構預留，執行時僅警告
- **時序 LightRAG**：`--lightrag_temporal_graph`（`TemporalLightRAGWrapper`）

### Graph RAG (模組化)

靈活組合 Builder + Retriever:

**Builders**:
- `autoschema`: AutoSchemaKG 建圖
- `lightrag`: LightRAG 建圖
- `dynamic`: DynamicSchema 建圖
- `property`: PropertyGraph 建圖

**Retrievers**:
- `lightrag`: LightRAG 多模式檢索
- `csr`: CSR Graph 遍歷檢索
- `neo4j`: Neo4j Cypher 查詢 (待實作)

**預設組合**:
- `autoschema_lightrag`: AutoSchemaKG Builder + LightRAG Retriever
- `lightrag_csr`: LightRAG Builder + CSR Retriever
- `dynamic_csr`: DynamicSchema Builder + CSR Retriever
- `dynamic_lightrag`: DynamicSchema Builder + LightRAG Retriever

詳見 [模組化 Pipeline 文檔](docs/MODULAR_PIPELINE.md)

## 評估指標

### 檢索指標

- **Hit Rate**: 命中率
- **MRR**: 平均倒數排名
- **Precision**: 精準度
- **Recall**: 召回率
- **F1 Score**: F1 分數

### 生成指標

- **ROUGE**: ROUGE-1/2/L/Lsum
- **BLEU**: BLEU 分數
- **METEOR**: METEOR 分數
- **BERTScore**: 基於 BERT embeddings 的語義相似度
- **Token F1**: 字元層級 F1
- **Jieba F1**: 詞彙層級 F1（使用 jieba 分詞）

### LLM-as-Judge

- **Correctness**: 答案正確性（0-5 分）
- **Faithfulness**: 答案忠實度（0/1 分）

## 命令列參數

### RAG 方法選擇

```bash
--vector_method {none,hybrid,vector,bm25,all}
--adv_vector_method {none,parent_child,self_query,all}
--graph_rag_method {none,propertyindex,lightrag,dynamic_schema,autoschema,graphiti,neo4j,cq_driven,all}
# 註：propertyindex / dynamic_schema / autoschema 已棄用，請用 --unified_graph_type property_graph 或模組化參數
--unified_graph_type {none,property_graph,lightrag}
--pg_extractors implicit,schema,simple,dynamic   # 僅 property_graph
--pg_retrievers vector,synonym,text2cypher       # 僅 property_graph
--pg_combination_mode {ensemble,cascade,single}  # 僅 property_graph
--lightrag_mode {none,local,global,hybrid,mix,naive,bypass,original,all}
--lightrag_native_mode {local,global,hybrid,mix,naive,bypass}  # 僅 original 時：官方 aquery 的 mode，預設 hybrid
```

### 模組化 Pipeline 參數

```bash
--graph_builder {autoschema,lightrag,property,dynamic}
--graph_retriever {lightrag,csr,neo4j}   # neo4j：PipelineFactory 尚未實作，會拋錯
--graph_preset {autoschema_lightrag,lightrag_csr,dynamic_csr,dynamic_lightrag}
```

### 資料參數

```bash
--data_type {DI,GEN}              # 資料類型
--data_mode {natural_text,markdown,key_value_text,unstructured_text}  # 資料格式
--model_type {small,big}          # 模型大小
```

### 評估參數

```bash
--top_k 2                         # 檢索數量
--qa_dataset_fast_test            # 快速測試（僅前 2 題）
--postfix my_experiment           # 結果資料夾後綴
--sup cache_tag                   # Cache 標籤 (用於區分不同實驗的 storage)
```

### Storage 管理

新版本使用集中化的 storage 管理：

```bash
# 所有 storage 統一在專案下 storage/ 目錄（預設 /home/End_to_End_RAG/storage）
storage/
├── vector_index/DI_hybrid/              # Vector 索引（持久化）
├── graph_index/DI_propertyindex/        # Graph 索引
├── lightrag/DI_lightrag_default/        # LightRAG 圖譜
├── autoschema/DI_natural_text/          # AutoSchemaKG（slug = data_type[_data_mode][_sup][_fast_test]）
├── lightrag_temporal/                   # Temporal LightRAG（見 config lightrag_storage_path_DIR）
└── csr_graph/DI_natural_text_khop.pkl   # CSR Graph cache
```

### 遷移舊 Storage

如果你有舊版的 storage，可使用遷移工具：

```bash
# 試運行（不實際移動檔案）
python scripts/migrate_storage.py --dry-run

# 實際執行遷移
python scripts/migrate_storage.py
```

### 插件系統

啟用 KG 增強插件：

```bash
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode hybrid \
    --lightrag_plugins autoschema,dynamic_path \
    --data_type DI
```

可用插件名稱（逗號分隔）：
- `autoschema`、`dynamic_path`：可掛載於 LightRAG 流程
- `graphiti`、`cq_driven`、`neo4j`：類別存在，**多為占位／TODO**，行為以原始碼 `src/plugins/` 為準

### Schema 相關

```bash
--lightrag_schema_method {lightrag_default,iterative_evolution,llm_dynamic,llamaindex_dynamic}
--lightrag_temporal_graph         # 啟用時序 LightRAG
```

完整參數說明請參考 [API 文檔](docs/API.md)。

## 評估結果

評估結果依 `--data_type`（如 DI、GEN）寫入 `results/{exp|test}/{資料類別}/evaluation_results_{timestamp}{postfix}/`：

```
results/exp/DI/evaluation_results_20260317_120000_DI_lightrag_hybrid/
├── global_summary_report.csv                  # 各 Pipeline 平均指標（CSV）
├── global_summary_report.xlsx                 # 同上（Excel）
├── LightRAG_Hybrid/
│   └── detailed_results.csv                   # 逐題評估結果
└── Vector_hybrid_RAG/
    └── detailed_results.csv

results/exp/{資料類別}/（與上層同目錄；跨執行彙總）
└── global_summary.xlsx                        # 多次實驗附加彙總（該資料類別內）
```

快速測試模式路徑為 `results/test/{資料類別}/`，結構相同。

### 報告內容

- **detailed_results.csv**：每題的詳細評估結果（含所有指標）
- **global_summary_report.csv** / **global_summary_report.xlsx**：單次執行內各 Pipeline 平均指標
- **global_summary.xlsx**（可選）：該 `資料類別` 下跨執行附加彙總（見 `src/evaluation/reporters.py`）

## 進階使用

### 自訂評估指標

```python
from src.evaluation.metrics import BaseMetric, MetricRegistry

@MetricRegistry.register("custom_metric")
class CustomMetric(BaseMetric):
    def compute(self, **kwargs):
        # 自訂邏輯
        return score
```

### 新增 RAG Pipeline

```python
from src.rag.wrappers import BaseRAGWrapper

class MyRAGWrapper(BaseRAGWrapper):
    async def _execute_query(self, query: str):
        # 實作查詢邏輯
        return {
            "generated_answer": answer,
            "retrieved_contexts": contexts,
            "retrieved_ids": ids,
            "source_nodes": nodes,
        }
```

詳細說明請參考 [API 文檔](docs/API.md)。

## 測試

```bash
# 執行所有測試
python -m pytest tests/

# 執行特定測試
python -m pytest tests/test_metrics.py
```

## 貢獻指南

1. Fork 此專案
2. 建立 feature 分支 (`git checkout -b feature/AmazingFeature`)
3. 提交變更 (`git commit -m 'Add some AmazingFeature'`)
4. Push 到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 授權

此專案採用 MIT 授權 - 詳見 LICENSE 檔案

## 文檔

- **[文檔索引（請從此進入）](docs/README.md)**
- [系統架構](docs/ARCHITECTURE.md)
- [指令參考](docs/COMMAND_REFERENCE.md)
- [API 參考](docs/API.md)
- [評估指標說明](docs/METRICS.md)
- [模組化 Pipeline 架構](docs/MODULAR_PIPELINE.md)
- [PropertyGraph 統一架構](PROPERTYGRAPH_REFACTOR_README.md)
- [Schema 記錄功能](docs/SCHEMA_RECORDING.md)
- [測試與實驗指南](docs/TESTING_GUIDE.md)
- [使用範例](docs/EXAMPLES.md)
- [改進模組紀錄（精簡）](docs/IMPLEMENTATION_HISTORY.md)

## 聯絡方式

如有任何問題或建議，歡迎開啟 Issue 討論。
