# RAG 方法完整使用指令參考

本文檔提供 End_to_End_RAG 專案中所有 RAG 方法的詳細使用指令。

## 目錄

- [Vector RAG 方法](#vector-rag-方法)
- [LightRAG 方法](#lightrag-方法)
- [Graph RAG 方法](#graph-rag-方法)
- [模組化組合方法](#模組化組合方法)
- [進階功能與參數](#進階功能與參數)
- [快速參考表](#快速參考表)

---

## Vector RAG 方法

### 1. Vector Hybrid RAG（推薦）

**說明**: 結合向量檢索與 BM25 關鍵字檢索的混合方法，平衡語義相似度與關鍵字匹配。

**使用指令**:
```bash
python scripts/run_evaluation.py \
    --vector_method hybrid \
    --data_type DI \
    --top_k 20
```

**主要參數**:
- `--top_k`: 檢索文件數量（預設 20）
- `--data_type`: 資料集類型（DI 或 GEN）
- `--retrieval_max_tokens`: 限制檢索內容的最大 token 數

**適用場景**: 需要同時考慮語義理解和精確關鍵字匹配的查詢

---

### 2. Vector Only RAG

**說明**: 純向量檢索，基於語義相似度進行檢索。

**使用指令**:
```bash
python scripts/run_evaluation.py \
    --vector_method vector \
    --data_type DI \
    --top_k 20
```

**適用場景**: 查詢與文檔在語義上相似但用詞可能不同的情況

---

### 3. BM25 RAG

**說明**: 基於 BM25 演算法的關鍵字檢索，不依賴向量嵌入。

**使用指令**:
```bash
python scripts/run_evaluation.py \
    --vector_method bm25 \
    --data_type DI \
    --top_k 20
```

**適用場景**: 需要精確關鍵字匹配，或向量模型不適用的場景

---

### 4. Self-Query RAG

**說明**: 自動從查詢中提取 metadata 過濾條件，結合向量檢索進行精準查找。

**使用指令**:
```bash
python scripts/run_evaluation.py \
    --adv_vector_method self_query \
    --data_type DI \
    --top_k 20
```

**適用場景**: 查詢包含明確的過濾條件（如時間、類別等）

---

### 5. Parent-Child RAG

**說明**: 階層式檢索，使用小塊檢索但返回完整的父文檔內容。

**使用指令**:
```bash
python scripts/run_evaluation.py \
    --adv_vector_method parent_child \
    --data_type DI \
    --top_k 20
```

**切塊層級**: 2048 → 1024 → 512 tokens

**適用場景**: 需要完整上下文但又要精準定位的查詢

---

## LightRAG 方法

### LightRAG 檢索模式總覽

LightRAG 提供 7 種檢索模式 × 3 種 Schema 生成方法 = 21 種組合。

### Schema 生成方法

#### 1. Default Schema (`lightrag_default`)
**說明**: 使用預定義的 136 個實體類型（從 config.yml 載入）

#### 2. Iterative Evolution Schema (`iterative_evolution`)
**說明**: 透過迭代演化自動學習和優化 Schema

**特點**: 
- 自動收斂檢測
- 支援 Schema Cache
- 最多 50 個實體類型

#### 3. LLM Dynamic Schema (`llm_dynamic`)
**說明**: 使用 LLM 動態生成適合特定領域的 Schema

**特點**:
- 完全自動化
- 支援 Schema Cache
- 領域自適應

---

### LightRAG 檢索模式

#### 1. Local Mode

**說明**: 關注特定實體與細節，適合精確查詢。

**使用指令**:
```bash
# 使用 Default Schema
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode local \
    --lightrag_schema_method lightrag_default \
    --data_type DI

# 使用 Iterative Evolution Schema（推薦）
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode local \
    --lightrag_schema_method iterative_evolution \
    --data_type DI

# 使用 LLM Dynamic Schema
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode local \
    --lightrag_schema_method llm_dynamic \
    --data_type DI
```

**適用場景**: 查詢特定實體或事件的詳細資訊

---

#### 2. Global Mode

**說明**: 關注整體趨勢與總結，適合宏觀查詢。

**使用指令**:
```bash
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode global \
    --lightrag_schema_method iterative_evolution \
    --data_type DI
```

**適用場景**: 需要了解整體趨勢、統計或總結的查詢

---

#### 3. Hybrid Mode（推薦）

**說明**: 混合 Local 和 Global 模式，平衡細節與整體。

**使用指令**:
```bash
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode hybrid \
    --lightrag_schema_method iterative_evolution \
    --data_type DI
```

**適用場景**: 通用查詢，需要同時考慮細節和整體

---

#### 4. Mix Mode

**說明**: 結合知識圖譜檢索與向量檢索。

**使用指令**:
```bash
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode mix \
    --lightrag_schema_method lightrag_default \
    --data_type DI
```

**適用場景**: 需要結構化知識和非結構化文本檢索的複雜查詢

---

#### 5. Naive Mode

**說明**: 僅使用向量檢索，不使用圖譜結構。

**使用指令**:
```bash
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode naive \
    --lightrag_schema_method lightrag_default \
    --data_type DI
```

**適用場景**: 作為基準測試，或圖譜建構不完整時的後備方案

---

#### 6. Bypass Mode

**說明**: 直接將查詢傳給 LLM，不進行檢索（測試用）。

**使用指令**:
```bash
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode bypass \
    --lightrag_schema_method lightrag_default \
    --data_type DI
```

**適用場景**: 測試 LLM 無檢索時的基準效能

---

#### 7. Original Mode

**說明**: 單次官方 `aquery`（`only_need_context=False`），不另匯出結構化檢索 context；與 `local/global/...` 的「先 `query_data` + 自訂 LLM」管線不同。可選 `--lightrag_native_mode` 指定官方查詢的 LightRAG mode（預設 `hybrid`）。

**結果目錄命名**: 該 pipeline 在輸出中的名稱為 `LightRAG_original_{lightrag_native_mode}`（例如 `LightRAG_original_hybrid`），以便與不同 native mode 區分。

**使用指令**:
```bash
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode original \
    --lightrag_native_mode hybrid \
    --lightrag_schema_method lightrag_default \
    --data_type DI
```

**適用場景**: 與自訂 context 管線對照之 LightRAG 內建端到端基線；依賴 `retrieved_contexts` 的指標可能為空

---

## Graph RAG 方法

> **重要（與 `run_evaluation.py` 一致）**  
> `--graph_rag_method` 的 **`propertyindex`、`dynamic_schema`、`autoschema` 已棄用**：若指定，程式僅印遷移提示並**不執行**該 Graph 評估。  
> 請改用 **統一 PropertyGraph**（本節下列）或 **[模組化組合](#模組化組合方法)**（`--graph_preset`／`--graph_builder`+`--graph_retriever`）。  
> `graphiti`、`neo4j`、`cq_driven` 仍為架構預留（警告後跳過核心邏輯）。

### 統一 PropertyGraph（取代舊端到端 propertyindex／dynamic_schema／autoschema）

**說明**：透過 `UnifiedGraphBuilder` 組合多種 extractors，並以 `UnifiedGraphRetriever` 組合多種 retrievers（`ensemble`／`cascade`／`single`）。細節見 [PROPERTYGRAPH_REFACTOR_README.md](../PROPERTYGRAPH_REFACTOR_README.md)。

**使用指令**（與棄用提示中建議參數對齊）:
```bash
python scripts/run_evaluation.py \
    --unified_graph_type property_graph \
    --pg_extractors implicit,schema,simple,dynamic \
    --pg_retrievers vector,synonym \
    --pg_combination_mode ensemble \
    --data_type DI
```

- **`--pg_extractors`**：逗號分隔，`implicit`、`schema`、`simple`、`dynamic`（`dynamic` 對應 DynamicLLMPathExtractor 風格建圖）
- **`--pg_retrievers`**：`vector`、`synonym`、`text2cypher`
- **`--pg_combination_mode`**：`ensemble`、`cascade`、`single`

**模組化替代**（若不需統一多 extractor 管線）:
- 動態 Schema 建圖 + 檢索：`--graph_preset dynamic_lightrag` 或 `--graph_builder dynamic --graph_retriever lightrag`
- AutoSchema 建圖 + LightRAG 檢索：`--graph_preset autoschema_lightrag`
- 純 PropertyGraph builder + LightRAG：`--graph_builder property --graph_retriever lightrag`

---

### 遷移對照（僅供理解，請勿再當作主命令）

| 舊旗標（已棄用） | 建議替代 |
|------------------|----------|
| `--graph_rag_method propertyindex` | `--unified_graph_type property_graph` + 調整 `pg_*` |
| `--graph_rag_method dynamic_schema` | 上列 unified，或 `--graph_preset dynamic_csr`／`dynamic_lightrag` |
| `--graph_rag_method autoschema` | `--graph_preset autoschema_lightrag`，或 unified 管線內含所需 extractor |

---

### Temporal LightRAG

**說明**: 支援時序資訊的 LightRAG 版本。

**使用指令**:
```bash
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode hybrid \
    --lightrag_temporal_graph \
    --data_type DI
```

**特點**: 考慮實體和關係的時間演變

**適用場景**: 時序性強的資料（如事件序列、歷史記錄）

---

## 模組化組合方法

模組化架構允許自由組合 Builder（建圖）+ Retriever（檢索）。

### 預設組合

#### 1. AutoSchemaKG + LightRAG

**說明**: 使用 AutoSchemaKG 建圖，LightRAG 檢索。

**使用指令**:
```bash
python scripts/run_evaluation.py \
    --graph_preset autoschema_lightrag \
    --data_type DI
```

**優勢**: 結合自動 Schema 學習與強大的圖譜檢索

---

#### 2. LightRAG Builder + CSR Retriever

**說明**: LightRAG 建圖，CSR 圖遍歷檢索。

**使用指令**:
```bash
python scripts/run_evaluation.py \
    --graph_preset lightrag_csr \
    --data_type DI
```

**優勢**: 高效的圖遍歷演算法

---

#### 3. DynamicSchema + CSR

**說明**: 動態 Schema 建圖，CSR 檢索。

**使用指令**:
```bash
python scripts/run_evaluation.py \
    --graph_preset dynamic_csr \
    --data_type DI
```

---

#### 4. DynamicSchema + LightRAG

**說明**: 動態 Schema 建圖，LightRAG 檢索。

**使用指令**:
```bash
python scripts/run_evaluation.py \
    --graph_preset dynamic_lightrag \
    --data_type DI
```

---

### 自訂組合

**使用指令**:
```bash
python scripts/run_evaluation.py \
    --graph_builder [autoschema|lightrag|property|dynamic] \
    --graph_retriever [lightrag|csr|neo4j] \
    --data_type DI
```

**註**：`--graph_retriever neo4j` 目前於 `PipelineFactory` 尚未實作，執行會失敗；請使用 `lightrag` 或 `csr`。

**範例**:
```bash
# Property Graph Builder + LightRAG Retriever
python scripts/run_evaluation.py \
    --graph_builder property \
    --graph_retriever lightrag \
    --data_type DI
```

---

## 進階功能與參數

### Token Budget 控制（公平比較）

**說明**: 動態調整 LightRAG 參數，使其 token 使用量與基準方法相同。

**使用指令**:
```bash
python scripts/run_evaluation.py \
    --vector_method hybrid \
    --graph_rag_method lightrag \
    --lightrag_mode hybrid \
    --enable_token_budget \
    --token_budget_baseline vector_hybrid \
    --data_type DI
```

**參數說明**:
- `--enable_token_budget`: 啟用功能
- `--token_budget_baseline`: 基準方法名稱（預設 `vector_hybrid`）

**目的**: 確保不同方法在相同 token 預算下的公平比較

---

### Retrieval Max Tokens 限制

**說明**: 限制檢索內容傳給 LLM 的最大 token 數。

**使用指令**:
```bash
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode hybrid \
    --retrieval_max_tokens 2048 \
    --data_type DI
```

**建議值**: 2048 或 4096

---

### 快速測試模式

**說明**: 僅使用前 2 題 QA 進行快速驗證。

**使用指令**:
```bash
python scripts/run_evaluation.py \
    --vector_method hybrid \
    --qa_dataset_fast_test \
    --vector_build_fast_test \
    --data_type DI
```

**參數說明**:
- `--qa_dataset_fast_test`: 僅評估前 2 題
- `--vector_build_fast_test`: 使用快速索引建立
- `--graph_build_fast_test`: 圖譜快速建立

**預期時間**: 30-60 秒

---

### 批次評估所有方法

**使用指令**:
```bash
bash scripts/run_all_experiments.sh
```

**執行內容**:
- 所有 Vector RAG 方法
- 所有 Advanced Vector RAG 方法
- 主要 Graph RAG 方法
- 所有 LightRAG Schema 方法

**預期時間**: 3-6 小時（完整資料集）

**輸出**:
- 評估結果：`results/exp/{資料類別}/`（`--data_type`，預設 DI）
- 日誌：`experiment_logs/experiment_*.log`
- HTML 報告：`experiment_logs/experiment_*_report.html`

---

### 互動式測試（最推薦）

**使用指令**:
```bash
python scripts/run_comprehensive_tests.py
```

**功能**:
- 選單式介面
- 11 種測試選項
- 支援單一或批次執行
- 自動生成測試總結
- Schema Cache 管理

**適用場景**: 探索性測試、快速驗證、方法比較

---

### 資料格式選擇

**說明**: 支援 4 種資料格式。

**使用指令**:
```bash
python scripts/run_evaluation.py \
    --vector_method hybrid \
    --data_type DI \
    --data_mode [natural_text|markdown|key_value_text|unstructured_text]
```

**格式說明**:
- `natural_text`: 自然語言格式（預設）
- `markdown`: Markdown 格式
- `key_value_text`: 鍵值對格式
- `unstructured_text`: 非結構化文本

---

### 模型大小選擇

**使用指令**:
```bash
python scripts/run_evaluation.py \
    --vector_method hybrid \
    --data_type DI \
    --model_type [small|big]
```

**模型對應**:
- `small`: qwen2.5:7b
- `big`: qwen2.5:14b 或更大模型

---

### LightRAG 插件系統（實驗性）

**使用指令**:
```bash
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode hybrid \
    --lightrag_plugins autoschema,dynamic_path \
    --data_type DI
```

**可用插件**:
- `autoschema`: AutoSchemaKG 增強
- `dynamic_path`: 動態實體類型檢測
- `graphiti`: 時序感知圖譜（開發中）
- `cq_driven`: 能力問題驅動本體（開發中）
- `neo4j`: Neo4j 圖資料庫整合（開發中）

---

## 快速參考表

### Vector RAG 方法對比

| 方法 | 指令 | 優勢 | 適用場景 |
|------|------|------|---------|
| Hybrid | `--vector_method hybrid` | 平衡語義與關鍵字 | 通用查詢 |
| Vector | `--vector_method vector` | 語義理解強 | 語義查詢 |
| BM25 | `--vector_method bm25` | 關鍵字精準 | 精確匹配 |
| Self-Query | `--adv_vector_method self_query` | 自動過濾 | 條件查詢 |
| Parent-Child | `--adv_vector_method parent_child` | 完整上下文 | 需要上下文 |

---

### LightRAG 模式對比

| 模式 | 指令 | 關注點 | 適用場景 |
|------|------|--------|---------|
| Local | `--lightrag_mode local` | 特定實體細節 | 精確查詢 |
| Global | `--lightrag_mode global` | 整體趨勢 | 宏觀查詢 |
| Hybrid | `--lightrag_mode hybrid` | 平衡 | 通用查詢 |
| Mix | `--lightrag_mode mix` | 圖譜+向量 | 複雜查詢 |
| Naive | `--lightrag_mode naive` | 僅向量 | 基準測試 |

---

### Schema 方法對比

| 方法 | 指令 | 特點 | Cache |
|------|------|------|-------|
| Default | `lightrag_default` | 預定義 136 類型 | ❌ |
| Evolution | `iterative_evolution` | 自動演化 | ✅ |
| LLM Dynamic | `llm_dynamic` | LLM 生成 | ✅ |

---

### 常用組合推薦

#### 快速驗證
```bash
python scripts/run_comprehensive_tests.py
# 選擇 1 → 1 → 1 → 1
```

#### 最佳效能組合
```bash
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode hybrid \
    --lightrag_schema_method iterative_evolution \
    --data_type DI \
    --top_k 20
```

#### 公平比較
```bash
python scripts/run_evaluation.py \
    --vector_method hybrid \
    --graph_rag_method lightrag \
    --lightrag_mode hybrid \
    --enable_token_budget \
    --data_type DI
```

#### 完整評估
```bash
bash scripts/run_all_experiments.sh
```

#### PropertyGraph 組合批次掃描（text2cypher）

**說明**：[`scripts/run_all_propertygraphindex.py`](../scripts/run_all_propertygraphindex.py) 以雙層迴圈對 **所有非空 extractor 子集** × **含 `text2cypher` 的 retriever 子集** 呼叫 `run_evaluation.py`（每次一組 `--pg_extractors`／`--pg_retrievers`）。執行次數多、耗時長，請確認資源與 `results/` 空間。

```bash
python scripts/run_all_propertygraphindex.py
```

---

## Schema Cache 管理

### 列出快取

```bash
python scripts/manage_schema_cache.py --list
```

### 清理特定方法的快取

```bash
python scripts/manage_schema_cache.py --clean --method iterative_evolution
```

### 匯出快取報告

```bash
python scripts/manage_schema_cache.py --export --output schema_report.json
```

### 顯示快取統計

```bash
python scripts/manage_schema_cache.py --info
```

---

## 結果查看

### 最新結果

```bash
ls -lt /home/End_to_End_RAG/results/exp/DI/ | head -5
```

### 查看全域摘要

```bash
cat /home/End_to_End_RAG/results/exp/DI/evaluation_results_*/global_summary_report.csv
# 單次執行亦產生 global_summary_report.xlsx；跨執行彙總可見 results/exp/DI/global_summary.xlsx（GEN 實驗則為 results/exp/GEN/...）
```

### 查看詳細結果（含 Schema 資訊）

```bash
head -10 /home/End_to_End_RAG/results/exp/DI/evaluation_results_*/*/detailed_results.csv
```

---

## 故障排除

### Ollama 連接失敗

```bash
# 檢查 Ollama 服務
curl http://192.168.63.174:11434/api/tags
```

### 記憶體不足

使用快速測試模式：
```bash
--qa_dataset_fast_test --vector_build_fast_test
```

### 索引建立錯誤

清空舊索引：
```bash
rm -rf /home/End_to_End_RAG/storage/vector_index/*
rm -rf /home/End_to_End_RAG/storage/lightrag/*
```

---

## 更多資源

- [主要 README](../README.md)
- [測試指南](TESTING_GUIDE.md)
- [API 文檔](API.md)
- [評估指標](METRICS.md)
- [快速開始範例](../QUICKSTART_EXAMPLES.md)
