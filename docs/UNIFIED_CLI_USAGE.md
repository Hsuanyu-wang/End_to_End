# End-to-End RAG 評估框架：統一 CLI 使用手冊

## Quick Start

```bash
cd /home/End_to_End_RAG

# LightRAG hybrid（最常用）
python scripts/run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --data_type DI

# Vector baseline
python scripts/run_evaluation.py --vector_method hybrid --data_type DI

# 快速測試（僅前 2 題）
python scripts/run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --qa_dataset_fast_test
```

---

## Architecture Overview

框架由四層組成：**Builder → Plugin → Retriever → Strategy**。
所有 Graph 實驗透過 `--graph_type` + `--graph_retrieval` 兩個核心參數控制。

```
┌──────────────────────────────────────────────────────┐
│                    CLI 入口                           │
│  --graph_type        --graph_retrieval               │
│  lightrag            native / ppr / pcst / tog       │
│  property_graph      pg_ensemble / pg_cascade        │
│  autoschema          native                          │
│  dynamic             native                          │
└──────────────┬───────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │   Builder Layer      │
    │  LightRAG Builder    │    --schema_method
    │  PropertyGraph Builder│   --pg_extractors
    │  AutoSchema Builder  │
    │  Dynamic Builder     │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │   Plugin Layer       │
    │  SimMerge            │    --plugin_simmerge
    │  Temporal            │    --plugin_temporal
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │   Retriever Layer    │
    │  LightRAG Native     │    --lightrag_mode
    │  Strategy-based      │    --graph_retrieval ppr/pcst/tog
    │  PropertyGraph       │    --pg_retrievers
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  Traversal Strategy  │
    │  PPR / PCST / ToG    │    --ppr_alpha / --pcst_cost_mode / --tog_*
    │  OneHop              │
    └─────────────────────┘
```

---

## CLI Reference

### Core: Graph 類型與檢索策略

| 參數 | 型別 | 預設值 | 可選值 | 說明 |
|------|------|--------|--------|------|
| `--graph_type` | str | `none` | `none`, `lightrag`, `property_graph`, `autoschema`, `dynamic` | Graph Builder 類型 |
| `--graph_retrieval` | str | `native` | `native`, `one_hop`, `ppr`, `pcst`, `tog`, `pg_ensemble`, `pg_cascade`, `pg_single` | 檢索策略 |

### LightRAG 模式

| 參數 | 型別 | 預設值 | 可選值 | 說明 |
|------|------|--------|--------|------|
| `--lightrag_mode` | str | `hybrid` | `hybrid`, `local`, `global`, `mix`, `naive`, `bypass`, `original`, `all` | LightRAG 檢索模式 |
| `--lightrag_native_mode` | str | `hybrid` | `local`, `global`, `hybrid`, `mix`, `naive`, `bypass` | `original` 模式下官方 aquery 使用的 mode |

**模式說明：**
- `hybrid/local/global/mix/naive/bypass`：以 `only_need_context=True` 取結構化 context，再以自訂 prompt + LLM 生成
- `original`：單次官方 `aquery`（端到端），不另匯出 retrieved_contexts
- `all`：展開上述六種 context 模式各跑一次

### Schema / Extractor

| 參數 | 型別 | 預設值 | 可選值 | 說明 |
|------|------|--------|--------|------|
| `--schema_method` | str | `lightrag_default` | `lightrag_default`, `iterative_evolution`, `llm_dynamic`, `llamaindex_dynamic` | Schema 生成方法 |
| `--pg_extractors` | str | `implicit,schema,simple` | 逗號分隔：`implicit`, `schema`, `simple`, `dynamic` | PropertyGraph extractors |
| `--pg_retrievers` | str | `vector,synonym` | 逗號分隔：`vector`, `synonym`, `text2cypher` | PropertyGraph retrievers |
| `--pg_combination_mode` | str | `ensemble` | `ensemble`, `cascade`, `single` | PropertyGraph retriever 組合模式 |

### Plugins

| 參數 | 型別 | 預設值 | 說明 |
|------|------|--------|------|
| `--plugin_simmerge` | flag | - | 啟用相似實體合併 (Similar Entity Merge) |
| `--simmerge_threshold` | float | config.yml | 餘弦相似度下界 |
| `--simmerge_threshold_max` | float | config.yml | 餘弦相似度上界（不含） |
| `--simmerge_text_mode` | str | config.yml | `name` 或 `name_desc`，embedding 用文本模式 |
| `--simmerge_force_recopy` | flag | - | 強制重新複製 storage |
| `--simmerge_dry_run` | flag | - | 僅產生 log 不實際合併 |
| `--plugin_temporal` | flag | - | 啟用時序圖譜 (Temporal Graph) |

### Traversal Strategy 參數

| 參數 | 型別 | 預設值 | 說明 |
|------|------|--------|------|
| `--ppr_alpha` | float | `0.85` | PPR damping factor |
| `--ppr_weight_mode` | str | `semantic` | `semantic`, `degree`, `combined` |
| `--pcst_cost_mode` | str | `inverse_weight` | `inverse_weight`, `inverse_log_weight`, `uniform` |
| `--tog_max_iterations` | int | `3` | ToG 最大迭代次數 |
| `--tog_beam_width` | int | `5` | ToG beam search 寬度 |

### Data / Model

| 參數 | 型別 | 預設值 | 說明 |
|------|------|--------|------|
| `--data_type` | str | `DI` | 資料類型（DI、GEN 等） |
| `--data_mode` | str | `natural_text` | `natural_text`, `markdown`, `key_value_text`, `unstructured_text` |
| `--model_type` | str | `small` | `small`, `big` |
| `--top_k` | int | `20` | 檢索數量 |

### Token Control

| 參數 | 型別 | 預設值 | 說明 |
|------|------|--------|------|
| `--retrieval_max_tokens` | int | `0` | 檢索內容最大 token 數（0=不限制） |
| `--enable_token_budget` | flag | - | 啟用 token budget 控制 |
| `--token_budget_baseline` | str | `vector_hybrid` | baseline 方法名稱 |

### Debug / Test

| 參數 | 型別 | 預設值 | 說明 |
|------|------|--------|------|
| `--qa_dataset_fast_test` | flag | - | 快速測試（僅前 2 題，自動連動 build fast test） |
| `--vector_build_fast_test` | flag | - | Vector 索引快速建立 |
| `--graph_build_fast_test` | flag | - | Graph 索引快速建立 |
| `--postfix` | str | `""` | 結果資料夾名稱後綴 |
| `--sup` | str | `""` | 快取方法標識 |

---

## Experiment Recipes

### LightRAG 基本實驗

```bash
# Hybrid 模式（最常用：context + 自訂生成）
python scripts/run_evaluation.py \
    --graph_type lightrag --lightrag_mode hybrid --data_type DI

# Original 模式（官方端到端 aquery）
python scripts/run_evaluation.py \
    --graph_type lightrag --lightrag_mode original --lightrag_native_mode hybrid --data_type DI

# 展開所有六種 context 模式
python scripts/run_evaluation.py \
    --graph_type lightrag --lightrag_mode all --data_type DI
```

### LightRAG + Traversal Strategy

```bash
# PPR (Personalized PageRank)
python scripts/run_evaluation.py \
    --graph_type lightrag --lightrag_mode hybrid \
    --graph_retrieval ppr --ppr_alpha 0.85 --ppr_weight_mode semantic \
    --data_type DI

# PCST (Prize-Collecting Steiner Tree)
python scripts/run_evaluation.py \
    --graph_type lightrag --lightrag_mode hybrid \
    --graph_retrieval pcst --pcst_cost_mode inverse_weight \
    --data_type DI

# ToG (Think-on-Graph)
python scripts/run_evaluation.py \
    --graph_type lightrag --lightrag_mode hybrid \
    --graph_retrieval tog --tog_max_iterations 3 --tog_beam_width 5 \
    --data_type DI
```

### LightRAG + SimMerge Plugin

```bash
# 基本相似實體合併
python scripts/run_evaluation.py \
    --graph_type lightrag --lightrag_mode hybrid \
    --plugin_simmerge --simmerge_threshold 0.85 --simmerge_text_mode name_desc \
    --data_type DI

# 帶上界的相似實體合併
python scripts/run_evaluation.py \
    --graph_type lightrag --lightrag_mode hybrid \
    --plugin_simmerge --simmerge_threshold 0.85 --simmerge_threshold_max 0.95 \
    --simmerge_text_mode name_desc \
    --data_type DI
```

### LightRAG Schema 方法比較

```bash
# 預設 Schema
python scripts/run_evaluation.py \
    --graph_type lightrag --lightrag_mode hybrid --schema_method lightrag_default

# Iterative Evolution Schema
python scripts/run_evaluation.py \
    --graph_type lightrag --lightrag_mode hybrid --schema_method iterative_evolution

# LLM Dynamic Schema
python scripts/run_evaluation.py \
    --graph_type lightrag --lightrag_mode hybrid --schema_method llm_dynamic
```

### PropertyGraph 實驗

```bash
# Ensemble 模式（預設）
python scripts/run_evaluation.py \
    --graph_type property_graph \
    --pg_extractors implicit,schema,simple \
    --pg_retrievers vector,synonym \
    --data_type DI

# 含 text2cypher
python scripts/run_evaluation.py \
    --graph_type property_graph \
    --pg_extractors implicit,schema,simple \
    --pg_retrievers vector,synonym,text2cypher \
    --pg_combination_mode cascade \
    --data_type DI
```

### AutoSchema / Dynamic Builder

```bash
# AutoSchema builder + LightRAG retriever
python scripts/run_evaluation.py \
    --graph_type autoschema --data_type DI

# Dynamic builder + LightRAG retriever
python scripts/run_evaluation.py \
    --graph_type dynamic --data_type DI
```

### Temporal Graph

```bash
python scripts/run_evaluation.py \
    --graph_type lightrag --lightrag_mode hybrid \
    --plugin_temporal --data_type DI
```

### Token Budget 控制

```bash
python scripts/run_evaluation.py \
    --graph_type lightrag --lightrag_mode hybrid \
    --enable_token_budget --token_budget_baseline vector_hybrid \
    --data_type DI
```

### Vector RAG

```bash
# Hybrid vector
python scripts/run_evaluation.py --vector_method hybrid --data_type DI

# 所有 vector 方法
python scripts/run_evaluation.py --vector_method all --data_type DI

# 進階 vector
python scripts/run_evaluation.py --adv_vector_method parent_child --data_type DI
```

### 組合實驗

```bash
# Vector + LightRAG 同時評估
python scripts/run_evaluation.py \
    --vector_method hybrid \
    --graph_type lightrag --lightrag_mode hybrid \
    --data_type DI

# 帶 retrieval token 限制的公平比較
python scripts/run_evaluation.py \
    --vector_method hybrid \
    --graph_type lightrag --lightrag_mode hybrid \
    --retrieval_max_tokens 4096 \
    --data_type DI
```

---

## Plugin Reference

### SimMerge (Similar Entity Merge)

合併 LightRAG 圖譜中的相似實體，減少重複節點。

**啟用方式：** `--plugin_simmerge`

**參數：**
- `--simmerge_threshold`：餘弦相似度下界（含），預設從 `config.yml` 讀取
- `--simmerge_threshold_max`：餘弦相似度上界（不含），限制合併範圍
- `--simmerge_text_mode`：用於計算 embedding 的文本來源
  - `name`：僅使用實體名稱
  - `name_desc`：使用名稱 + 描述
- `--simmerge_force_recopy`：強制刪除 plugin storage 後重新從 baseline 複製
- `--simmerge_dry_run`：僅產生合併 log，不實際執行合併

**運作原理：**
1. 複製 baseline LightRAG storage 到帶 custom_tag 的目錄
2. 計算所有實體對的餘弦相似度
3. 在 threshold 範圍內的實體對呼叫 `merge_entities` 合併
4. 查詢時指向合併後的 storage

### Temporal (Temporal Graph)

**啟用方式：** `--plugin_temporal`

為 LightRAG 加入時序處理能力，在插入文本時加上時間戳記，
查詢時可依時序排序結果。

### Dynamic Path (已實作但 deprecated)

透過 `DynamicLLMPathExtractor` 動態發現實體類型和關係。
建議改用 `--graph_type property_graph --pg_extractors dynamic`。

---

## Migration Guide

從舊版 CLI 遷移到新版統一 CLI：

| 舊指令 | 新指令 |
|--------|--------|
| `--graph_rag_method lightrag --lightrag_mode hybrid` | `--graph_type lightrag --lightrag_mode hybrid` |
| `--graph_rag_method lightrag --lightrag_mode original --lightrag_native_mode hybrid` | `--graph_type lightrag --lightrag_mode original --lightrag_native_mode hybrid` |
| `--graph_rag_method lightrag --graph_traversal_strategy ppr` | `--graph_type lightrag --graph_retrieval ppr` |
| `--unified_graph_type lightrag --lightrag_mode hybrid` | `--graph_type lightrag --lightrag_mode hybrid` |
| `--unified_graph_type property_graph --pg_extractors implicit,schema` | `--graph_type property_graph --pg_extractors implicit,schema` |
| `--graph_preset autoschema_lightrag` | `--graph_type autoschema` |
| `--graph_preset dynamic_lightrag` | `--graph_type dynamic` |
| `--lightrag_plugins similar_entity_merge --lightrag_sim_merge_threshold 0.85` | `--plugin_simmerge --simmerge_threshold 0.85` |
| `--lightrag_sim_merge_text_mode name_desc` | `--simmerge_text_mode name_desc` |
| `--lightrag_temporal_graph` | `--plugin_temporal` |
| `--lightrag_schema_method iterative_evolution` | `--schema_method iterative_evolution` |

> 舊參數仍可使用（向後相容），系統會自動映射到新參數並印出遷移提示。

---

## Config File (`config.yml`)

| 區塊 | 說明 |
|------|------|
| `model` | 主 LLM 設定（Ollama URL、模型名稱、embedding） |
| `eval_model` | 評估用 LLM |
| `builder_model` | 建圖用 LLM |
| `data` | 資料路徑（JSONL、QA CSV） |
| `lightrag` | LightRAG storage 路徑、語言、實體類型、plugins 預設值 |
| `storage` | 各種 storage 根目錄 |
| `schema` | Schema cache、方法、disambiguation 設定 |

SimMerge 的預設值可在 `config.yml` 的 `lightrag.plugins.similar_entity_merge` 區塊設定。

---

## Batch Scripts

| 腳本 | 說明 |
|------|------|
| `scripts/run_all_experiments.sh` | 完整實驗批次（Vector + LightRAG + PropertyGraph + AutoSchema） |
| `scripts/run_all_hybrid.sh` | LightRAG original + hybrid 快速比較 |
| `scripts/example_fair_comparison.sh` | Vector vs LightRAG 公平比較（含 retrieval_max_tokens） |
