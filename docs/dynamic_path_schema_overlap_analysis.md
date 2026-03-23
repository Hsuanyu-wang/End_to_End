# DynamicPath vs Schema Methods Overlap Analysis

## Scope

本文件比較 `dynamic_path` plugin 與下列 `lightrag_schema_method`：

- `lightrag_default`
- `iterative_evolution`
- `llm_dynamic`
- `llamaindex_dynamic`

判斷是否存在功能重疊，並給出可刪除/可替代建議。

## Capability Matrix

| Ability | dynamic_path plugin | lightrag_default | iterative_evolution | llm_dynamic | llamaindex_dynamic |
|---|---|---|---|---|---|
| Schema entities 產生 | 部分（`enhance_schema` 只合併已發現類型） | 是（固定清單） | 是（LLM 結構化） | 是（LLM comma list） | 是（Dynamic extractor labels） |
| Relations schema 產生 | 否 | 否 | 是 | 否 | 否 |
| Validation schema 產生 | 否 | 否 | 是 | 否 | 否 |
| Triple 抽取 | 是（`enhance_extraction`） | 否 | 否 | 否 | 間接（只拿 label，不回傳 triples） |
| Graph 後處理 | 是（`post_process_graph`） | 否 | 否 | 否 | 否 |
| 核心依賴 DynamicLLMPathExtractor | 是 | 否 | 否 | 否 | 是 |
| 目前 `run_evaluation.py` 主流程是否接線 | 否 | 是 | 是 | 是 | 是 |

## Current Wiring (Actual Runtime)

### 1) `dynamic_path` plugin

- 定義於 `src/plugins/dynamic_path_plugin.py`，使用 `@register_plugin("dynamic_path")` 註冊。
- 但 `scripts/run_evaluation.py` 只明確 import `similar_entity_merge_plugin`，沒有 import `dynamic_path_plugin`。
- `setup_lightrag_pipeline()` 解析 `--lightrag_plugins` 後，也沒有呼叫 `dynamic_path` 的 `enhance_schema` / `enhance_extraction` / `post_process_graph`。
- 結論：在目前主流程中，`dynamic_path` 屬於未接線路徑。

### 2) 四種 schema method

- 全部走 `src/rag/schema/factory.py:get_schema_by_method()`。
- `lightrag_default`：固定 entities。
- `iterative_evolution`：可回傳 entities/relations/validation schema。
- `llm_dynamic`：只回傳 entities。
- `llamaindex_dynamic`：同樣基於 `DynamicLLMPathExtractor`，但只回傳 entities。

### 3) 另一條 Dynamic 抽取路徑

- `src/graph_builder/unified/property_graph/extractor_factory.py` 的 `dynamic` extractor 也直接建立 `DynamicLLMPathExtractor`。
- 這與 `dynamic_path` plugin 在技術核心上重疊，但屬於不同接線點。

## Overlap Judgment

### 可替代（重疊）

- `dynamic_path` 的「動態類型偵測核心能力」可由 `llamaindex_dynamic`（schema 產生）與 property-graph `dynamic` extractor（抽取階段）覆蓋。
- 二者都使用 `DynamicLLMPathExtractor`，差別主要在接線位置與輸出包裝，而非底層能力。

### 不可直接替代（差異）

- `dynamic_path` 的 `post_process_graph` 會把 runtime 發現類型回填到 `graph_data["schema_info"]`。
- 目前四種 schema method 都不做這件事。
- 但由於主流程未呼叫該 plugin，此差異在現行可觀測行為中不構成回歸風險。

### 目前未生效

- `dynamic_path` plugin 在 `run_evaluation.py` LightRAG 主流程中未生效，可視為 dead path。

## Decision

- 建議將 `dynamic_path` 視為可淘汰的重疊實作。
- 保留統一路徑：
  - Schema：`get_schema_by_method(..., method="llamaindex_dynamic")`（或其他 method）
  - 抽取：`property_graph` 的 `dynamic` extractor 設定

## Removal Strategy

### Phase 1 (Low risk, now)

- 標記 `dynamic_path` 為 deprecated（文件/說明層）。
- 修正 CLI 或設定註解，避免暗示不存在或未接線功能。

### Phase 2 (Medium risk, optional)

- 移除 `dynamic_path_plugin.py` 與相關引用（若後續確認無外部腳本依賴）。
- 若仍需 `post_process_graph` 回填能力，應在統一路徑新增明確 hook，再移除舊 plugin。

## Verification Checklist

- `--lightrag_schema_method lightrag_default` 可正常跑。
- `--lightrag_schema_method llamaindex_dynamic` 可正常跑。
- `--lightrag_plugins similar_entity_merge` 可正常跑（不受 dynamic_path 影響）。
- 不使用 `dynamic_path` 時，輸出路徑與索引建置行為與既有流程一致。
