# 文檔索引

本目錄為 **End_to_End_RAG** 的說明文件入口；實作細節以程式碼與 [`scripts/run_evaluation.py`](../scripts/run_evaluation.py) 為準。

## Graph / CLI 快速對照（權威行為）

| 用途 | 建議旗標 | 說明 |
|------|-----------|------|
| 端到端 LightRAG | `--graph_rag_method lightrag` | 現行保留的端到端 Graph 主路徑 |
| 統一 PropertyGraph（多 extractor / retriever） | `--unified_graph_type property_graph` + `--pg_extractors` / `--pg_retrievers` / `--pg_combination_mode` | 取代已棄用的 `--graph_rag_method propertyindex` |
| 統一 LightRAG（模組化包裝） | `--unified_graph_type lightrag` | 經 `UnifiedGraphBuilder` + `ModularGraphWrapper` |
| Builder + Retriever 組合 | `--graph_preset` 或同時 `--graph_builder` + `--graph_retriever` | 見 [MODULAR_PIPELINE.md](MODULAR_PIPELINE.md) |

**已棄用（僅印遷移提示，不執行評估）**：`--graph_rag_method autoschema`、`dynamic_schema`、`propertyindex` → 請改用上表 `unified_graph_type property_graph` 或模組化參數。

**架構預留（核心邏輯待實作）**：`--graph_rag_method graphiti`、`neo4j`、`cq_driven`。

**PropertyGraph 統一架構（設計與 API 細節）**：[../PROPERTYGRAPH_REFACTOR_README.md](../PROPERTYGRAPH_REFACTOR_README.md)（專案根目錄，與 `src/graph_builder/unified.py` 對齊）。

---

## 使用者指南（建議優先閱讀）

| 主題 | 文檔 | 適用對象 | 狀態 |
|------|------|----------|------|
| 整體架構與資料流 | [ARCHITECTURE.md](ARCHITECTURE.md) | 開發／擴充 Pipeline | 維護中 |
| 指令與方法大全 | [COMMAND_REFERENCE.md](COMMAND_REFERENCE.md) | 跑實驗、查參數 | 維護中 |
| Builder + Retriever 模組化 | [MODULAR_PIPELINE.md](MODULAR_PIPELINE.md) | Graph 組合實驗 | 維護中 |
| 測試與互動腳本 | [TESTING_GUIDE.md](TESTING_GUIDE.md) | 本機驗證 | 維護中 |
| 設定速查 | [SETTINGS_QUICK_REFERENCE.md](SETTINGS_QUICK_REFERENCE.md) | `config.yml` / 模型 | 維護中 |
| 使用範例 | [EXAMPLES.md](EXAMPLES.md) | 複製貼上範例 | 維護中 |
| API 與程式介面 | [API.md](API.md) | 二次開發 | 維護中 |

---

## 專題文檔

| 主題 | 文檔 | 說明 |
|------|------|------|
| 評估指標 | [METRICS.md](METRICS.md) | 檢索／生成／Judge |
| RAGAS | [RAGAS_INTEGRATION.md](RAGAS_INTEGRATION.md) | 整合說明 |
| Token 預算 | [TOKEN_BUDGET_GUIDE.md](TOKEN_BUDGET_GUIDE.md) | LightRAG 與 baseline 對齊 |
| Schema 寫入結果 | [SCHEMA_RECORDING.md](SCHEMA_RECORDING.md) | 結果檔中的 schema 紀錄 |
| Schema 快取管理 | 見 [IMPLEMENTATION_HISTORY.md](IMPLEMENTATION_HISTORY.md) 與 `scripts/manage_schema_cache.py` | — |

---

## 歷史與實作紀錄

| 文檔 | 說明 |
|------|------|
| [IMPLEMENTATION_HISTORY.md](IMPLEMENTATION_HISTORY.md) | **單一權威**：LightRAG／ontology 相關改進模組清單與路徑（精簡版） |
| [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) | 已併入 IMPLEMENTATION_HISTORY；頂部導向新檔 |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | 已併入 IMPLEMENTATION_HISTORY；頂部導向新檔 |
| 專案根目錄 `*_SUMMARY.md`、`*_REPORT.md` | 各次迭代紀錄，**非**日常維護主文檔；以本索引與 ARCHITECTURE／COMMAND_REFERENCE 為準 |

---

## 其他

| 文檔 | 說明 |
|------|------|
| [RUN_COMPREHENSIVE_TESTS_UPDATE.md](RUN_COMPREHENSIVE_TESTS_UPDATE.md) | 綜合測試腳本更新紀錄 |
| [RAGAS_TEST_REPORT.md](RAGAS_TEST_REPORT.md) | RAGAS 測試報告 |
