# LightRAG／Ontology 改進模組紀錄（精簡版）

**用途**：單一權威索引，對應 2026-03-17 前後導入的模組與路徑。詳細長文與範例程式已併入本檔精簡表；完整舊版全文見 [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)（僅追溯）。

**日常文檔**：請優先 [README.md](README.md)（索引）、[ARCHITECTURE.md](ARCHITECTURE.md)、[COMMAND_REFERENCE.md](COMMAND_REFERENCE.md)。

## 模組清單

| 分類 | 項目 | 路徑 |
|------|------|------|
| Schema 快取 | 統一 cache | `src/rag/schema/schema_cache.py`、`scripts/manage_schema_cache.py` |
| 檢索管線 | 可插拔 Pipeline | `src/rag/retrieval/retrieval_pipeline.py` |
| 重排序 | Cross-Encoder | `src/evaluation/reranker.py` |
| Schema | 實體消歧 | `src/rag/schema/entity_disambiguation.py` |
| Schema | 收斂判斷 | `src/rag/schema/convergence.py` |
| Graph | DynamicSchema 建圖優化 | `src/graph_builder/dynamic_schema_builder.py` |
| Graph | ToG 風格檢索 | `src/graph_retriever/tog_retriever.py` |
| Graph | 查詢路由 | `src/graph_retriever/adaptive_router.py` |
| 設定 | 全域 YAML | `config.yml` |

## 維護備註

- 端到端評估入口以 `scripts/run_evaluation.py` 為準；Graph 旗標遷移見 [README.md](README.md) 開頭對照表。
- 外掛套件 `src/plugins/` 部分為 stub／TODO，與「已完成模組」表分開看待。
