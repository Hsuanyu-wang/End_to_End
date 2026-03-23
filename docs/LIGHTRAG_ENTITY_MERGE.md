# LightRAG 實體合併行為說明

## 建圖階段（上游 LightRAG）

上游套件在 `insert` 建圖時，實體節點以 **正規化／截斷後的實體名稱字串** 作為圖節點 ID；`_merge_nodes_then_upsert` 僅對 **相同名稱** 的節點合併描述與來源欄位（見 `lightrag/operate.py` 中 `maybe_nodes[truncated_name]` 與 `get_node(entity_name)`）。  
**不同字串名稱** 的實體不會因 embedding 相似度而在建圖時自動併成同一節點。

## 本專案：相似實體合併 plugin（可選）

若要在建圖後依 **embedding 餘弦相似度** 合併近似實體，請使用：

- CLI：`--lightrag_plugins similar_entity_merge`（可與其他 plugin 逗號分隔）
- 設定：`config.yml` → `lightrag.plugins.similar_entity_merge`（閾值、`text_mode` 等預設值）

合併會寫入 **獨立 storage**（`custom_tag` 含閾值與 `text_mode`），並在 baseline 已有索引時 **先複製整個 working_dir** 再於副本上呼叫官方 `LightRAG.merge_entities`，不修改 baseline 目錄。

詳見實作：`src/rag/plugins/lightrag_similar_entity_merge.py`。
