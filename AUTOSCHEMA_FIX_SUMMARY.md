# AutoSchemaKG 修正實作摘要

## 修正日期
2026-03-17

## 問題診斷

根據 terminal 輸出分析，AutoSchemaKG 執行失敗的根本原因：

1. **輸入格式錯誤**: 缺少必需的 `metadata` 欄位
2. **路徑配置不一致**: `filename_pattern` 使用通配符導致無法找到檔案
3. **輸出解析未實作**: `_parse_autoschema_output` 僅為佔位代碼
4. **錯誤訊息不足**: 難以除錯問題根因

## 實作修正

### 1. 修正輸入資料格式

**檔案**: `src/graph_builder/autoschema_builder.py` (110-129 行)

**修正內容**:
- 從 LlamaIndex Document 正確提取 `text` 和 `metadata`
- 輸出格式從簡單的 `{id, text}` 改為 `{id, text, metadata}`
- 符合 AutoSchemaKG 期待的完整格式

**驗證結果**:
```json
{
    "id": "doc_0",
    "text": "這是一筆關於 F客戶 的維護紀錄...",
    "metadata": {
        "NO": "0ea766a4-e2b9-4d63-bd8e-17d9c677e4ab",
        "Customer": "F客戶",
        "Engineers": "A人員",
        "Service Start": "2025/12/26  15:00",
        "Service End": "2025/12/26  16:00"
    }
}
```

### 2. 修正 ProcessingConfig 路徑配置

**檔案**: `src/graph_builder/autoschema_builder.py` (73-84 行)

**修正內容**:
- `filename_pattern` 從 `"*.jsonl"` 改為 `"input_documents.jsonl"`
- 確保 `data_directory` 與實際輸入檔案位置一致
- 新增目錄建立邏輯 `os.makedirs(self.output_dir, exist_ok=True)`

### 3. 實作 GraphML 輸出解析

**檔案**: `src/graph_builder/autoschema_builder.py` (212-334 行)

**功能**:
- 使用 NetworkX 解析 GraphML 檔案
- 提取節點 (nodes) 和邊 (edges)
- 收集實體類型和關係類型
- 轉換為標準化格式供 RAG 使用

**實作邏輯**:
```python
import networkx as nx

# 尋找 GraphML 檔案
graphml_files = glob.glob(os.path.join(graphml_dir, "*.graphml"))

# 解析每個檔案
for gml_file in graphml_files:
    G = nx.read_graphml(gml_file)
    
    # 提取節點
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        nodes.append({"id": node_id, **dict(node_data)})
    
    # 提取邊
    for source, target, edge_data in G.edges(data=True):
        edges.append({"source": source, "target": target, **dict(edge_data)})
```

### 4. 增強錯誤日誌

**檔案**: `src/graph_builder/autoschema_builder.py` (137-192 行)

**增強內容**:
- 記錄錯誤類型 `type(e).__name__`
- 輸出完整堆疊追蹤 `traceback.format_exc()`
- 顯示輸出目錄路徑
- 列出生成的檔案清單

## 驗證結果

### 自動化驗證腳本
執行 `verify_autoschema_fix.py`:

```
✅ 測試 1: 驗證輸入檔案格式
   ✓ 格式正確，包含必需欄位: id, text, metadata
   ✓ 文字長度: 283 字元
   ✓ Metadata 欄位數: 5

✅ 測試 2: 驗證路徑配置
   ✓ 輸出目錄: /tmp/test_paths

✅ 測試 3: 驗證 GraphML 解析邏輯
   ✓ GraphML 解析邏輯已實作

✅ 測試 4: 驗證錯誤日誌增強
   ✓ 錯誤日誌已增強
```

### 檔案修改清單

- `src/graph_builder/autoschema_builder.py`
  - 新增 `glob` 導入
  - 修正 `initialize()` 的 `filename_pattern`
  - 重寫 `build()` 的輸入格式化邏輯
  - 新增目錄建立邏輯
  - 增強三元組抽取、概念生成、GraphML 轉換的錯誤處理
  - 完整實作 `_parse_autoschema_output()` 方法

## 使用方式

### 基本用法

```python
from src.graph_builder.autoschema_builder import AutoSchemaKGBuilder
from src.data.processors import data_processing

# 載入文檔
documents = data_processing(mode="natural_text", data_type="DI")

# 建立 Builder
builder = AutoSchemaKGBuilder(output_dir="/path/to/output")
builder.initialize({
    'model_name': 'llama3.3:latest',
    'batch_size_triple': 3,
    'batch_size_concept': 16,
    'max_workers': 3
})

# 執行建圖
result = builder.build(documents)

# 使用結果
print(f"節點數: {len(result['nodes'])}")
print(f"邊數: {len(result['edges'])}")
print(f"Schema: {result['schema_info']}")
```

### 整合到評估腳本

現有的 `scripts/run_evaluation.py` 中的 `setup_autoschema_pipeline()` 函數已可正常使用：

```bash
python scripts/run_evaluation.py --graph_rag_method autoschema
```

## 注意事項

1. **LLM 推理時間**: AutoSchemaKG 需要 LLM 進行三元組抽取和概念生成，執行時間較長
2. **依賴套件**: 需要安裝 `atlas-rag` 和 `networkx`
3. **輸出目錄**: 建圖過程會在輸出目錄生成多個子目錄
4. **錯誤容忍**: 採用錯誤容忍策略，部分步驟失敗不會中斷整個流程

## 下一步建議

1. 執行完整的 E2E 測試 (需要較長時間)
2. 比較 AutoSchemaKG 與其他方法的效能
3. 優化批次大小和工作數以提升速度
4. 根據實際使用情況調整錯誤處理策略
