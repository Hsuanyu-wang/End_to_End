# QA 自動生成腳本使用說明

## 概述

`generate_qa.py` 是用於從 `CSR_full.jsonl` 自動生成高品質問答對的腳本，支援兩種類型的 QA：

- **Local QA**: 針對單一維護紀錄的細節問題（故障排除、具體步驟）
- **Global QA**: 跨紀錄的趨勢分析與模式識別問題

## 主要特性

### 1. 智能 Cache 管理
- 自動檢查 LightRAG 索引是否已存在
- 若 cache 存在則直接使用，避免重複建圖
- 支援 `--force_rebuild` 強制重建索引

### 2. 資料處理
- 使用專案統一的 `data_processing()` 函數
- 預設使用 `"natural_text"` 模式（敘述化文本）
- 自動從 `config.yml` 讀取 `raw_file_path_GEN` 指向 `CSR_full.jsonl`
- **不使用 CIX、PID 欄位**，僅使用：NO, Customer, Service Start, Service End, Engineers, Description, Action

### 3. LightRAG 整合
- 使用專案的 `get_lightrag_engine()` 和 `build_lightrag_index()`
- Storage 路徑透過 `get_storage_path()` 自動管理
- 支援 `local` 和 `global` 兩種檢索模式

### 4. QA 生成策略

#### Local QA
- 隨機抽取維護紀錄作為種子
- 使用 LightRAG local 模式查詢細節
- 透過 LLM 生成具體的技術問題
- 包含處置步驟、技術細節、問題排除方法

#### Global QA
- 基於客戶、工程師、時間範圍等維度分組
- 使用 LightRAG global 模式進行趨勢分析
- 生成跨紀錄的整合性問題
- 涵蓋模式識別、優化建議、共通問題分析

### 5. 輸出格式

符合 `qa_global_group_100.jsonl` 標準格式：

```json
{
  "id": "qa_local_00000001",
  "question": "問題內容",
  "answer": "答案內容",
  "qa_type": "local",
  "source_doc_ids": ["37748", "37745"],
  "confidence": 0.85
}
```

## 使用方法

### 基本用法

```bash
# 生成各 50 組 Local 和 Global QA（預設）
python scripts/generate_qa.py

# 指定數量
python scripts/generate_qa.py --local_count 100 --global_count 100

# 僅生成 Local QA
python scripts/generate_qa.py --local_count 50 --global_count 0

# 僅生成 Global QA
python scripts/generate_qa.py --local_count 0 --global_count 50
```

### 進階選項

```bash
# 強制重建 LightRAG 索引（當資料更新時使用）
python scripts/generate_qa.py --local_count 50 --global_count 50 --force_rebuild

# 指定輸出目錄
python scripts/generate_qa.py --local_count 50 --global_count 50 --output_dir ./output

# 組合使用
python scripts/generate_qa.py \
  --local_count 100 \
  --global_count 100 \
  --output_dir ./Data/qa_generated \
  --force_rebuild
```

### 查看幫助

```bash
python scripts/generate_qa.py --help
```

## 命令列參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--local_count` | int | 50 | 要生成的 Local QA 數量 |
| `--global_count` | int | 50 | 要生成的 Global QA 數量 |
| `--output_dir` | str | ./Data | 輸出目錄路徑 |
| `--force_rebuild` | flag | False | 強制重建 LightRAG 索引 |

## 輸出檔案

執行後會在指定的輸出目錄產生以下檔案：

- `qa_local_{count}.jsonl`: Local QA 問答對
- `qa_global_{count}.jsonl`: Global QA 問答對

檔案名稱中的 `{count}` 會自動替換為實際生成的數量。

## 執行流程

腳本執行分為 5 個步驟：

1. **載入專案設定**: 從 `config.yml` 讀取模型配置
2. **設定 LightRAG 引擎**: 檢查 cache 並決定是否重建索引
3. **載入 CSR 紀錄**: 從 `CSR_full.jsonl` 讀取資料
4. **生成 QA 對**: 分別生成 Local 和 Global QA
5. **儲存結果**: 以 JSONL 格式輸出

## LightRAG Cache 位置

索引會儲存在以下位置：

```
/home/End_to_End_RAG/storage/lightrag/GEN_qa_generation/
```

關鍵檔案：
- `kv_store_full_docs.json`: 文檔儲存
- `graph_chunk_entity_relation.graphml`: 知識圖譜

## 效能考量

### 首次執行
- 需要建立 LightRAG 索引（約 10-30 分鐘，取決於資料量）
- 包含 LLM 實體與關係抽取

### 後續執行
- 自動使用已建立的索引（數秒內完成載入）
- 僅執行 QA 生成步驟

### 建議
- 資料更新時使用 `--force_rebuild` 重建索引
- 大量生成時可分批執行避免超時

## 錯誤處理

腳本包含完整的錯誤處理機制：

- **檔案不存在**: 檢查 `CSR_full.jsonl` 路徑
- **LLM 連線失敗**: 確認 Ollama 服務運行中
- **生成失敗**: 自動跳過並繼續下一組
- **使用者中斷**: Ctrl+C 可安全中斷執行

## 技術架構

### 依賴模組

```python
from src.config.settings import get_settings
from src.data.processors import data_processing
from src.rag.graph.lightrag import get_lightrag_engine, build_lightrag_index
from src.storage import get_storage_path
from lightrag import QueryParam
```

### 關鍵函數

- `check_lightrag_cache()`: 檢查 cache 完整性
- `setup_or_load_lightrag()`: 智能載入或建構索引
- `load_csr_records()`: 載入原始紀錄
- `generate_local_qa()`: 生成 Local QA
- `generate_global_qa()`: 生成 Global QA
- `save_qa_jsonl()`: 儲存為 JSONL 格式

## 與專案整合

### 配置檔案
使用 `config.yml` 中的設定：
- `data.raw_file_path_GEN`: 資料來源路徑
- `model`: LLM 模型配置
- `lightrag`: LightRAG 配置

### Storage 管理
使用專案統一的 storage 管理機制：
```python
storage_path = get_storage_path(
    storage_type="lightrag",
    data_type="GEN",
    method="qa_generation"
)
```

## 範例輸出

### 執行畫面

```
================================================================================
🚀 自動 QA 生成腳本
================================================================================
📊 目標數量: Local QA = 50, Global QA = 50
📂 輸出目錄: ./Data
🔧 強制重建: 否
================================================================================

[步驟 1/5] 載入專案設定...
✅ 設定載入完成

[步驟 2/5] 設定 LightRAG 引擎...
✅ 使用現有 LightRAG 索引
📂 索引路徑: /home/End_to_End_RAG/storage/lightrag/GEN_qa_generation
✅ LightRAG 引擎就緒

[步驟 3/5] 載入 CSR 紀錄...
📊 載入 2321 筆 CSR 紀錄
✅ CSR 紀錄載入完成

[步驟 4/5] 生成 QA 對...

🔍 開始生成 50 組 Local QA...
  已生成 10/50 組 Local QA
  已生成 20/50 組 Local QA
  ...
✅ Local QA 生成完成，共 50 組

🌐 開始生成 50 組 Global QA...
  已生成 10/50 組 Global QA
  已生成 20/50 組 Global QA
  ...
✅ Global QA 生成完成，共 50 組

✅ QA 生成完成，共 100 組

[步驟 5/5] 儲存結果...
💾 已儲存至: ./Data/qa_local_50.jsonl
💾 已儲存至: ./Data/qa_global_50.jsonl

================================================================================
✅ 所有任務完成！
================================================================================
📊 生成摘要:
  - Local QA:  50 組
  - Global QA: 50 組
  - 總計:      100 組
================================================================================
```

## 注意事項

1. **首次執行時間較長**: 需要建立知識圖譜
2. **確保 Ollama 運行**: 檢查配置的模型是否可用
3. **儲存空間**: LightRAG 索引約需 100-500MB
4. **記憶體需求**: 建議至少 8GB RAM

## 常見問題

### Q: 如何更新資料後重新生成？
A: 使用 `--force_rebuild` 參數重建索引

### Q: 生成速度太慢怎麼辦？
A: 確認 cache 已建立，後續執行會快很多

### Q: 可以只生成其中一種類型的 QA 嗎？
A: 可以，將另一種類型的 count 設為 0

### Q: 輸出格式可以自訂嗎？
A: 目前採用標準 JSONL 格式，可修改 `save_qa_jsonl()` 函數

## 更新歷史

### 2026-03-19
- ✅ 完整重寫腳本
- ✅ 整合專案 LightRAG 架構
- ✅ 使用統一的 data processing 模組
- ✅ 實作智能 cache 管理
- ✅ 移除 CIX、PID 欄位依賴
- ✅ 支援命令列參數控制
- ✅ 完整錯誤處理機制
