# RAG 測試與實驗腳本使用指南

本文件說明如何使用新的測試與實驗腳本來執行 RAG 評估。

## 可用腳本

### 1. 互動式測試腳本 (推薦)

**檔案**: `scripts/run_comprehensive_tests.py`

**功能**: 提供互動式選單，讓使用者靈活選擇要執行的測試

**使用方式**:
```bash
cd /home/End_to_End_RAG
python scripts/run_comprehensive_tests.py
```

**特點**:
- 互動式選單介面
- 支援單一或批次測試
- 可選擇快速測試（2 題）或完整測試（全部題目）
- 可選擇資料類型（DI 或 GEN）
- 自動產生測試總結

**選單選項**:
1. Vector RAG 測試 (hybrid, vector, bm25)
2. Advanced Vector RAG 測試 (parent_child, self_query)
3. Graph RAG 測試 (propertyindex, dynamic_schema)
4. LightRAG Schema 方法測試 (lightrag_default, iterative_evolution, llm_dynamic)
5. LightRAG 檢索模式測試 (local, global, hybrid, mix, naive, bypass)
6. 完整實驗 (所有方法)
7. 自訂選擇

### 2. 批次實驗腳本

**檔案**: `scripts/run_all_experiments.sh`

**功能**: 自動執行所有 RAG 方法的完整評估（不使用 fast_test）

**使用方式**:
```bash
cd /home/End_to_End_RAG
bash scripts/run_all_experiments.sh
```

**特點**:
- 完全自動化
- 使用完整資料集評估
- 自動產生日誌檔案
- 自動產生 HTML 報告
- 錯誤處理與統計

**預設測試項目**:
- Vector RAG (hybrid, vector, bm25)
- Advanced Vector RAG (parent_child, self_query)
- Graph RAG (propertyindex, dynamic_schema)
- LightRAG Schema 方法 (lightrag_default, iterative_evolution, llm_dynamic)
- (可選) LightRAG 檢索模式 (local, global, mix, naive, bypass)

### 3. Schema 快速測試腳本

**檔案**: `run_all_schema_tests.sh`

**功能**: 快速測試所有 LightRAG schema 方法（使用 fast_test）

**使用方式**:
```bash
cd /home/End_to_End_RAG
bash run_all_schema_tests.sh
```

**特點**:
- 只測試 LightRAG schema 方法
- 使用快速測試模式（2 題）
- 適合驗證 schema 生成是否正常

## 測試配置總覽

### Vector RAG (3 種)
- `hybrid`: Vector + BM25 混合檢索
- `vector`: 純向量檢索
- `bm25`: 純 BM25 關鍵字檢索

### Advanced Vector RAG (2 種)
- `parent_child`: 父子文件檢索
- `self_query`: 自查詢（帶 metadata 過濾）

### Graph RAG (2 種)
- `propertyindex`: Property Graph 索引
- `dynamic_schema`: 動態 Schema 圖譜

### LightRAG Schema 方法 (3 種)
- `lightrag_default`: 預設實體類型
- `iterative_evolution`: 演化式 Schema 生成
- `llm_dynamic`: LLM 動態生成 Schema

### LightRAG 檢索模式 (6 種)
- `local`: 關注特定實體與細節
- `global`: 關注整體趨勢與總結
- `hybrid`: 混合 local 與 global
- `mix`: 知識圖譜 + 向量檢索
- `naive`: 僅向量檢索
- `bypass`: 直接查詢 LLM

## 使用建議

### 快速驗證

如果只想快速驗證系統是否正常運作：

```bash
# 使用互動式腳本，選擇快速測試模式
python scripts/run_comprehensive_tests.py
# 然後選擇任一測試類型 + 快速測試模式
```

### Schema 方法測試

如果要測試所有 schema 方法：

```bash
# 快速測試
bash run_all_schema_tests.sh

# 或使用互動式腳本
python scripts/run_comprehensive_tests.py
# 選擇選項 4 (LightRAG Schema 方法測試)
```

### 完整實驗

如果要執行完整的方法比較實驗：

```bash
# 批次執行所有測試（需數小時）
bash scripts/run_all_experiments.sh

# 或使用互動式腳本選擇需要的測試
python scripts/run_comprehensive_tests.py
```

## 結果檔案

### 評估結果

所有評估結果儲存在 `/home/End_to_End_RAG/results/` 目錄下：

```
results/
├── evaluation_results_20260317_HHMMSS_<postfix>/
│   ├── global_summary_report.csv          # 所有方法的平均指標比較
│   ├── <Pipeline_Name>/
│   │   └── detailed_results.csv           # 逐題評估結果（包含 schema 資訊）
```

### 實驗日誌

批次實驗腳本會產生日誌：

```
experiment_logs/
├── experiment_20260317_HHMMSS.log         # 文字日誌
└── experiment_20260317_HHMMSS_report.html # HTML 報告
```

## 注意事項

### 執行時間

- **快速測試** (2 題): 每個方法約 30-60 秒
- **完整測試** (43 題): 每個方法約 10-30 分鐘
- **所有方法完整測試**: 可能需要 3-6 小時

### 系統需求

1. **Ollama 服務**: 確保 Ollama 在 `http://192.168.63.174:11434` 正常運行
2. **模型可用性**: 
   - qwen2.5:7b (主要 LLM)
   - qwen2.5:14b (評估 & 建圖 LLM)
   - nomic-embed-text (Embedding)
3. **磁碟空間**: 每個方法會產生索引檔案，建議至少 10GB 可用空間
4. **記憶體**: 建議至少 16GB RAM

### 索引管理

- 索引會儲存在 `/home/End_to_End_RAG/storage/` 目錄
- 快速測試會使用 `_fast_test` 後綴，與完整測試隔離
- 如需清理索引：
  ```bash
  rm -rf /home/End_to_End_RAG/storage/lightrag/*
  rm -rf /home/End_to_End_RAG/storage/vector_index/*
  rm -rf /home/End_to_End_RAG/storage/graph_index/*
  ```

## 常見問題

### Q: llm_dynamic 出現 BadRequestError

**A**: 已在 2026-03-17 修復。如果仍出現此錯誤，請確認：
1. `src/rag/schema/factory.py` 中的修復已套用
2. Ollama 服務正常運行
3. 模型 qwen2.5:14b 已下載

### Q: 如何只測試特定組合？

**A**: 使用互動式腳本的「自訂選擇」選項（選項 7），可以精確選擇要測試的方法。

### Q: 如何修改資料類型？

**A**: 
- 互動式腳本：執行時會詢問資料類型
- 批次腳本：修改 `scripts/run_all_experiments.sh` 中的 `DATA_TYPE` 變數

### Q: 測試失敗怎麼辦？

**A**: 
1. 檢查 Ollama 服務是否正常
2. 查看錯誤訊息
3. 檢查日誌檔案（批次腳本）
4. 單一測試失敗不會中斷其他測試

## 範例使用流程

### 情境 1: 第一次使用，驗證系統

```bash
# 使用互動式腳本
python scripts/run_comprehensive_tests.py

# 選擇：
# 1. Vector RAG 測試
# 2. 選擇 1 (Hybrid)
# 3. 選擇 1 (快速測試)
# 4. 選擇 1 (DI 資料)
```

### 情境 2: 比較所有 Schema 方法

```bash
# 使用專用腳本
bash run_all_schema_tests.sh

# 或使用互動式腳本
python scripts/run_comprehensive_tests.py
# 選擇 4 (LightRAG Schema 方法測試)
# 選擇全部執行
```

### 情境 3: 完整論文實驗

```bash
# 執行完整批次實驗
bash scripts/run_all_experiments.sh

# 等待數小時後，查看：
# - results/ 目錄中的評估結果
# - experiment_logs/ 中的 HTML 報告
```

## 支援與回報

如有問題或建議，請參考：
- [主要 README](../README.md)
- [Schema 記錄功能說明](SCHEMA_RECORDING.md)
- [API 文檔](API.md)
