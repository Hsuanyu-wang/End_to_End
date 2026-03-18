# run_comprehensive_tests.py 更新說明

## 🎉 更新內容

`run_comprehensive_tests.py` 已更新以整合新實作的功能，包括 Schema Cache、可插拔 Pipeline 和 Ablation Study。

## ✨ 新增功能

### 1. Schema Cache 支援
- 自動檢測測試是否支援 Schema Cache
- 提供啟用/停用快取選項
- 支援強制重建 Schema
- 顯著加速重複實驗

### 2. 新增測試類別

#### 進階檢索方法（選項 7）
- **Re-ranking**: Cross-Encoder 重排序
- **Adaptive Routing**: 智能查詢路由
- **ToG Retrieval**: Think-on-Graph 迭代檢索
- **Full Pipeline**: 啟用所有檢索增強元件

#### Ablation Study（選項 8）
- **Disambiguation Only**: 僅實體消歧
- **Re-ranking Only**: 僅 Re-ranking
- **Disambiguation + Re-ranking**: 組合測試
- **Auto Ablation**: 自動測試所有元件組合（2^n 種配置）

### 3. Schema Cache 管理工具（選項 11）
直接從測試腳本管理 Schema Cache：
- 列出所有快取
- 查看快取統計
- 清理快取（選擇方法或全部）
- 匯出快取報告

### 4. 更新的 Schema 方法
在「LightRAG Schema 方法測試」中新增：
- **AutoSchemaKG with Cache**: 支援快取的 AutoSchemaKG

## 📋 使用範例

### 基本使用
```bash
python scripts/run_comprehensive_tests.py
```

### 測試流程範例

#### 1. 測試 Schema 方法（使用快取）
```
選擇: 4 (LightRAG Schema 方法測試)
→ 選擇方法: 2 (Iterative Evolution Schema)
→ 測試模式: 1 (快速測試)
→ 資料類型: 1 (DI)
→ 使用 Cache: 1 (是)
→ 強制重建: 1 (否)
```

**第一次執行**: 生成並快取 Schema  
**第二次執行**: 直接載入快取（快！）

#### 2. 測試進階檢索方法
```
選擇: 7 (進階檢索方法測試)
→ 選擇方法: 4 (Full Pipeline - 所有元件)
→ 測試模式: 1 (快速測試)
→ 資料類型: 1 (DI)
```

執行包含所有增強元件的完整 Pipeline：
- Base Retriever
- Entity Disambiguation
- Re-ranking
- ToG (可選)

#### 3. 執行 Ablation Study
```
選擇: 8 (Ablation Study)
→ 選擇方法: 4 (Auto Ablation)
→ 測試模式: 1 (快速測試)
→ 資料類型: 1 (DI)
```

自動測試所有元件組合：
- Config 1: base_retriever only
- Config 2: base_retriever + disambiguation
- Config 3: base_retriever + reranking
- Config 4: base_retriever + disambiguation + reranking
- ... (共 2^n 種配置)

#### 4. 管理 Schema Cache
```
選擇: 11 (Schema Cache 管理工具)
→ 選擇: 1 (列出所有快取)
```

查看已快取的 Schema：
```
📦 iterative_evolution: 2 個快取
  - iterative_evolution_DI_natural_text_a3f2b1c4_9d8e7f6a.json (15.2 KB)
  - iterative_evolution_GEN_natural_text_b4f3c2d5_8e9f0g1b.json (18.5 KB)

📦 llm_dynamic: 1 個快取
  - llm_dynamic_DI_natural_text_c5f4d3e6_7f8g9h0c.json (8.3 KB)
```

## 🔧 新增參數

測試腳本現在支援以下新參數（傳遞給 `run_evaluation.py`）：

- `--use_schema_cache`: 啟用 Schema Cache
- `--force_rebuild_schema`: 強制重建 Schema（忽略快取）
- `--use_reranking`: 啟用 Re-ranking
- `--adaptive_routing`: 啟用 Adaptive Routing
- `--use_tog`: 啟用 ToG 檢索
- `--enable_all_components`: 啟用所有 Pipeline 元件
- `--enable_disambiguation`: 僅啟用實體消歧
- `--enable_reranking`: 僅啟用 Re-ranking
- `--ablation_study`: 自動 Ablation Study

## 📊 主選單更新

```
請選擇測試類型：
  1. Vector RAG 測試 (3 種方法)
  2. Advanced Vector RAG 測試 (2 種方法)
  3. Graph RAG 測試 (3 種方法)
  4. LightRAG Schema 方法測試 (4 種 schema) ✨ 支援 Cache
  5. LightRAG 檢索模式測試 (6 種模式)
  6. 模組化組合測試 (4 種組合)
  7. 進階檢索方法測試 (4 種方法) ✨ 新功能
  8. Ablation Study (4 種配置) ✨ 新功能
  9. 完整實驗 (所有方法)
 10. 自訂選擇
 11. Schema Cache 管理工具 ✨ 新功能
  0. 退出
```

## 🎯 測試配置統計

### 總測試數量
- Vector RAG: 3 種
- Advanced Vector RAG: 2 種
- Graph RAG: 3 種
- LightRAG Schema: 4 種（新增 1 種）
- LightRAG 模式: 6 種
- 模組化組合: 4 種
- **進階檢索: 4 種（新）**
- **Ablation Study: 4 種（新）**

**總計**: 30 種測試配置（增加 8 種）

## 💡 使用建議

### 1. 第一次執行（建立 Schema Cache）
```bash
# 測試 Iterative Evolution（會建立快取）
選擇: 4 → 2 → 快速測試 → DI → 使用快取: 是 → 強制重建: 否
```

### 2. 重複實驗（使用快取）
```bash
# 使用相同的 Schema，測試不同的檢索方法
選擇: 7 → 1 (Re-ranking) → 快速測試 → DI
```
快取會自動加速 Schema 載入！

### 3. Ablation Study
```bash
# 自動測試所有元件組合
選擇: 8 → 4 (Auto Ablation) → 快速測試 → DI
```

### 4. 清理快取
```bash
# 定期清理舊快取
選擇: 11 → 3 (清理快取) → 輸入方法名稱
```

## 🔍 注意事項

1. **Schema Cache 路徑**: `/home/End_to_End_RAG/storage/schema_cache/`
2. **快取檔名格式**: `{method}_{data_type}_{data_mode}_{corpus_hash}_{config_hash}.json`
3. **自動檢測**: 腳本會自動檢測測試是否支援快取
4. **Ablation Study**: Auto Ablation 會生成 2^n 種配置，測試時間較長

## 📝 更新日誌

**v2.0 - 2026-03-17**
- ✨ 新增 Schema Cache 支援
- ✨ 新增進階檢索方法測試類別
- ✨ 新增 Ablation Study 測試類別
- ✨ 新增 Schema Cache 管理工具
- 🔧 更新 LightRAG Schema 測試（新增 AutoSchemaKG with Cache）
- 📝 改進選單和說明文字
- 🎨 標記新功能（✨）

---

**實作完成**: 2026-03-17  
**對應模組**: Schema Cache, Retrieval Pipeline, Re-ranking, Router  
**相容性**: 向下相容所有原有功能
