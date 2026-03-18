# Token Budget統一評估使用說明

## 概述

本功能實現了統一的token budget標準，確保不同RAG方法在相同的context token預算下進行公平比較。

## 主要功能

### 1. 自動Token計數

所有RAG wrapper現在會自動計算並記錄context tokens：

- **Vector RAG**: 記錄retrieved chunks的總token數
- **LightRAG**: 記錄entities、relations、chunks各組件的token數，並提供詳細breakdown

### 2. 兩階段評估流程

啟用`--enable_token_budget`參數後，評估流程分為兩個階段：

**階段1: Baseline評估**
- 執行Vector RAG (或指定的baseline方法)
- 計算平均context token使用量
- 設定為token budget基準

**階段2: 其他方法評估**
- 根據baseline動態建議LightRAG參數
- 評估所有其他方法
- 記錄token使用情況並生成分析報告

### 3. Token分析報告

自動生成以下報告：
- `token_budget_report.txt`: Token使用比較總結
- `token_budget_stats.json`: 詳細token統計數據
- `token_comparison.csv`: 各方法token使用比較表
- `token_analysis_report.txt`: 完整分析報告

## 使用方法

### 基本用法

```bash
# 標準評估（不啟用token budget）
python scripts/run_evaluation.py \
    --vector_method hybrid \
    --graph_rag_method lightrag \
    --lightrag_mode hybrid \
    --data_type DI \
    --top_k 2

# 啟用token budget評估
python scripts/run_evaluation.py \
    --vector_method hybrid \
    --graph_rag_method lightrag \
    --lightrag_mode hybrid \
    --data_type DI \
    --top_k 2 \
    --enable_token_budget
```

### 自訂Baseline方法

```bash
# 指定特定方法作為baseline
python scripts/run_evaluation.py \
    --vector_method hybrid \
    --graph_rag_method lightrag \
    --lightrag_mode hybrid \
    --data_type DI \
    --enable_token_budget \
    --token_budget_baseline "hybrid"
```

## 評估結果解讀

### CSV欄位說明

評估結果CSV新增以下欄位：

| 欄位名稱 | 說明 | 適用方法 |
|---------|------|---------|
| `context_tokens` | 總context token數 | 所有方法 |
| `entity_tokens` | Entity context tokens | LightRAG |
| `relation_tokens` | Relation context tokens | LightRAG |
| `chunk_tokens` | Chunk/text context tokens | LightRAG |

### Token Budget報告範例

```
================================================================================
Token Budget分析報告
================================================================================

Baseline Tokens: 1524
Target Tokens (含10%緩衝): 1676

--------------------------------------------------------------------------------
方法                            平均Tokens                   範圍        標準差      利用率
--------------------------------------------------------------------------------
Vector_Hybrid                   1524.0         [1445, 1612]       53.9   100.0%
LightRAG_Hybrid                 1517.4         [1476, 1587]       38.5    99.6%
LightRAG_Local                  1612.3         [1523, 1689]       47.2   105.8%
================================================================================
```

### 利用率解讀

- **≤ 100%**: 在baseline範圍內，公平比較
- **100-110%**: 在緩衝範圍內，可接受
- **> 110%**: 超出預算，需調整參數

## LightRAG參數調整

系統會根據baseline自動建議LightRAG參數：

### Hybrid模式建議
```python
max_total_tokens: 1676      # baseline * 1.1
max_entity_tokens: 419      # 25%
max_relation_tokens: 586    # 35%
chunk_top_k: 3              # 剩餘空間 / 200
```

### Local/Global模式建議
```python
max_total_tokens: 1676
max_entity_tokens: 586      # 35%
max_relation_tokens: 754    # 45%
chunk_top_k: 1
```

### 如何應用建議參數

目前需要手動在建立LightRAG實例時設定：

```python
from lightrag import LightRAG

rag = LightRAG(
    working_dir="./storage",
    max_total_tokens=1676,
    max_entity_tokens=419,
    max_relation_tokens=586,
    chunk_top_k=3
)
```

## 進階功能

### 使用Token分析工具

```python
from src.evaluation.token_analysis import analyze_evaluation_results

# 分析評估結果目錄
analyze_evaluation_results(
    eval_results_dir="results/evaluation_results_20260317_120000",
    baseline_method="Vector_Hybrid"
)
```

### 使用TokenBudgetController

```python
from src.rag.token_budget_controller import TokenBudgetController

controller = TokenBudgetController()

# 設定baseline
vector_tokens = [1523, 1487, 1612, 1445, 1589]
controller.set_baseline("Vector_Hybrid", vector_tokens)

# 取得調整參數
params = controller.adjust_lightrag_params(mode="hybrid")
print(params)

# 添加其他方法統計
lightrag_tokens = [1512, 1489, 1587, 1476]
controller.add_method_stats("LightRAG_Hybrid", lightrag_tokens)

# 生成報告
report = controller.generate_report()
print(report)

# 儲存統計
controller.save_stats("token_stats.json")
```

## 測試與驗證

執行測試腳本驗證功能：

```bash
python scripts/test_token_budget.py
```

## 常見問題

### Q1: 為什麼LightRAG的retrieval指標是0？

**A**: 確認是否使用Context模式（`use_context=True`）。原生模式無法獲取retrieved_ids。

### Q2: ChunkIDMapper映射失敗怎麼辦？

**A**: 檢查`chunk_id_mapping.json`是否存在於LightRAG storage目錄中。如需重建映射，刪除該文件後重新建立索引。

### Q3: 如何驗證token計數準確性？

**A**: TokenCounter使用tiktoken，與OpenAI API一致。可透過以下方式驗證：

```python
from src.utils.token_counter import count_tokens

text = "測試文本"
tokens = count_tokens(text)
print(f"Token數: {tokens}")
```

### Q4: 建議的LightRAG參數沒有自動應用？

**A**: 目前實現僅提供參數建議，需手動設定。未來版本將支援自動參數調整。

## 實現檔案清單

### 新建檔案
- `src/utils/token_counter.py` - Token計數工具
- `src/rag/token_budget_controller.py` - Token budget控制器
- `src/evaluation/token_analysis.py` - Token分析工具
- `scripts/test_token_budget.py` - 測試腳本

### 修改檔案
- `src/rag/wrappers/base_wrapper.py` - 新增token追蹤
- `src/rag/wrappers/lightrag_wrapper.py` - LightRAG詳細token統計
- `src/evaluation/evaluator.py` - 新增token欄位
- `src/evaluation/reporters.py` - 兩階段評估流程
- `src/rag/graph/lightrag_id_mapper.py` - 強化驗證
- `scripts/run_evaluation.py` - 新增token budget參數

## 未來改進方向

1. **自動參數調整**: 直接修改LightRAG實例參數，無需手動設定
2. **多baseline比較**: 支援多個baseline method的比較分析
3. **視覺化報告**: 生成token使用分布圖表
4. **實時監控**: 評估過程中實時顯示token使用情況
5. **成本估算**: 根據token使用量估算API成本

## 聯絡與支援

如有問題或建議，請參考：
- 主要文檔: `docs/MODULAR_PIPELINE.md`
- 專案狀態: `PROJECT_STATUS.md`
- 快速開始: `QUICKSTART_EXAMPLES.md`
