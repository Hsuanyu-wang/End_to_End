# RAGAS 指標整合文檔

## 概述

本專案已成功整合 RAGAS 評估指標到現有的評估系統中。RAGAS 指標與 LlamaIndex 指標互補使用，提供更全面的 RAG 系統評估。

## 新增指標

### 1. AnswerRelevancy (答案相關性)
- **定義**: 評估生成答案是否直接回答使用者問題
- **評估維度**: 答案的針對性、完整性、是否偏題
- **範圍**: [0, 1]（越高越好）
- **輸入**: query, response, ground_truth (可選)

### 2. ContextPrecision (上下文精準度)
- **定義**: 評估檢索到的上下文是否精準（少雜訊）
- **評估維度**: 相關上下文佔比、是否包含無關資訊
- **範圍**: [0, 1]（越高越好）
- **輸入**: query, contexts (實際檢索到的上下文), ground_truth

### 3. ContextRecall (上下文召回率)
- **定義**: 評估檢索到的上下文是否包含所有必要資訊
- **評估維度**: 是否遺漏重要上下文、資訊完整度
- **範圍**: [0, 1]（越高越好）
- **輸入**: query, contexts (實際檢索到的上下文), ground_truth

### 4. RAGASFaithfulness (RAGAS 版本忠實度)
- **定義**: 使用 RAGAS 框架評估生成答案是否忠實於檢索到的上下文
- **評估維度**: 是否包含幻覺、是否基於檢索結果生成
- **範圍**: [0, 1]（越高越好）
- **輸入**: query, response, contexts (實際檢索到的上下文)

## 與 LlamaIndex 指標的對比

| 指標類型 | LlamaIndex | RAGAS |
|---------|-----------|-------|
| **檢索品質** | Hit Rate, MRR, Retrieval F1 | Context Precision, Context Recall |
| **生成品質** | ROUGE, BLEU, METEOR, BERTScore, Token F1, Jieba F1 | Answer Relevancy |
| **LLM Judge** | Correctness (0-5), Faithfulness (0/1) | Faithfulness (0-1), Answer Relevancy (0-1) |

## 使用方式

### 在評估腳本中自動使用

RAGAS 指標已整合到 `RAGEvaluator` 中，會自動計算：

```python
from src.evaluation.evaluator import RAGEvaluator
from llama_index.core import Settings

# 建立評估器
evaluator = RAGEvaluator(eval_llm=Settings.eval_llm)

# 計算單一樣本的所有指標（包含 RAGAS）
result = await evaluator.compute_metrics_for_sample(
    idx=1,
    source="test_dataset",
    query="台灣的首都是哪裡？",
    gt_answer="台北市",
    gen_answer="台灣的首都是台北市。",
    gt_ids=["doc1"],
    retrieved_ids=["doc1", "doc2"],
    retrieved_contexts=["台北市是中華民國的首都..."],
    execution_time_sec=1.5
)

# 結果包含以下 RAGAS 指標：
# - result["answer_relevancy"]
# - result["context_precision"]
# - result["context_recall"]
# - result["ragas_faithfulness"]
```

### 單獨使用 RAGAS 指標

```python
from src.evaluation.metrics import (
    AnswerRelevancyMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    RAGASFaithfulnessMetric,
)

# 初始化指標
answer_relevancy = AnswerRelevancyMetric()
context_precision = ContextPrecisionMetric()

# 計算分數
score = await answer_relevancy.compute_async(
    query="台灣的首都是哪裡？",
    response="台灣的首都是台北市。",
    ground_truth="台北市"
)

print(f"Answer Relevancy: {score}")
```

## 環境配置

### 必要依賴

RAGAS 指標需要以下套件：

```bash
pip install ragas datasets
```

### API Key 配置

RAGAS 指標需要 API Key 才能執行（使用 LLM 進行評估）：

```bash
# 方法 1: 使用 OpenAI API
export OPENAI_API_KEY="your-api-key"

# 方法 2: 使用自定義 API 端點
export EVAL_LLM_BINDING_API_KEY="your-api-key"
export EVAL_LLM_BINDING_HOST="http://your-endpoint"
export EVAL_LLM_MODEL="gpt-4o-mini"

# 方法 3: 使用獨立的 Embedding 端點
export EVAL_EMBEDDING_BINDING_API_KEY="your-embedding-api-key"
export EVAL_EMBEDDING_BINDING_HOST="http://your-embedding-endpoint"
export EVAL_EMBEDDING_MODEL="text-embedding-3-large"
```

### 可選配置

```bash
# LLM 配置
export EVAL_LLM_MODEL="gpt-4o-mini"           # 預設: gpt-4o-mini
export EVAL_LLM_MAX_RETRIES="5"               # 預設: 5
export EVAL_LLM_TIMEOUT="180"                 # 預設: 180

# Embedding 配置
export EVAL_EMBEDDING_MODEL="text-embedding-3-large"  # 預設: text-embedding-3-large
```

## 優雅降級

如果未安裝 RAGAS 或未設定 API Key：

- RAGAS 指標會自動設為不可用（`available=False`）
- 評估腳本會繼續執行，RAGAS 指標欄位回傳 `None`
- 不影響其他指標的計算
- 會輸出警告訊息提示使用者

```
⚠️ RAGAS 指標需要 API Key，請設定環境變數
⚠️ RAGAS 指標計算失敗: ...
```

## 效能考量

### API 呼叫次數

每個 RAGAS 指標會產生額外的 LLM API 呼叫：

- **AnswerRelevancy**: ~2-3 次 LLM 呼叫
- **ContextPrecision**: ~2-3 次 LLM 呼叫
- **ContextRecall**: ~2-3 次 LLM 呼叫
- **RAGASFaithfulness**: ~2-3 次 LLM 呼叫

**總計**: 每個樣本約 8-12 次額外的 LLM 呼叫

### 執行時間

- 每個 RAGAS 指標約需 5-10 秒（取決於 LLM 回應速度）
- 建議在批次評估時使用較快的 LLM 模型（如 `gpt-4o-mini`）

### 成本估算

以 OpenAI `gpt-4o-mini` 為例：

- 每個樣本約 8-12 次呼叫
- 每次呼叫約 200-500 tokens
- 總計約 2000-6000 tokens/樣本
- 成本約 $0.0003-0.0009/樣本

## 測試

執行整合測試：

```bash
python tests/test_ragas_integration.py
```

測試項目：

1. ✅ RAGAS 指標導入
2. ✅ RAGAS 指標初始化
3. ✅ RAGAS 指標計算（需要 API Key）
4. ✅ RAGEvaluator 整合

## 參考實作

本整合參考了 `eval_rag_quality.py` 的正確實作，確保：

- ✅ 使用實際檢索到的上下文 (`retrieved_contexts`)
- ✅ 使用 `LangchainLLMWrapper` 包裝 LLM
- ✅ 啟用 `bypass_n` 模式以相容自定義端點
- ✅ 適當的錯誤處理與降級策略

## 注意事項

1. **Context 來源**: 所有 RAGAS 指標均使用**實際檢索到的上下文**（`retrieved_contexts`），而非 ground truth
2. **異步執行**: RAGAS 指標建議使用 `compute_async()` 方法
3. **向後相容**: 未安裝 RAGAS 時不影響現有評估流程
4. **API 限制**: 注意 API 速率限制，避免短時間內大量呼叫

## 故障排除

### 問題: RAGAS 套件未安裝

```
ImportWarning: answer_relevancy 需要 RAGAS 套件。請執行: pip install ragas datasets
```

**解決方案**:
```bash
pip install ragas datasets
```

### 問題: 缺少 API Key

```
UserWarning: RAGAS 指標需要 API Key，請設定環境變數
```

**解決方案**:
```bash
export OPENAI_API_KEY="your-api-key"
# 或
export EVAL_LLM_BINDING_API_KEY="your-api-key"
```

### 問題: API 超時

```
⚠️ RAGAS 指標計算失敗: Request timeout...
```

**解決方案**:
```bash
# 增加超時時間
export EVAL_LLM_TIMEOUT="300"
```

## 相關文件

- [RAGAS 官方文檔](https://docs.ragas.io/)
- [eval_rag_quality.py](../src/evaluation/eval_rag_quality.py) - 參考實作
- [evaluator.py](../src/evaluation/evaluator.py) - 整合實作
