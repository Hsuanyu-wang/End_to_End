# 評估指標說明

本文檔詳細說明所有評估指標的定義、計算方式與使用場景。

## 目錄

- [檢索指標](#檢索指標)
- [生成指標](#生成指標)
- [LLM-as-Judge 指標](#llm-as-judge-指標)

---

## 檢索指標

評估檢索系統的品質，衡量是否能找到相關文件。

### Hit Rate (命中率)

**定義**: 檢索結果中是否至少包含一個相關文件

**計算方式**:
```
Hit Rate = 1 if (任何檢索到的文件 ∈ 相關文件集合) else 0
```

**範圍**: `{0, 1}`

**適用場景**:
- 評估檢索系統的基本可用性
- 快速檢查是否完全漏檢

**範例**:
```python
retrieved_ids = ["doc1", "doc2", "doc3"]
ground_truth_ids = ["doc1", "doc5"]
hit_rate = 1  # doc1 被檢索到
```

---

### MRR (Mean Reciprocal Rank)

**定義**: 第一個相關文件在檢索結果中的排名倒數

**計算方式**:
```
MRR = 1 / (第一個相關文件的排名位置)
```

**範圍**: `(0, 1]`

**適用場景**:
- 評估檢索結果的排序品質
- 重視相關文件的排名位置

**範例**:
```python
retrieved_ids = ["doc2", "doc1", "doc3"]  # doc1 是相關文件
ground_truth_ids = ["doc1"]
mrr = 1 / 2 = 0.5  # doc1 在第 2 位
```

---

### Precision (精準度)

**定義**: 檢索到的文件中，相關文件的比例

**計算方式**:
```
Precision = |檢索到的相關文件| / |所有檢索文件|
```

**範圍**: `[0, 1]`

**適用場景**:
- 評估檢索結果的準確性
- 關注減少無關文件

**範例**:
```python
retrieved_ids = ["doc1", "doc2", "doc3"]
ground_truth_ids = ["doc1", "doc4"]
precision = 1 / 3 = 0.333  # 只有 doc1 相關
```

---

### Recall (召回率)

**定義**: 所有相關文件中，被檢索到的比例

**計算方式**:
```
Recall = |檢索到的相關文件| / |所有相關文件|
```

**範圍**: `[0, 1]`

**適用場景**:
- 評估檢索系統的完整性
- 關注避免遺漏相關文件

**範例**:
```python
retrieved_ids = ["doc1", "doc2", "doc3"]
ground_truth_ids = ["doc1", "doc4", "doc5"]
recall = 1 / 3 = 0.333  # 只檢索到 doc1
```

---

### F1 Score

**定義**: Precision 與 Recall 的調和平均

**計算方式**:
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**範圍**: `[0, 1]`

**適用場景**:
- 平衡 Precision 與 Recall
- 綜合評估檢索品質

---

## 生成指標

評估生成答案的品質，衡量與標準答案的相似度。

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

**定義**: 評估生成文本與參考文本之間的 n-gram 重疊程度

**變體**:
- **ROUGE-1**: unigram 重疊（單詞匹配）
- **ROUGE-2**: bigram 重疊（雙詞匹配）
- **ROUGE-L**: 最長公共子序列
- **ROUGE-Lsum**: 用於多句摘要

**範圍**: `[0, 1]`

**適用場景**:
- 評估摘要品質
- 關注詞彙重疊程度

**特點**:
- 重視召回率（Recall-Oriented）
- 不考慮詞序

---

### BLEU (Bilingual Evaluation Understudy)

**定義**: 評估生成文本與參考文本之間的精準匹配程度

**計算方式**: 基於 n-gram 精準度的幾何平均

**範圍**: `[0, 1]`

**適用場景**:
- 機器翻譯評估
- 對詞序敏感的場景

**特點**:
- 重視精準度（Precision-Oriented）
- 懲罰過短的生成結果

---

### METEOR (Metric for Evaluation of Translation with Explicit ORdering)

**定義**: 結合精準度與召回率，並考慮同義詞與詞幹變化

**範圍**: `[0, 1]`

**適用場景**:
- 評估翻譯或生成品質
- 考慮語義相似性

**特點**:
- 比 BLEU 更關注召回率
- 考慮詞彙變化（stemming, synonyms）

---

### BERTScore

**定義**: 使用 BERT embeddings 計算語義相似度

**計算方式**: 基於 BERT 的 token-level 相似度

**範圍**: `[0, 1]`

**適用場景**:
- 評估語義相似度而非字面匹配
- 適用於改寫或同義表達

**特點**:
- 考慮語義而非僅字面匹配
- 可適應不同語言（透過不同 BERT 模型）

**返回值**:
- Precision
- Recall
- F1

---

### Token F1 Score

**定義**: 基於字元集合的精準度與召回率

**計算方式**:
```
Token Set = {生成答案的所有字元}
Precision = |共同字元| / |生成答案字元|
Recall = |共同字元| / |標準答案字元|
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**範圍**: `[0, 1]`

**適用場景**:
- 快速評估文本相似度
- 相容中英混合文本

**特點**:
- 基於字元集合（不考慮順序）
- 計算簡單快速

---

### Jieba F1 Score

**定義**: 使用 jieba 分詞後計算詞彙層級的 F1 Score

**計算方式**:
1. 使用 jieba 進行中文分詞
2. 過濾純標點符號
3. 計算詞彙集合的 Precision、Recall、F1

**範圍**: `[0, 1]`

**適用場景**:
- 中文或中英混合文本評估
- 考慮詞彙而非字元

**特點**:
- 更符合人類理解（詞彙層級）
- 適用於中文場景

**範例**:
```python
generated = "我喜歡吃蘋果"
ground_truth = "我愛吃蘋果"

# jieba 分詞
gen_tokens = ["我", "喜歡", "吃", "蘋果"]
gt_tokens = ["我", "愛", "吃", "蘋果"]

# 計算
common = {"我", "吃", "蘋果"}  # 共 3 個
precision = 3 / 4 = 0.75
recall = 3 / 4 = 0.75
f1 = 0.75
```

---

## LLM-as-Judge 指標

使用 LLM 作為評估者，提供更接近人類判斷的評估。

### Correctness (正確性)

**定義**: 使用 LLM 評估生成答案與標準答案的一致性

**評估維度**:
- 語義正確性
- 資訊完整性
- 準確度

**範圍**: `[0, 5]`

**評分標準**:
- **5**: 完全正確，資訊完整
- **4**: 大部分正確，有些微差異
- **3**: 部分正確，遺漏部分資訊
- **2**: 僅少部分正確
- **1**: 幾乎完全錯誤
- **0**: 完全錯誤或無關

**適用場景**:
- 評估答案的實際可用性
- 考慮語義而非字面匹配

**注意事項**:
- 需要消耗 LLM API
- 評估速度較慢
- 可能受 LLM 能力影響

---

### Faithfulness (忠實度)

**定義**: 使用 LLM 評估生成答案是否忠實於檢索到的上下文

**評估維度**:
- 是否包含幻覺（hallucination）
- 是否基於檢索結果生成
- 是否憑空捏造資訊

**範圍**: `{0, 1}`

**評分標準**:
- **1.0**: 忠實於上下文，無幻覺
- **0.0**: 包含幻覺或憑空捏造

**適用場景**:
- 評估 RAG 系統的可靠性
- 檢測幻覺問題

**注意事項**:
- 需要提供檢索到的上下文
- 需要消耗 LLM API

---

## 指標選擇建議

### 檢索系統評估

優先使用：
- **F1 Score**: 綜合評估
- **MRR**: 評估排序品質
- **Recall**: 評估完整性

### 生成品質評估

中文場景：
- **Jieba F1**: 詞彙層級評估
- **BERTScore**: 語義相似度
- **Correctness**: 實際可用性

英文場景：
- **ROUGE**: 詞彙重疊
- **BLEU**: 精準匹配
- **BERTScore**: 語義相似度

### RAG 系統評估

建議組合：
- **檢索**: F1 + MRR
- **生成**: Jieba F1 + BERTScore
- **LLM-as-Judge**: Correctness + Faithfulness

---

## 指標限制

### 傳統 NLP 指標

- 僅考慮字面匹配，不考慮語義
- 可能低估同義改寫的品質
- 對語言敏感（BLEU、ROUGE 主要針對英文）

### BERTScore

- 依賴 BERT 模型品質
- 計算成本較高
- 可能受模型偏見影響

### LLM-as-Judge

- 需要消耗 LLM API（成本）
- 評估速度慢
- 可能不一致（受 prompt 影響）
- 可能受 LLM 能力限制

---

## 參考文獻

- ROUGE: Lin, C. Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries
- BLEU: Papineni, K., et al. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation
- BERTScore: Zhang, T., et al. (2019). BERTScore: Evaluating Text Generation with BERT
- METEOR: Banerjee, S., & Lavie, A. (2005). METEOR: An Automatic Metric for MT Evaluation
