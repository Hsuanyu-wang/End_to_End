# API 參考文檔

## 目錄

- [配置管理](#配置管理)
- [資料處理](#資料處理)
- [RAG Wrappers](#rag-wrappers)
- [評估指標](#評估指標)
- [評估引擎](#評估引擎)

---

## 配置管理

### ModelSettings

模型設定管理類別，封裝 LlamaIndex Settings 並提供額外的配置管理功能。

```python
from src.config import ModelSettings, get_settings, my_settings

# 方式 1：使用全域單例（推薦）
from src.config import my_settings

llm = my_settings.llm
eval_llm = my_settings.eval_llm
embed_model = my_settings.embed_model

# 訪問資料配置
qa_path = my_settings.data_config.qa_file_path_DI
raw_path = my_settings.data_config.raw_file_path_DI

# 訪問 LightRAG 配置
entity_types = my_settings.lightrag_config.entity_types
storage_path = my_settings.lightrag_config.storage_path_DIR

# 方式 2：使用 get_settings() 函數
settings = get_settings(model_type="small")
llm = settings.llm
```

**ModelSettings 屬性**:
- `llm`: 主要 LLM 模型
- `eval_llm`: 評估用 LLM 模型
- `builder_llm`: 建圖用 LLM 模型
- `embed_model`: Embedding 模型
- `data_config`: DataConfig 實例（資料路徑配置）
- `lightrag_config`: LightRAGConfig 實例（LightRAG 配置）

### DataConfig

資料路徑配置管理類別

**屬性**:
- `raw_file_path_DI`: DI 原始資料路徑
- `raw_file_path_GEN`: GEN 原始資料路徑
- `qa_file_path_DI`: DI 問答資料路徑
- `qa_file_path_GEN`: GEN 問答資料路徑

### LightRAGConfig

LightRAG 專用配置管理類別

**屬性**:
- `storage_path_DIR`: LightRAG 儲存路徑
- `language`: LightRAG 語言設定
- `entity_types`: LightRAG 實體類型列表

**重要提醒**：
- LlamaIndex 的全域 Settings：`from llama_index.core import Settings as LlamaSettings`
- 我們的配置管理：`from src.config import my_settings`
- 避免混淆兩者，使用命名空間明確區分

---

## 資料處理

### DataProcessor

資料處理器類別

```python
from src.data.processors import DataProcessor, data_processing

# 使用類別
processor = DataProcessor(mode="natural_text", data_type="DI")
documents = processor.process()

# 使用函數（向後兼容）
documents = data_processing(mode="natural_text", data_type="DI")
```

**參數**:
- `mode`: 資料模式
  - `natural_text`: 自然語言格式（適合 Embedding）
  - `markdown`: Markdown 格式（明確區分背景與目標）
  - `key_value_text`: 鍵值對格式
  - `unstructured_text`: 非結構化格式（適合 LLM 抽取）
- `data_type`: 資料類型 (`DI` 或 `GEN`)

### QADataLoader

QA 資料載入器

```python
from src.data.loaders import QADataLoader

loader = QADataLoader(data_type="DI")
qa_datasets = loader.load_and_normalize()
```

**輸出格式**:
```python
{
    "source": str,              # 資料來源
    "query": str,               # 問題
    "ground_truth_answer": str, # 標準答案
    "ground_truth_doc_ids": List[str]  # 相關文件 ID
}
```

---

## RAG Wrappers

### BaseRAGWrapper

所有 Wrapper 的抽象基底類別

```python
from src.rag.wrappers import BaseRAGWrapper

class MyWrapper(BaseRAGWrapper):
    async def _execute_query(self, query: str):
        # 實作查詢邏輯
        return {
            "generated_answer": answer,
            "retrieved_contexts": contexts,
            "retrieved_ids": ids,
            "source_nodes": nodes,
        }
```

### VectorRAGWrapper

Vector RAG 封裝器

```python
from src.rag.wrappers import VectorRAGWrapper
from src.rag.vector import get_vector_query_engine

engine = get_vector_query_engine(
    Settings,
    vector_method="hybrid",
    top_k=2,
    data_mode="natural_text",
    data_type="DI"
)

wrapper = VectorRAGWrapper(name="My_Vector_RAG", query_engine=engine)

# 非同步查詢
result = await wrapper.aquery_and_log("問題內容")

# 同步查詢
result = wrapper.query_and_log("問題內容")
```

### LightRAGWrapper

LightRAG 封裝器

```python
from src.rag.wrappers import LightRAGWrapper
from src.rag.graph import get_lightrag_engine

lightrag = get_lightrag_engine(Settings, data_type="DI", sup="v1")

wrapper = LightRAGWrapper(
    name="My_LightRAG",
    rag_instance=lightrag,
    mode="hybrid",
    use_context=True  # 使用自訂 context 模式
)

result = await wrapper.aquery_and_log("問題內容")
```

**模式說明**:
- `local`: 關注特定實體與細節
- `global`: 關注整體趨勢與總結
- `hybrid`: 混合 local 與 global
- `mix`: 知識圖譜 + 向量檢索
- `naive`: 僅向量檢索
- `bypass`: 直接查詢 LLM

---

## 評估指標

### BaseMetric

所有指標的抽象基底類別

```python
from src.evaluation.metrics import BaseMetric

class MyMetric(BaseMetric):
    def compute(self, **kwargs):
        # 計算邏輯
        return score
```

### 檢索指標

```python
from src.evaluation.metrics import (
    HitRateMetric,
    MRRMetric,
    RetrievalF1Metric
)

# Hit Rate
hit_rate_metric = HitRateMetric()
hit_rate = hit_rate_metric.compute(
    retrieved_ids=["doc1", "doc2"],
    ground_truth_ids=["doc1", "doc3"]
)  # 返回 1（命中）

# MRR
mrr_metric = MRRMetric()
mrr = mrr_metric.compute(
    retrieved_ids=["doc2", "doc1", "doc3"],
    ground_truth_ids=["doc1"]
)  # 返回 0.5（第二位）

# Retrieval F1
f1_metric = RetrievalF1Metric()
recall, precision, f1 = f1_metric.compute(
    retrieved_ids=["doc1", "doc2", "doc3"],
    ground_truth_ids=["doc1", "doc4"]
)
```

### 生成指標

```python
from src.evaluation.metrics import (
    ROUGEMetric,
    BLEUMetric,
    JiebaF1Metric
)

# ROUGE
rouge_metric = ROUGEMetric()
r1, r2, rL, rLsum = rouge_metric.compute(
    generated_answer="生成的答案",
    ground_truth_answer="標準答案"
)

# Jieba F1
jieba_metric = JiebaF1Metric()
recall, precision, f1 = jieba_metric.compute(
    generated_answer="生成的答案",
    ground_truth_answer="標準答案"
)
```

### LLM-as-Judge

```python
from src.evaluation.metrics import (
    CorrectnessMetric,
    FaithfulnessMetric
)

# Correctness
correctness = CorrectnessMetric(llm=eval_llm)
score = await correctness.compute_async(
    query="問題",
    response="生成的答案",
    reference="標準答案"
)  # 返回 0-5 分

# Faithfulness
faithfulness = FaithfulnessMetric(llm=eval_llm)
score = await faithfulness.compute_async(
    query="問題",
    response="生成的答案",
    contexts=["檢索到的上下文1", "檢索到的上下文2"]
)  # 返回 0.0 或 1.0
```

---

## 評估引擎

### RAGEvaluator

RAG 評估器

```python
from src.evaluation import RAGEvaluator
from src.config import my_settings

evaluator = RAGEvaluator(
    eval_llm=my_settings.eval_llm,
    base_eval_dir="results/my_experiment"
)

# 評估單一 Pipeline
results = await evaluator.evaluate_pipeline(
    pipeline=wrapper,
    qa_datasets=datasets,
    pipeline_name="My_RAG"
)

# 計算單一樣本的指標
metrics = await evaluator.compute_metrics_for_sample(
    idx=1,
    source="QA_43.csv",
    query="問題",
    gt_answer="標準答案",
    gen_answer="生成的答案",
    gt_ids=["doc1"],
    retrieved_ids=["doc1", "doc2"],
    retrieved_contexts=["context1", "context2"],
    execution_time_sec=1.234
)
```

### EvaluationReporter

評估報告生成器

```python
from src.evaluation import EvaluationReporter

reporter = EvaluationReporter(base_dir="results/my_experiment")

# 儲存 Pipeline 結果
reporter.save_pipeline_results(
    pipeline_name="My_RAG",
    results=results
)

# 生成全局比較報告（寫入 global_summary_report.csv 與 global_summary_report.xlsx；run_evaluation 會附加彙總至 results/{exp|test}/{data_type}/global_summary.xlsx）
reporter.generate_global_summary(summary_records)
```

### 完整評估流程

```python
from src.evaluation import run_evaluation

# 自動執行完整評估流程
await run_evaluation(
    qa_datasets=datasets,
    pipelines=pipelines_to_test,
    postfix="_my_experiment"
)
```

---

## 指標註冊機制

使用 `MetricRegistry` 註冊自訂指標：

```python
from src.evaluation.metrics import BaseMetric, MetricRegistry

@MetricRegistry.register("my_custom_metric")
class MyCustomMetric(BaseMetric):
    def compute(self, **kwargs):
        # 自訂邏輯
        return score

# 取得已註冊的指標
metric = MetricRegistry.get("my_custom_metric")

# 列出所有已註冊的指標
all_metrics = MetricRegistry.list_names()
```

---

## 型別提示

```python
from typing import List, Dict, Any, Literal

DataMode = Literal["natural_text", "markdown", "key_value_text", "unstructured_text"]
DataType = Literal["DI", "GEN"]
```
