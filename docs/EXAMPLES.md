# 使用範例

本文檔提供常見使用場景的完整範例。

## 目錄

- [基礎評估](#基礎評估)
- [自訂評估指標](#自訂評估指標)
- [自訂 RAG Pipeline](#自訂-rag-pipeline)
- [批次評估](#批次評估)
- [結果分析](#結果分析)

---

## 基礎評估

### 範例 1: 評估 Vector RAG (Hybrid)

```bash
python scripts/run_evaluation.py \
    --vector_method hybrid \
    --data_type DI \
    --top_k 2
```

### 範例 2: 評估 LightRAG (所有模式)

```bash
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode all \
    --data_type DI \
    --lightrag_schema_method lightrag_default
```

### 範例 3: 比較多種 RAG 方法

```bash
python scripts/run_evaluation.py \
    --vector_method all \
    --graph_rag_method lightrag \
    --lightrag_mode hybrid \
    --data_type DI \
    --postfix comparison_test
```

### 範例 4: 快速測試

```bash
python scripts/run_evaluation.py \
    --vector_method hybrid \
    --graph_rag_method lightrag \
    --lightrag_mode hybrid \
    --qa_dataset_fast_test
```

---

## 自訂評估指標

### 範例 5: 建立自訂指標

```python
from src.evaluation.metrics import BaseMetric, MetricRegistry

@MetricRegistry.register("custom_semantic_similarity")
class SemanticSimilarityMetric(BaseMetric):
    """使用自訂的語義相似度計算"""
    
    def __init__(self):
        super().__init__(
            name="semantic_similarity",
            description="自訂語義相似度指標"
        )
        # 初始化自己的模型
        self.model = load_similarity_model()
    
    def compute(self, generated_answer: str, ground_truth_answer: str) -> float:
        """計算語義相似度"""
        embedding1 = self.model.encode(generated_answer)
        embedding2 = self.model.encode(ground_truth_answer)
        
        similarity = cosine_similarity(embedding1, embedding2)
        return float(similarity)

# 使用自訂指標
metric = MetricRegistry.get("custom_semantic_similarity")
score = metric.compute(
    generated_answer="生成的答案",
    ground_truth_answer="標準答案"
)
print(f"語義相似度: {score}")
```

### 範例 6: 在評估中使用自訂指標

```python
from src.evaluation import RAGEvaluator
from src.config.settings import get_settings

# 註冊自訂指標（見範例 5）
# ...

# 建立評估器
from src.config import my_settings
evaluator = RAGEvaluator(eval_llm=my_settings.eval_llm)

# 評估時會自動使用所有已註冊的指標
results = await evaluator.evaluate_pipeline(
    pipeline=my_pipeline,
    qa_datasets=datasets
)
```

---

## 自訂 RAG Pipeline

### 範例 7: 建立自訂 Wrapper

```python
from src.rag.wrappers import BaseRAGWrapper
from typing import Dict, Any

class CustomRAGWrapper(BaseRAGWrapper):
    """自訂的 RAG Wrapper"""
    
    def __init__(self, name: str, custom_engine):
        super().__init__(name)
        self.engine = custom_engine
    
    async def _execute_query(self, query: str) -> Dict[str, Any]:
        """實作自訂的查詢邏輯"""
        # 步驟 1: 查詢改寫
        rewritten_query = await self._rewrite_query(query)
        
        # 步驟 2: 檢索
        contexts = await self.engine.retrieve(rewritten_query)
        
        # 步驟 3: 重排序
        reranked_contexts = self._rerank(contexts, query)
        
        # 步驟 4: 生成
        answer = await self.engine.generate(query, reranked_contexts)
        
        return {
            "generated_answer": answer,
            "retrieved_contexts": [c.text for c in reranked_contexts],
            "retrieved_ids": [c.id for c in reranked_contexts],
            "source_nodes": reranked_contexts,
        }
    
    async def _rewrite_query(self, query: str) -> str:
        """查詢改寫邏輯"""
        # 實作查詢改寫
        return rewritten_query
    
    def _rerank(self, contexts, query):
        """重排序邏輯"""
        # 實作重排序
        return reranked_contexts

# 使用自訂 Wrapper
wrapper = CustomRAGWrapper(
    name="My_Custom_RAG",
    custom_engine=my_engine
)

result = await wrapper.aquery_and_log("問題內容")
```

### 範例 8: 混合多種檢索方法

```python
from src.rag.wrappers import BaseRAGWrapper
from src.rag.vector import get_vector_query_engine
from src.rag.graph import get_lightrag_engine

class HybridRAGWrapper(BaseRAGWrapper):
    """混合 Vector 與 Graph RAG"""
    
    def __init__(self, name: str, vector_engine, graph_engine, settings):
        super().__init__(name)
        self.vector_engine = vector_engine
        self.graph_engine = graph_engine
        self.llm = settings.llm
    
    async def _execute_query(self, query: str) -> Dict[str, Any]:
        """混合檢索策略"""
        # 平行檢索
        vector_response = await self.vector_engine.aquery(query)
        graph_response = await self.graph_engine.aquery(query)
        
        # 合併結果
        all_contexts = []
        all_ids = []
        
        for node in vector_response.source_nodes:
            all_contexts.append(node.get_content())
            all_ids.append(node.metadata.get("NO", ""))
        
        # 使用 graph_response 的 context（如果有）
        # ...
        
        # 使用合併的 context 生成答案
        combined_text = "\n\n---\n\n".join(all_contexts[:5])  # 取前 5 個
        
        prompt = f"根據以下資料回答問題：\n\n{combined_text}\n\n問題：{query}\n\n答案："
        llm_response = await self.llm.acomplete(prompt)
        
        return {
            "generated_answer": llm_response.text,
            "retrieved_contexts": all_contexts,
            "retrieved_ids": all_ids,
            "source_nodes": vector_response.source_nodes,
        }

# 使用混合 Wrapper
from src.config.settings import get_settings

Settings = get_settings()
vector_engine = get_vector_query_engine(Settings, vector_method="hybrid", top_k=3)
graph_engine = get_lightrag_engine(Settings, data_type="DI")

wrapper = HybridRAGWrapper(
    name="Hybrid_Vector_Graph_RAG",
    vector_engine=vector_engine,
    graph_engine=graph_engine,
    Settings=Settings
)
```

---

## 批次評估

### 範例 9: 評估多個 Pipeline 並比較

```python
import asyncio
from src.config.settings import get_settings
from src.data.loaders import load_and_normalize_qa_CSR_DI
from src.rag.vector import get_vector_query_engine
from src.rag.graph import get_lightrag_engine
from src.rag.wrappers import VectorRAGWrapper, LightRAGWrapper
from src.evaluation import run_evaluation

async def main():
    # 載入設定與資料
    from src.config import my_settings
    datasets = load_and_normalize_qa_CSR_DI(csv_path=my_settings.data_config.qa_file_path_DI)
    
    # 建立多個 Pipeline
    pipelines = []
    
    # Vector RAG - Hybrid
    vector_engine = get_vector_query_engine(
        my_settings,
        vector_method="hybrid",
        top_k=2,
        data_type="DI"
    )
    pipelines.append(VectorRAGWrapper(name="Vector_Hybrid", query_engine=vector_engine))
    
    # Vector RAG - Pure Vector
    vector_engine2 = get_vector_query_engine(
        Settings,
        vector_method="vector",
        top_k=2,
        data_type="DI"
    )
    pipelines.append(VectorRAGWrapper(name="Vector_Pure", query_engine=vector_engine2))
    
    # LightRAG - Hybrid
    lightrag = get_lightrag_engine(Settings, data_type="DI")
    pipelines.append(LightRAGWrapper(
        name="LightRAG_Hybrid",
        rag_instance=lightrag,
        mode="hybrid"
    ))
    
    # LightRAG - Local
    pipelines.append(LightRAGWrapper(
        name="LightRAG_Local",
        rag_instance=lightrag,
        mode="local"
    ))
    
    # 執行評估
    await run_evaluation(
        qa_datasets=datasets[:5],  # 測試前 5 題
        pipelines=pipelines,
        postfix="_batch_comparison"
    )

# 執行
asyncio.run(main())
```

### 範例 10: 自訂評估流程

```python
from src.evaluation import RAGEvaluator, EvaluationReporter
from src.config.settings import get_settings
import pandas as pd

async def custom_evaluation():
    from src.config import my_settings
    
    # 建立評估器與報告器
    evaluator = RAGEvaluator(
        eval_llm=my_settings.eval_llm,
        base_eval_dir="results/custom_experiment"
    )
    reporter = EvaluationReporter(base_dir="results/custom_experiment")
    
    # 評估各 Pipeline
    all_summaries = []
    
    for pipeline in pipelines:
        print(f"\n評估 {pipeline.name}")
        
        results = await evaluator.evaluate_pipeline(
            pipeline=pipeline,
            qa_datasets=datasets
        )
        
        # 儲存結果
        reporter.save_pipeline_results(pipeline.name, results)
        
        # 提取總結
        df = pd.DataFrame(results)
        summary = reporter.extract_summary_from_df(pipeline.name, df)
        all_summaries.append(summary)
    
    # 生成比較報告
    reporter.generate_global_summary(all_summaries)

asyncio.run(custom_evaluation())
```

---

## 結果分析

### 範例 11: 分析評估結果

```python
import pandas as pd
import matplotlib.pyplot as plt

# 讀取結果
df = pd.read_csv("results/evaluation_results_20260317_120000/global_summary_report.csv")

# 顯示平均 F1 Score
print("各 Pipeline 的平均 F1 Score:")
print(df[["pipeline_name", "avg_retrieval_f1_score", "avg_jieba_f1"]].sort_values("avg_jieba_f1", ascending=False))

# 視覺化比較
pipelines = df["pipeline_name"].tolist()
f1_scores = df["avg_jieba_f1"].tolist()

plt.figure(figsize=(10, 6))
plt.barh(pipelines, f1_scores)
plt.xlabel("Jieba F1 Score")
plt.title("Pipeline 比較")
plt.tight_layout()
plt.savefig("pipeline_comparison.png")
```

### 範例 12: 詳細結果分析

```python
# 讀取詳細結果
df_detail = pd.read_csv("results/evaluation_results_20260317_120000/LightRAG_Hybrid/detailed_results.csv")

# 分析表現不佳的題目
poor_performance = df_detail[df_detail["jieba_f1"] < 0.3]
print(f"表現不佳的題目（F1 < 0.3）: {len(poor_performance)} 題")
print(poor_performance[["query", "jieba_f1", "generated_answer"]])

# 分析執行時間
print(f"平均執行時間: {df_detail['execution_time_sec'].mean():.2f} 秒")
print(f"最長執行時間: {df_detail['execution_time_sec'].max():.2f} 秒")

# 比較不同指標的相關性
correlation = df_detail[["retrieval_f1_score", "jieba_f1", "bertscore_f1", "correctness_score"]].corr()
print("\n指標相關性:")
print(correlation)
```

---

## 進階技巧

### 範例 13: 使用不同的資料模式

```python
# 比較不同資料格式的效果
data_modes = ["natural_text", "markdown", "unstructured_text"]

for mode in data_modes:
    engine = get_vector_query_engine(
        Settings,
        vector_method="hybrid",
        top_k=2,
        data_mode=mode,  # 使用不同格式
        data_type="DI"
    )
    
    wrapper = VectorRAGWrapper(name=f"Vector_{mode}", query_engine=engine)
    # 評估...
```

### 範例 14: Schema 實驗

```python
# 比較不同 Schema 生成方法
schema_methods = ["lightrag_default", "iterative_evolution", "llm_dynamic"]

for method in schema_methods:
    # 建立 LightRAG（會自動使用指定的 schema 方法）
    lightrag = get_lightrag_engine(
        Settings,
        data_type="DI",
        sup=f"schema_{method}"
    )
    
    wrapper = LightRAGWrapper(
        name=f"LightRAG_{method}",
        rag_instance=lightrag,
        mode="hybrid"
    )
    # 評估...
```

---

## 常見問題

### Q: 如何加速評估？

A: 使用快速測試模式：
```bash
python scripts/run_evaluation.py \
    --vector_method hybrid \
    --qa_dataset_fast_test
```

### Q: 如何只計算部分指標？

A: 修改 `RAGEvaluator._init_metrics()` 方法，移除不需要的指標。

### Q: 如何處理 LLM API 超時？

A: 在 `config.yml` 中調整 `request_timeout` 參數：
```yaml
model:
  request_timeout: 600.0  # 增加超時時間
```

### Q: 如何保存中間結果？

A: 評估結果會自動保存到 `results/` 目錄，包括詳細的逐題結果。
