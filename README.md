# End-to-End RAG 評估框架

一個完整、模組化的 RAG (Retrieval-Augmented Generation) Pipeline 評估框架，支援多種檢索方法與全面的評估指標。

## 專案特色

- ✨ **模組化設計**：清晰的程式碼結構，易於維護與擴展
- 🎯 **多種 RAG 方法**：支援 Vector RAG、Graph RAG、LightRAG 等多種方法
- 📊 **全面評估**：涵蓋檢索指標、生成指標、LLM-as-Judge 評估
- ⚡ **高效能**：支援非同步執行與快取機制
- 🔧 **易於擴展**：基於抽象類別與工廠模式，方便新增新方法或指標

## 快速開始

### 安裝依賴

```bash
pip install -r requirements.txt
```

### 基本使用

#### 1. Vector RAG 評估

```bash
python scripts/run_evaluation.py \
    --vector_method hybrid \
    --data_type DI \
    --top_k 2
```

#### 2. LightRAG 評估

```bash
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode hybrid \
    --data_type DI
```

#### 3. 快速測試模式

```bash
python scripts/run_evaluation.py \
    --vector_method hybrid \
    --qa_dataset_fast_test
```

## 專案結構

```
End_to_End_RAG/
├── src/                      # 核心源碼
│   ├── config/               # 配置管理
│   ├── data/                 # 資料處理
│   ├── rag/                  # RAG Pipeline 實作
│   │   ├── wrappers/         # RAG 封裝器
│   │   ├── vector/           # Vector RAG
│   │   ├── graph/            # Graph RAG
│   │   └── schema/           # Schema 管理
│   ├── graph_builder/        # 圖譜建構
│   ├── graph_retriever/      # 圖譜檢索
│   ├── evaluation/           # 評估系統
│   │   └── metrics/          # 評估指標
│   └── utils/                # 工具函數
├── scripts/                  # 執行腳本
│   └── run_evaluation.py     # 主執行入口
├── tests/                    # 測試
├── docs/                     # 文檔
├── Data/                     # 資料目錄
├── results/                  # 評估結果
└── config.yml                # 全域配置
```

## 支援的 RAG 方法

### Vector RAG

- **Hybrid**: Vector + BM25 混合檢索
- **Vector**: 純向量檢索
- **BM25**: 基於 BM25 的關鍵字檢索

### 進階 Vector RAG

- **Self-Query**: 帶有 metadata 過濾的自查詢
- **Parent-Child**: 多層級檢索（Auto Merging）

### Graph RAG

- **Property Graph**: 基於 PropertyGraphIndex 的圖譜檢索
- **Dynamic Schema**: 動態 Schema 生成的圖譜檢索
- **LightRAG**: 支援 local/global/hybrid/mix/naive/bypass 模式

## 評估指標

### 檢索指標

- **Hit Rate**: 命中率
- **MRR**: 平均倒數排名
- **Precision**: 精準度
- **Recall**: 召回率
- **F1 Score**: F1 分數

### 生成指標

- **ROUGE**: ROUGE-1/2/L/Lsum
- **BLEU**: BLEU 分數
- **METEOR**: METEOR 分數
- **BERTScore**: 基於 BERT embeddings 的語義相似度
- **Token F1**: 字元層級 F1
- **Jieba F1**: 詞彙層級 F1（使用 jieba 分詞）

### LLM-as-Judge

- **Correctness**: 答案正確性（0-5 分）
- **Faithfulness**: 答案忠實度（0/1 分）

## 命令列參數

### RAG 方法選擇

```bash
--vector_method {none,hybrid,vector,bm25,all}
--adv_vector_method {none,parent_child,self_query,all}
--graph_rag_method {none,propertyindex,lightrag,dynamic_schema,all}
--lightrag_mode {none,local,global,hybrid,mix,naive,bypass,original,all}
```

### 資料參數

```bash
--data_type {DI,GEN}              # 資料類型
--data_mode {natural_text,markdown,key_value_text,unstructured_text}  # 資料格式
--model_type {small,big}          # 模型大小
```

### 評估參數

```bash
--top_k 2                         # 檢索數量
--qa_dataset_fast_test            # 快速測試（僅前 2 題）
--postfix my_experiment           # 結果資料夾後綴
```

### Schema 相關

```bash
--lightrag_schema_method {lightrag_default,iterative_evolution,llm_dynamic}
--lightrag_temporal_graph         # 啟用時序 LightRAG
```

完整參數說明請參考 [API 文檔](docs/API.md)。

## 評估結果

評估結果儲存在 `results/evaluation_results_{timestamp}{postfix}/` 目錄下：

```
results/evaluation_results_20260317_120000_DI_lightrag_hybrid/
├── global_summary_report.csv                  # 所有 Pipeline 比較
├── LightRAG_Hybrid/
│   └── detailed_results.csv                   # 逐題評估結果
└── Vector_hybrid_RAG/
    └── detailed_results.csv
```

### 報告內容

- **detailed_results.csv**: 每題的詳細評估結果（含所有指標）
- **global_summary_report.csv**: 各 Pipeline 的平均指標比較

## 進階使用

### 自訂評估指標

```python
from src.evaluation.metrics import BaseMetric, MetricRegistry

@MetricRegistry.register("custom_metric")
class CustomMetric(BaseMetric):
    def compute(self, **kwargs):
        # 自訂邏輯
        return score
```

### 新增 RAG Pipeline

```python
from src.rag.wrappers import BaseRAGWrapper

class MyRAGWrapper(BaseRAGWrapper):
    async def _execute_query(self, query: str):
        # 實作查詢邏輯
        return {
            "generated_answer": answer,
            "retrieved_contexts": contexts,
            "retrieved_ids": ids,
            "source_nodes": nodes,
        }
```

詳細說明請參考 [API 文檔](docs/API.md)。

## 測試

```bash
# 執行所有測試
python -m pytest tests/

# 執行特定測試
python -m pytest tests/test_metrics.py
```

## 貢獻指南

1. Fork 此專案
2. 建立 feature 分支 (`git checkout -b feature/AmazingFeature`)
3. 提交變更 (`git commit -m 'Add some AmazingFeature'`)
4. Push 到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 授權

此專案採用 MIT 授權 - 詳見 LICENSE 檔案

## 文檔

- [API 參考](docs/API.md)
- [評估指標說明](docs/METRICS.md)
- [使用範例](docs/EXAMPLES.md)

## 聯絡方式

如有任何問題或建議，歡迎開啟 Issue 討論。
