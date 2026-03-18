# Schema 記錄功能使用說明

## 概述

已實作完整的 Schema 資訊記錄功能，可在評估結果中記錄每個 RAG Pipeline 使用的 Schema 詳細資訊。

## 功能特點

1. **完整 Schema 資訊記錄**
   - Schema 生成方法名稱
   - Entity types 列表（實體類型）
   - Relations 列表（關係類型）
   - Validation schema（驗證規則）

2. **高效儲存格式**
   - Schema 資訊只在 `detailed_results.csv` 的第一行記錄
   - 其他行保持空白，避免重複
   - 使用 JSON 格式儲存複雜資料結構

3. **向後兼容**
   - 不影響現有程式碼
   - `get_schema_by_method()` 支援舊版只返回 entity types 的方式

## 使用方式

### 測試單一 Schema 方法

```bash
# 測試 lightrag_default
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode hybrid \
    --lightrag_schema_method lightrag_default \
    --data_type DI \
    --qa_dataset_fast_test

# 測試 iterative_evolution
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode hybrid \
    --lightrag_schema_method iterative_evolution \
    --data_type DI \
    --qa_dataset_fast_test

# 測試 llm_dynamic
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode hybrid \
    --lightrag_schema_method llm_dynamic \
    --data_type DI \
    --qa_dataset_fast_test
```

### 批次測試所有 Schema 方法

```bash
# 執行預設的批次測試腳本
bash /home/End_to_End_RAG/run_all_schema_tests.sh
```

## 結果檔案格式

### detailed_results.csv

每個 Pipeline 的詳細結果檔案會包含以下 Schema 欄位：

| 欄位 | 說明 | 範例 |
|------|------|------|
| schema_method | Schema 生成方法 | "lightrag_default" |
| schema_entities | 實體類型列表（JSON） | ["Person", "Organization", "Location"] |
| schema_relations | 關係類型列表（JSON） | ["WORKS_AT", "LOCATED_IN"] |
| schema_validation | 驗證規則（JSON） | {"Person": ["WORKS_AT"]} |

**重要**: 這些欄位只在第一行（idx=1）有值，其他行為空字串。

### 範例

```csv
idx,query,...,schema_method,schema_entities,schema_relations,schema_validation,...
1,"問題1",...,lightrag_default,"[""Person"", ""Organization""]",[],{},...
2,"問題2",...,,,,...
平均,,...,,,,...
```

## 支援的 Schema 方法

1. **lightrag_default**
   - LightRAG 預設實體類型
   - 包含：Person, Creature, Organization, Location, Event, Concept, Method, Content, Data, Artifact, NaturalObject

2. **iterative_evolution**
   - 基於文本演化的動態 Schema
   - 使用 Pydantic 模型進行結構化演化
   - 包含完整的 entities, relations, validation_schema

3. **llm_dynamic**
   - LLM 動態生成的 Schema
   - 連接 Ollama API 進行實體抽取
   - 最多生成 10 個最重要的實體類型

## 程式碼架構

### 修改的檔案

1. **src/rag/wrappers/base_wrapper.py**
   - 新增 `schema_info` 屬性

2. **src/rag/schema/factory.py**
   - `get_schema_by_method()` 支援返回完整 schema 物件
   - 新增 `return_full_schema` 參數

3. **src/rag/wrappers/lightrag_wrapper.py**
   - `LightRAGWrapper` 接收並儲存 `schema_info`
   - `LightRAGWrapper_Original` 同步更新

4. **src/rag/wrappers/temporal_wrapper.py**
   - `TemporalLightRAGWrapper` 接收並儲存 `schema_info`

5. **src/rag/wrappers/vector_wrapper.py**
   - `VectorRAGWrapper` 接收並儲存 `schema_info`

6. **scripts/run_evaluation.py**
   - `setup_lightrag_pipeline()` 取得並傳遞完整 schema 資訊

7. **src/evaluation/evaluator.py**
   - `compute_metrics_for_sample()` 接收 `schema_info` 參數
   - 只在第一筆資料記錄 schema 資訊

8. **src/evaluation/reporters.py**
   - 確保 schema 欄位正確儲存到 CSV

## API 參考

### get_schema_by_method()

```python
def get_schema_by_method(
    method: str,
    text_corpus: list = None,
    settings = None,
    return_full_schema: bool = True
) -> Union[dict, list]:
    """
    根據指定方法，回傳 schema 資訊
    
    Args:
        method: Schema 生成方法
        text_corpus: 文本語料
        settings: 設定物件
        return_full_schema: 若為 True，返回完整 schema 物件；
                           若為 False，只返回 entity types 列表
    
    Returns:
        若 return_full_schema=True:
            {
                "method": str,
                "entities": list,
                "relations": list,
                "validation_schema": dict
            }
        若 return_full_schema=False:
            list（entity types）
    """
```

### BaseRAGWrapper

```python
class BaseRAGWrapper(ABC):
    def __init__(self, name: str, schema_info: Dict[str, Any] = None):
        """
        初始化 Wrapper
        
        Args:
            name: Wrapper 名稱
            schema_info: Schema 資訊字典，包含 method、entities、
                        relations、validation_schema
        """
```

## 驗證測試

已通過的測試：

✅ Schema Factory 返回完整物件
✅ BaseRAGWrapper 儲存 schema_info 屬性
✅ LightRAGWrapper 接收並儲存 schema 資訊
✅ 評估流程正確傳遞 schema 資訊
✅ CSV 檔案正確記錄 schema 欄位
✅ 只在第一行記錄，其他行為空

## 注意事項

1. Schema 資訊只在第一筆資料（idx=1）記錄，避免重複
2. 使用 JSON 格式儲存複雜資料結構（確保 `ensure_ascii=False` 以正確顯示中文）
3. 向後兼容：舊程式碼不需要修改，`schema_info` 參數為可選
4. 平均行的 schema 欄位自動保持空白

## 故障排除

### 問題：Schema 欄位全部為空

**解決方案**：
- 確認 `get_schema_by_method()` 使用 `return_full_schema=True`
- 檢查 `setup_lightrag_pipeline()` 是否正確傳遞 `schema_info`

### 問題：JSON 格式錯誤

**解決方案**：
- 確保使用 `json.dumps()` 轉換為字串
- 使用 `ensure_ascii=False` 參數以正確處理中文

### 問題：TypeError 關於 schema_info

**解決方案**：
- 檢查所有 Wrapper 類別是否都更新為接收 `schema_info` 參數
- 確認基底類別 `BaseRAGWrapper` 的 `__init__()` 已更新

## 相關文件

- [API 文檔](docs/API.md)
- [評估指標說明](docs/METRICS.md)
- [主要 README](README.md)
