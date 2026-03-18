# Retrieval Token 限制功能實作總結

## 修改目的

為了確保不同 RAG 方法之間的公平比較，從限制 **生成 token 數** 改為限制 **檢索內容 token 數**。

## 主要修改

### 1. 新增參數 `retrieval_max_tokens`

**位置**: `scripts/run_evaluation.py` Line 115

```python
parser.add_argument("--retrieval_max_tokens", type=int, default=2048, help="檢索內容最大 token 數（限制傳給 LLM 的 context 長度）")
```

**說明**: 
- 預設值 2048 tokens
- 用於限制所有 RAG 方法檢索到的內容總 token 數
- 確保不同方法在相同 context budget 下進行比較

### 2. 隱藏 `generation_max_tokens` 功能

**修改檔案**:
- `scripts/run_evaluation.py`: 註解掉相關程式碼
- `src/rag/vector/basic.py`: 註解掉 `_llm_with_generation_cap` 函數

**原因**: 
- 限制生成長度會影響回答品質
- 不同方法應該在相同輸入條件下比較，而非相同輸出長度

### 3. 核心實作：BaseRAGWrapper

**新增方法** (`src/rag/wrappers/base_wrapper.py`):

#### `set_retrieval_max_tokens(max_tokens: int)`
設定檢索內容的最大 token 數

#### `_truncate_contexts_by_tokens(contexts: List[str]) -> List[str]`
根據 token 數量截斷檢索結果：
1. 逐個累加 context 的 token 數
2. 當總和超過 `retrieval_max_tokens` 時停止
3. 如果剩餘 token > 50，會部分截取當前 context

### 4. TokenCounter 擴充

**新增方法** (`src/utils/token_counter.py`):

#### `truncate_text_by_tokens(text: str, max_tokens: int) -> str`
- 使用 tiktoken 精確計算 token 數
- 截斷 token 陣列後解碼回文本
- Fallback: 簡單字符截斷（1 token ≈ 4 字符）

### 5. 各 Wrapper 整合

**已更新的 Wrapper**:

1. **VectorRAGWrapper** (`vector_wrapper.py`)
   - 在 `_execute_query` 中調用 `_truncate_contexts_by_tokens`

2. **LightRAGWrapper** (`lightrag_wrapper.py`)
   - 優先使用 `retrieval_max_tokens`
   - 相容既有的 `token_budget` 機制
   - 移除 `generation_max_tokens` 相關代碼

3. **AutoSchemaWrapper** (`autoschema_wrapper.py`)
   - 在檢索後應用 token 限制
   - 移除生成 token 限制

4. **ModularGraphWrapper** (`modular_graph_wrapper.py`)
   - 整合 token 截斷功能
   - 移除生成 token 限制

5. **DynamicSchemaWrapper** (`dynamic_schema_wrapper.py`)
   - 在提取 contexts 後應用限制

### 6. 進階 Vector 方法

**更新檔案**: `src/rag/vector/advanced.py`

- `get_self_query_engine`: 新增 `retrieval_max_tokens` 參數
- `get_parent_child_query_engine`: 新增 `retrieval_max_tokens` 參數

## 使用範例

### 基本用法

```bash
# 限制檢索內容為 1024 tokens
python scripts/run_evaluation.py \
    --vector_method vector \
    --top_k 10 \
    --retrieval_max_tokens 1024
```

### 比較不同方法

```bash
# Vector RAG (hybrid, vector, bm25) 都限制在 2048 tokens
python scripts/run_evaluation.py \
    --vector_method all \
    --retrieval_max_tokens 2048
```

### LightRAG 比較

```bash
# 所有 LightRAG 模式使用相同的 context budget
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode all \
    --retrieval_max_tokens 2048
```

## 測試驗證

執行測試腳本：
```bash
python test_retrieval_token_limit.py
```

測試項目：
1. ✅ TokenCounter 的 truncate 功能
2. ✅ BaseRAGWrapper 的 context 截斷
3. ✅ 參數傳遞正確性

## 預期效果

### 公平比較
- 所有方法在相同 retrieval token budget 下運作
- 消除因檢索量不同造成的效能差異
- 更準確評估各方法的檢索品質

### 輸出日誌
```
🎯 [Vector_hybrid_RAG] 設定 retrieval_max_tokens=2048
📊 [Vector_hybrid_RAG] Context 截斷: 10 → 7 chunks (總 tokens ≤ 2048)
```

## 注意事項

1. **不影響生成長度**: LLM 可以根據需要自由生成回答
2. **相容性**: 與既有的 `token_budget` 機制共存（LightRAG）
3. **預設值**: 2048 tokens 適用於大多數場景
4. **動態調整**: 可根據實驗需求調整 `--retrieval_max_tokens` 參數
