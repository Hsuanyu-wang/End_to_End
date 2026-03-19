# Settings 使用快速參考

## 基本使用

### 導入配置
```python
# 推薦方式：使用全域單例
from src.config import my_settings

# 訪問模型
llm = my_settings.llm
eval_llm = my_settings.eval_llm
builder_llm = my_settings.builder_llm
embed_model = my_settings.embed_model

# 訪問資料配置
qa_path = my_settings.data_config.qa_file_path_DI
raw_path = my_settings.data_config.raw_file_path_GEN

# 訪問 LightRAG 配置
entity_types = my_settings.lightrag_config.entity_types
storage_path = my_settings.lightrag_config.storage_path_DIR
language = my_settings.lightrag_config.language
```

### LlamaIndex Settings（當需要時）
```python
from llama_index.core import Settings as LlamaSettings

# 訪問 LlamaIndex 官方屬性
llm = LlamaSettings.llm
embed_model = LlamaSettings.embed_model
```

## 常見模式

### 在函數中使用
```python
def my_function():
    from src.config import my_settings
    
    # 使用配置
    llm = my_settings.llm
    # ...
```

### 在類別中使用
```python
class MyClass:
    def __init__(self, settings=None):
        if settings is None:
            from src.config import my_settings
            settings = my_settings
        
        self.llm = settings.llm
        self.data_path = settings.data_config.qa_file_path_DI
```

### 在測試中使用
```python
def test_my_feature():
    from src.config import get_settings
    
    # 獲取配置實例
    settings = get_settings()
    
    # 進行測試
    assert settings.llm is not None
```

## 配置屬性完整列表

### ModelSettings 屬性
- `llm` - 主要 LLM 模型
- `eval_llm` - 評估用 LLM 模型
- `builder_llm` - 建圖用 LLM 模型
- `embed_model` - Embedding 模型
- `data_config` - DataConfig 實例
- `lightrag_config` - LightRAGConfig 實例

### DataConfig 屬性
- `raw_file_path_DI` - DI 原始資料路徑
- `raw_file_path_GEN` - GEN 原始資料路徑
- `qa_file_path_DI` - DI 問答資料路徑
- `qa_file_path_GEN` - GEN 問答資料路徑

### LightRAGConfig 屬性
- `storage_path_DIR` - LightRAG 儲存路徑
- `language` - LightRAG 語言設定
- `entity_types` - LightRAG 實體類型列表

## 遷移範例

### 舊代碼
```python
from src.config.settings import get_settings
Settings = get_settings()

# 訪問 LLM
llm = Settings.llm

# 訪問自訂屬性（這些已經不存在了）
entity_types = Settings.lightrag_entity_types
qa_path = Settings.qa_file_path_DI
```

### 新代碼
```python
from src.config import my_settings

# 訪問 LLM（相同）
llm = my_settings.llm

# 訪問自訂屬性（新的結構化方式）
entity_types = my_settings.lightrag_config.entity_types
qa_path = my_settings.data_config.qa_file_path_DI
```

## 注意事項

1. **不要混淆兩個 Settings**：
   - `LlamaSettings`（from llama_index.core）- LlamaIndex 的全域物件
   - `my_settings`（from src.config）- 我們的配置管理

2. **使用 my_settings 而非 get_settings()**：
   - `my_settings` 是預先建立的全域單例，更方便
   - `get_settings()` 也可以用，但會返回相同的實例

3. **配置是單例**：
   - 多次調用 `get_settings()` 返回相同實例
   - 配置在程式啟動時初始化一次

4. **不要直接修改 LlamaIndex Settings 的自訂屬性**：
   - ❌ `LlamaSettings.my_custom_attr = value`
   - ✅ 使用 `my_settings.lightrag_config` 或 `my_settings.data_config`

## 故障排除

### 問題：找不到 my_settings
```python
# 錯誤
from src.config.settings import my_settings  # ❌

# 正確
from src.config import my_settings  # ✅
```

### 問題：訪問不到自訂屬性
```python
# 錯誤
from llama_index.core import Settings
entity_types = Settings.lightrag_entity_types  # ❌ 不存在

# 正確
from src.config import my_settings
entity_types = my_settings.lightrag_config.entity_types  # ✅
```

### 問題：單例沒有生效
```python
# 這樣會創建新實例（不推薦）
from src.config.settings import ModelSettings
settings = ModelSettings()  # ❌ 新實例

# 使用單例（推薦）
from src.config import my_settings  # ✅ 全域單例
# 或
from src.config import get_settings
settings = get_settings()  # ✅ 也是單例
```
