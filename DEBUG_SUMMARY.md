# Debug 修復總結

## 問題分析與修復

### 問題 1: Jieba F1 測試失敗 ❌→✅

**錯誤訊息**:
```
FAILED tests/test_metrics.py::TestGenerationMetrics::test_jieba_f1_partial
assert 0.5 == 0.75 ± 0.075
```

**原因分析**:
測試案例假設的分詞結果與實際 jieba 分詞結果不同。

- **預期分詞**: `["我", "喜歡", "吃", "蘋果"]` 和 `["我", "愛", "吃", "蘋果"]`
- **實際分詞**: `["我", "喜歡", "吃", "蘋果"]` 和 `["我愛吃", "蘋果"]`

Jieba 將「我愛吃」視為一個完整的詞彙，導致計算結果不同。

**修復方式**:
更新測試案例，使用實際的 jieba 分詞結果：

```python
# 實際分詞結果:
# generated: ["我", "喜歡", "吃", "蘋果"] (4個詞)
# ground_truth: ["我愛吃", "蘋果"] (2個詞)
# 共同詞: ["蘋果"] (1個)

# Recall = 1/2 = 0.5
# Precision = 1/4 = 0.25
# F1 = 2 * 0.5 * 0.25 / (0.5 + 0.25) = 0.333

assert recall == pytest.approx(0.5, rel=1e-1)
assert precision == pytest.approx(0.25, rel=1e-1)
assert f1 == pytest.approx(0.333, rel=1e-1)
```

**修復檔案**: `tests/test_metrics.py`

---

### 問題 2: Import 路徑錯誤 ❌→✅

**錯誤訊息**:
```
ModuleNotFoundError: No module named 'End_to_End_RAG'
ModuleNotFoundError: No module named 'model_settings'
```

**原因分析**:
1. `run_evaluation.py` 使用了錯誤的 import 路徑（包含專案名稱）
2. 從舊檔案複製的模組（如 `vector_package.py`）仍使用舊的 import 路徑

**修復方式**:

#### 修復 1: 更新執行腳本

在 `scripts/run_evaluation.py` 開頭加入：
```python
import sys
sys.path.insert(0, '/home/End_to_End_RAG')
```

#### 修復 2: 批量更新所有模組的 import 路徑

使用 sed 命令批量替換：
```bash
# 替換 model_settings → src.config.settings
find src/ -name "*.py" -type f -exec sed -i 's/from model_settings import/from src.config.settings import/g' {} \;

# 替換 data_processing → src.data.processors
find src/ -name "*.py" -type f -exec sed -i 's/from data_processing import/from src.data.processors import/g' {} \;

# 替換 schema_factory → src.rag.schema.factory
find src/ -name "*.py" -type f -exec sed -i 's/from schema_factory import/from src.rag.schema.factory import/g' {} \;

# 替換 extract_schema_only → src.rag.schema.evolution
find src/ -name "*.py" -type f -exec sed -i 's/from extract_schema_only import/from src.rag.schema.evolution import/g' {} \;
```

**修復檔案**:
- `scripts/run_evaluation.py`
- `src/rag/vector/basic.py`
- `src/rag/vector/advanced.py`
- `src/rag/graph/*.py`
- `src/rag/schema/*.py`

---

## 驗證結果

### ✅ 所有測試通過

```bash
cd /home/End_to_End_RAG && python -m pytest tests/ -v
```

**結果**: 24 passed in 2.39s

### ✅ 執行腳本正常運作

```bash
cd /home/End_to_End_RAG && python scripts/run_evaluation.py --help
```

**結果**: 正常顯示幫助訊息

---

## 測試覆蓋情況

### 檢索指標測試 (8個)
- ✅ Hit Rate (命中/未命中/空輸入)
- ✅ MRR (第一位/第二位/未命中)
- ✅ Retrieval F1 (一般/完美/空輸入)

### 生成指標測試 (6個)
- ✅ Token F1 (完全相同/部分重疊/空輸入)
- ✅ Jieba F1 (完全相同/部分重疊)

### 資料處理測試 (8個)
- ✅ 各種資料模式 (natural_text, markdown, etc.)
- ✅ QA 資料載入 (CSV, JSONL)
- ✅ 格式正規化

### 邊界測試 (2個)
- ✅ 所有輸入為空
- ✅ Unicode 字元處理

---

## 關鍵學習點

### 1. Jieba 分詞不可預測性

Jieba 的分詞結果會根據詞庫和上下文變化，測試時應：
- 使用實際的分詞結果來驗證
- 或使用更寬鬆的斷言（較大的誤差範圍）

### 2. Python 模組路徑管理

在重構專案時，需要注意：
- 確保所有 import 路徑一致
- 使用 `sys.path.insert()` 來處理相對路徑
- 批量替換工具（sed, awk）可以加速修復

### 3. 測試驅動開發的價值

這次測試成功發現了：
- 分詞邏輯的假設錯誤
- Import 路徑的遺漏
- 如果沒有測試，這些問題會在執行時才被發現

---

## 後續建議

### 短期
1. ✅ 所有測試已通過
2. ✅ Import 路徑已統一
3. ⚠️ 建議執行實際的評估測試（使用真實資料）

### 中期
1. 增加更多整合測試
2. 建立 CI/CD Pipeline 自動執行測試
3. 增加測試覆蓋率報告

### 長期
1. 考慮使用 `setuptools` 建立可安裝的套件
2. 建立 Docker 容器簡化部署
3. 增加效能測試

---

## 驗證檢查清單

- [x] 所有單元測試通過
- [x] 執行腳本可正常啟動
- [x] Import 路徑正確
- [x] 測試案例準確反映實際行為
- [ ] 實際評估測試（需要資料）

專案已準備好投入使用！
