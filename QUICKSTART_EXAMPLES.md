# 快速開始範例

本文件提供快速開始使用新測試腳本的範例。

## 範例 1：測試單一 Vector RAG 方法（快速）

```bash
cd /home/End_to_End_RAG
python scripts/run_comprehensive_tests.py
```

然後按照提示：
1. 選擇 `1` (Vector RAG 測試)
2. 選擇 `1` (Vector Hybrid RAG)
3. 選擇 `1` (快速測試)
4. 選擇 `1` (DI 資料)

預期結果：約 30-60 秒完成，產生評估結果。

## 範例 2：測試所有 LightRAG Schema 方法

```bash
cd /home/End_to_End_RAG
python scripts/run_comprehensive_tests.py
```

然後按照提示：
1. 選擇 `4` (LightRAG Schema 方法測試)
2. 選擇 `4` (全部執行)
3. 選擇 `1` (快速測試)
4. 選擇 `1` (DI 資料)

預期結果：
- ✅ LightRAG (Default Schema)
- ✅ LightRAG (Iterative Evolution Schema)
- ✅ LightRAG (LLM Dynamic Schema)

總計約 3-5 分鐘。

## 範例 3：自訂選擇多個測試

```bash
cd /home/End_to_End_RAG
python scripts/run_comprehensive_tests.py
```

然後按照提示：
1. 選擇 `7` (自訂選擇)
2. 對於 Vector RAG：選擇 `y` 包含 Hybrid
3. 對於 Advanced Vector RAG：選擇 `n` 跳過
4. 對於 Graph RAG：選擇 `y` 包含 Property Graph
5. 對於 LightRAG Schema：選擇 `y` 包含 Default Schema
6. 對於 LightRAG Mode：全部選擇 `n`
7. 選擇 `1` (快速測試)
8. 選擇 `1` (DI 資料)

結果：只執行選擇的 3 個測試。

## 範例 4：執行完整實驗（論文用）

```bash
cd /home/End_to_End_RAG
bash scripts/run_all_experiments.sh
```

這會自動執行：
- 所有 Vector RAG 方法
- 所有 Advanced Vector RAG 方法
- 所有 Graph RAG 方法
- 所有 LightRAG Schema 方法

預期時間：3-6 小時（使用完整資料集）

結果檔案：
- `results/` 目錄：所有評估結果
- `experiment_logs/experiment_*.log`：文字日誌
- `experiment_logs/experiment_*_report.html`：HTML 報告

## 範例 5：只測試 Schema 方法（快速驗證）

```bash
cd /home/End_to_End_RAG
bash run_all_schema_tests.sh
```

這會快速測試所有 3 種 schema 方法（使用 fast_test 模式）。

預期時間：2-3 分鐘

## 檢視結果

### 查看詳細結果

```bash
# 查看最新的評估結果
ls -lt /home/End_to_End_RAG/results/ | head -5

# 查看特定結果
cd /home/End_to_End_RAG/results/evaluation_results_TIMESTAMP_*/
cat global_summary_report.csv
```

### 查看 Schema 資訊

```bash
# Schema 資訊記錄在 detailed_results.csv 的第一行
cd /home/End_to_End_RAG/results/evaluation_results_TIMESTAMP_*/LightRAG_*/
head -2 detailed_results.csv
```

## 常見使用情境

### 情境 A：快速驗證新功能

使用互動式腳本 + 快速測試模式 + 選擇 1-2 個方法

### 情境 B：比較不同 Schema 方法

使用 `run_all_schema_tests.sh` 或互動式腳本選擇選項 4

### 情境 C：完整方法比較（發表論文）

使用 `run_all_experiments.sh` 執行所有方法的完整評估

### 情境 D：調試特定方法

直接使用 `run_evaluation.py` 加上特定參數

## 故障排除

### 問題：腳本無法執行

```bash
# 確保有執行權限
chmod +x scripts/run_comprehensive_tests.py
chmod +x scripts/run_all_experiments.sh
chmod +x run_all_schema_tests.sh
```

### 問題：Ollama 連接失敗

```bash
# 檢查 Ollama 服務
curl http://192.168.63.174:11434/api/tags
```

### 問題：記憶體不足

使用快速測試模式或減少並行測試數量。

## 更多資訊

- [完整測試指南](docs/TESTING_GUIDE.md)
- [主要 README](README.md)
- [Schema 記錄功能](docs/SCHEMA_RECORDING.md)
