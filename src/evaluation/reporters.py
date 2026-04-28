"""
評估報告生成器

負責生成與儲存評估報告
"""

import os
from typing import List, Dict, Any, Optional
import pandas as pd
import json
from datetime import datetime


class EvaluationReporter:
    SUMMARY_NUMERIC_COLUMNS = [
        "execution_time_sec",
        "context_tokens",
        "entity_tokens",
        "relation_tokens",
        "chunk_tokens",
        "hit_rate",
        "mrr",
        "retrieval_recall",
        "retrieval_precision",
        "retrieval_f1_score",
        "rouge1",
        "rouge2",
        "rougeL",
        "rougeLsum",
        "bleu",
        "meteor",
        "bertscore_f1",
        "token_f1",
        "jieba_f1",
        "rouge1_zh",
        "rouge2_zh",
        "rougeL_zh",
        "rougeLsum_zh",
        "bleu_zh",
        "meteor_zh",
        "correctness_score",
        "faithfulness_score",
        "kfc_recall",
        "kfc_precision",
        "kfc_f1",
        "kfc_gt_fact_count",
        "kfc_gen_claim_count",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "ragas_faithfulness",
        "ragas_answer_correctness",
        # 圖譜品質指標（per-pipeline 常數，avg_ 前綴僅為格式統一）
        "gq_node_count",
        "gq_edge_count",
        "gq_density",
        "gq_avg_degree",
        "gq_orphan_node_ratio",
        "gq_largest_component_ratio",
        "gq_num_connected_components",
        "gq_avg_clustering_coefficient",
    ]

    # GraphQualityMetrics 原始欄名 → gq_ 前綴欄名
    _GQ_COL_MAP = {
        "node_count": "gq_node_count",
        "edge_count": "gq_edge_count",
        "density": "gq_density",
        "avg_degree": "gq_avg_degree",
        "orphan_node_ratio": "gq_orphan_node_ratio",
        "largest_component_ratio": "gq_largest_component_ratio",
        "num_connected_components": "gq_num_connected_components",
        "avg_clustering_coefficient": "gq_avg_clustering_coefficient",
    }

    _SCOPE_COLUMN_CANDIDATES = [
        "qa_scope",
        "question_type",
        "Question_Type",
        "Q_type",
        "hop-level",
        "Hop_Level",
        "hop_level",
    ]

    """
    評估報告生成器
    
    負責：
    - 儲存詳細評估結果（每個 Pipeline 的逐題結果）
    - 計算並儲存平均指標
    - 生成跨 Pipeline 的比較報告
    
    Attributes:
        base_dir: 報告儲存的根目錄
    """
    
    def __init__(self, base_dir: str):
        """
        初始化報告生成器
        
        Args:
            base_dir: 報告儲存的根目錄
        """
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
    
    def save_pipeline_results(
        self,
        pipeline_name: str,
        results: List[Dict[str, Any]],
        schema_info: Dict[str, Any] = None,
        graph_quality: Dict[str, Any] = None,
    ) -> str:
        """
        儲存單一 Pipeline 的詳細評估結果

        Args:
            pipeline_name: Pipeline 名稱
            results: 評估結果列表
            schema_info: Schema 資訊（可選）
            graph_quality: 圖譜品質指標 dict（可選），會另存 graph_quality.json
                           並在 detailed_results.csv 中補入常數欄

        Returns:
            儲存的 CSV 檔案路徑
        """
        if not results:
            print(f"  ⚠️ {pipeline_name} 沒有評估結果")
            return None
        
        # 建立 Pipeline 專屬資料夾
        pipeline_dir = os.path.join(self.base_dir, pipeline_name)
        os.makedirs(pipeline_dir, exist_ok=True)

        # 若有圖譜品質指標，另存 JSON
        if graph_quality:
            self.save_graph_quality(pipeline_dir, graph_quality)
        
        # 轉換為 DataFrame（排除 list 型 retrieval 欄，避免 CSV 序列化雜訊）
        _LIST_COLS = {"retrieved_entities", "retrieved_relations", "retrieved_contexts"}
        df = pd.DataFrame([
            {k: v for k, v in r.items() if k not in _LIST_COLS}
            for r in results
        ])
        # 補齊 token 欄位，確保 detailed_results 欄位一致
        for token_col in ["context_tokens", "entity_tokens", "relation_tokens", "chunk_tokens"]:
            if token_col not in df.columns:
                df[token_col] = 0

        # 將圖譜品質指標以常數欄寫入 detailed_results（每行相同值，便於後續 filter/sort）
        if graph_quality:
            for raw_col, gq_col in self._GQ_COL_MAP.items():
                df[gq_col] = graph_quality.get(raw_col, None)
        
        # 加入平均 row
        df = self._add_average_row(df)
        
        # 儲存為 CSV
        csv_path = os.path.join(pipeline_dir, "detailed_results.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # 若有 schema 資訊，額外輸出同層 JSON，方便後續追蹤實驗配置
        if schema_info:
            self.save_pipeline_schema(pipeline_dir, schema_info)
        
        print(f"  ✅ 已儲存明細報表至: {csv_path}")
        
        return csv_path

    @staticmethod
    def save_pipeline_schema(pipeline_dir: str, schema_info: Dict[str, Any]) -> str:
        """將 pipeline 使用的 schema 另存為 JSON。"""
        schema_payload = {
            "method": schema_info.get("method", ""),
            "entities": schema_info.get("entities", []),
            "relations": schema_info.get("relations", []),
            "validation_schema": schema_info.get("validation_schema", {}),
        }
        schema_path = os.path.join(pipeline_dir, "schema_info.json")
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema_payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"  🧩 已儲存 Schema 設定至: {schema_path}")
        return schema_path

    @staticmethod
    def save_graph_quality(pipeline_dir: str, graph_quality: Dict[str, Any]) -> str:
        """將圖譜品質指標另存為 JSON。"""
        gq_path = os.path.join(pipeline_dir, "graph_quality.json")
        with open(gq_path, "w", encoding="utf-8") as f:
            json.dump(graph_quality, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"  📊 已儲存圖譜品質指標至: {gq_path}")
        return gq_path

    def save_retrieval_content(
        self,
        pipeline_name: str,
        results: List[Dict[str, Any]],
    ) -> str:
        """
        將每題的 retrieval 內容（entities、relations、chunks）儲存為 JSONL。

        每行格式：
        {
            "idx": int,
            "question": str,
            "gt_answer": str,
            "generated_answer": str,
            "retrieved_entities": List[str],
            "retrieved_relations": List[str],
            "retrieved_chunks": List[str]
        }

        Args:
            pipeline_name: Pipeline 名稱
            results: 評估結果列表（由 evaluator.evaluate_pipeline 回傳）

        Returns:
            儲存的 JSONL 檔案路徑，若無 retrieval 欄位則回傳 None
        """
        if not results:
            return None

        pipeline_dir = os.path.join(self.base_dir, pipeline_name)
        os.makedirs(pipeline_dir, exist_ok=True)

        jsonl_path = os.path.join(pipeline_dir, "retrieval_content.jsonl")
        count = 0
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in results:
                # 跳過平均 row（idx 為字串時）
                idx = r.get("idx", "")
                if not isinstance(idx, (int, float)):
                    continue
                record = {
                    "idx": idx,
                    "question": r.get("query", r.get("question", "")),
                    "gt_answer": r.get("ground_truth_answer", r.get("gt_answer", "")),
                    "generated_answer": r.get("generated_answer", ""),
                    "retrieved_entities": r.get("retrieved_entities", []),
                    "retrieved_relations": r.get("retrieved_relations", []),
                    "retrieved_chunks": r.get("retrieved_contexts", []),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        print(f"  💾 已儲存 {count} 題的 retrieval 內容至: {jsonl_path}")
        return jsonl_path
    
    def _add_average_row(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        在 DataFrame 底部加入平均值 row
        
        Args:
            df: 原始 DataFrame
        
        Returns:
            加入平均值的 DataFrame
        """
        if df.empty:
            return df
        
        # 自動抓取所有數值欄位
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        
        # 排除 idx
        if 'idx' in numeric_cols:
            numeric_cols.remove('idx')
        
        def _build_avg_row(subset: pd.DataFrame, label: str, q_type_value: str = "") -> Dict[str, Any]:
            row = {col: subset[col].mean() for col in numeric_cols}
            row["idx"] = label
            for col in df.columns:
                if col not in row:
                    row[col] = ""
            if "Q_type" in df.columns:
                row["Q_type"] = q_type_value
            return row

        # 計算平均值
        avg_row = _build_avg_row(df, label="平均")
        rows_to_append = [avg_row]

        # 追加 SINGLE-HOP / MULTI-HOP 分群平均（依 Q_type）
        if "Q_type" in df.columns:
            qtype_series = df["Q_type"].fillna("").astype(str).str.strip().str.upper()

            single_subset = df[qtype_series == "SINGLE-HOP"]
            if not single_subset.empty:
                rows_to_append.append(
                    _build_avg_row(single_subset, label="平均_SINGLE-HOP", q_type_value="SINGLE-HOP")
                )

            multi_subset = df[qtype_series == "MULTI-HOP"]
            if not multi_subset.empty:
                rows_to_append.append(
                    _build_avg_row(multi_subset, label="平均_MULTI-HOP", q_type_value="MULTI-HOP")
                )

        # 合併
        df = pd.concat([df, pd.DataFrame(rows_to_append)], ignore_index=True)
        
        return df

    @staticmethod
    def _normalize_qa_scope(value: Any) -> str:
        """將題型欄位值標準化為 local/global。"""
        if value is None:
            return ""
        text = str(value).strip().lower()
        if not text or text == "nan":
            return ""

        normalized = text.replace("_", "-").replace(" ", "")
        local_aliases = {
            "local",
            "single-hop",
            "singlehop",
            "one-hop",
            "1-hop",
            "1hop",
            "hop1",
        }
        global_aliases = {
            "global",
            "multi-hop",
            "multihop",
            "two-hop",
            "2-hop",
            "2hop",
            "hop2",
        }
        if normalized in local_aliases:
            return "local"
        if normalized in global_aliases:
            return "global"
        if "single" in normalized or "local" in normalized or normalized.startswith("1-") or normalized.startswith("1hop"):
            return "local"
        if "multi" in normalized or "global" in normalized or normalized.startswith("2-") or normalized.startswith("2hop"):
            return "global"
        return ""

    @classmethod
    def _resolve_scope_series(cls, df: pd.DataFrame) -> Optional[pd.Series]:
        """從多種欄位推導 qa_scope 欄位（local/global）。"""
        for col in cls._SCOPE_COLUMN_CANDIDATES:
            if col not in df.columns:
                continue
            scope_series = df[col].map(cls._normalize_qa_scope)
            if (scope_series != "").any():
                return scope_series
        return None
    
    def extract_summary_from_df(
        self,
        pipeline_name: str,
        df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        從 DataFrame 提取平均指標作為總結
        
        Args:
            pipeline_name: Pipeline 名稱
            df: 包含評估結果的 DataFrame
        
        Returns:
            總結字典列表（包含 total/local/global）
        """
        if df.empty:
            return [{"pipeline_name": pipeline_name, "summary_scope": "total"}]

        working_df = df.copy()
        scope_series = self._resolve_scope_series(working_df)
        if scope_series is not None:
            working_df["qa_scope"] = scope_series

        # 以固定欄位輸出，避免不同方法產生不同 summary header
        def _build_summary_row(scope_name: str, sub_df: pd.DataFrame) -> Dict[str, Any]:
            row = {
                "pipeline_name": pipeline_name,
                "summary_scope": scope_name,
            }
            for col in self.SUMMARY_NUMERIC_COLUMNS:
                if col not in sub_df.columns:
                    row[f"avg_{col}"] = 0
                else:
                    row[f"avg_{col}"] = sub_df[col].mean()
            return row

        summary_rows: List[Dict[str, Any]] = []
        summary_rows.append(_build_summary_row("total", working_df))

        if "qa_scope" in working_df.columns:
            for scope in ["local", "global"]:
                sub_df = working_df[working_df["qa_scope"] == scope]
                if not sub_df.empty:
                    summary_rows.append(_build_summary_row(scope, sub_df))

        return summary_rows
    
    def generate_global_summary(
        self,
        summary_records: List[Dict[str, Any]]
    ) -> str:
        """
        生成跨 Pipeline 的全局比較報告
        
        Args:
            summary_records: 每個 Pipeline 的總結字典列表
        
        Returns:
            儲存的 CSV 檔案路徑
        """
        if not summary_records:
            print("  ⚠️ 沒有總結資料")
            return None
        
        # 轉換為 DataFrame
        df_summary = pd.DataFrame(summary_records)
        
        # 儲存為 CSV
        summary_csv_path = os.path.join(self.base_dir, "global_summary_report.csv")
        df_summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

        # 同步儲存為 XLSX
        summary_xlsx_path = os.path.join(self.base_dir, "global_summary_report.xlsx")
        df_summary.to_excel(summary_xlsx_path, index=False)
        
        print(f"\n📊 已儲存外層綜合評估報告至: {summary_csv_path}")
        print(f"📊 已儲存外層綜合評估報告至: {summary_xlsx_path}")
        
        # 顯示總結
        print("\n" + "="*50)
        print("實驗結果總覽 (Summary)")
        print("="*50)
        print(df_summary.to_string(index=False))
        
        return summary_csv_path

    @staticmethod
    def _normalize_cell_value(value: Any) -> Any:
        """將 dict/list 轉為 JSON 字串，避免寫入 Excel 後不可讀。"""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return value

    @staticmethod
    def write_run_config(
        run_dir: str,
        postfix: str = "",
        is_fast_test: bool = False,
        run_metadata: Dict[str, Any] = None,
        pipeline_metadata_map: Dict[str, Dict[str, Any]] = None,
    ) -> str:
        """在每次 run 目錄寫入可回溯設定檔。"""
        run_config_path = os.path.join(run_dir, "run_config.json")
        payload = {
            "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_dir": os.path.basename(run_dir.rstrip("/")),
            "postfix": postfix,
            "is_fast_test": is_fast_test,
            "run_metadata": run_metadata or {},
            "pipeline_metadata_map": pipeline_metadata_map or {},
        }
        with open(run_config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"🧾 已寫入 run 設定快照: {run_config_path}")
        return run_config_path

    @staticmethod
    def append_master_summary(
        summary_records: List[Dict[str, Any]],
        root_results_dir: str,
        run_dir: str,
        postfix: str = "",
        is_fast_test: bool = False,
        run_metadata: Dict[str, Any] = None,
        pipeline_metadata_map: Dict[str, Dict[str, Any]] = None,
    ) -> str:
        """將本次 summary 附加到 results/{exp|test}/{資料類別}/global_summary.xlsx。"""
        if not summary_records:
            return None

        os.makedirs(root_results_dir, exist_ok=True)
        master_xlsx_path = os.path.join(root_results_dir, "global_summary.xlsx")

        run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_dir_name = os.path.basename(run_dir.rstrip("/"))

        enriched_records = []
        for record in summary_records:
            row = dict(record)
            row["run_timestamp"] = run_timestamp
            row["run_dir"] = run_dir_name
            row["postfix"] = postfix
            row["is_fast_test"] = is_fast_test
            if run_metadata:
                for key, value in run_metadata.items():
                    row[key] = EvaluationReporter._normalize_cell_value(value)
            if pipeline_metadata_map:
                pipeline_name = row.get("pipeline_name")
                pipeline_meta = pipeline_metadata_map.get(pipeline_name, {})
                for key, value in pipeline_meta.items():
                    row[key] = EvaluationReporter._normalize_cell_value(value)
            enriched_records.append(row)

        current_df = pd.DataFrame(enriched_records)
        if os.path.exists(master_xlsx_path):
            previous_df = pd.read_excel(master_xlsx_path)
            current_df = pd.concat([previous_df, current_df], ignore_index=True)

        current_df.to_excel(master_xlsx_path, index=False)
        print(f"📚 已更新累積總表: {master_xlsx_path}")
        return master_xlsx_path


async def run_evaluation(
    qa_datasets: List[Dict],
    pipelines: List[Any],
    postfix: str = "",
    results_root_dir: str = "/home/End_to_End_RAG/results",
    is_fast_test: bool = False,
    run_metadata: Dict[str, Any] = None,
    pipeline_metadata_map: Dict[str, Dict[str, Any]] = None,
) -> None:
    """
    執行完整的評估流程（向後兼容函數）
    
    Args:
        qa_datasets: QA 資料集列表
        pipelines: Pipeline Wrapper 列表
        postfix: 結果資料夾名稱後綴
    """
    from datetime import datetime
    from llama_index.core import Settings as LlamaSettings
    from src.evaluation.evaluator import RAGEvaluator
    
    # 建立評估器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_eval_dir = os.path.join(results_root_dir, f"evaluation_results_{timestamp}{postfix}")
    
    eval_metrics_mode = (run_metadata or {}).get("eval_metrics_mode", "kfc_only")
    evaluator = RAGEvaluator(
        eval_llm=LlamaSettings.eval_llm,
        base_eval_dir=base_eval_dir,
        metrics_mode=eval_metrics_mode,
    )
    print(f"🧪 評估模式: {evaluator.metrics_mode}")
    reporter = EvaluationReporter(base_dir=base_eval_dir)
    
    print(f"📁 建立實驗數據主資料夾: {base_eval_dir}")
    EvaluationReporter.write_run_config(
        run_dir=base_eval_dir,
        postfix=postfix,
        is_fast_test=is_fast_test,
        run_metadata=run_metadata,
        pipeline_metadata_map=pipeline_metadata_map,
    )
    
    summary_records = []
    
    for pipeline in pipelines:
        # 評估 Pipeline
        results = await evaluator.evaluate_pipeline(pipeline, qa_datasets)
        
        # 儲存詳細結果
        if results:
            gq = getattr(pipeline, "_graph_quality", None)
            reporter.save_pipeline_results(
                pipeline.name,
                results,
                schema_info=getattr(pipeline, "schema_info", None),
                graph_quality=gq,
            )
            reporter.save_retrieval_content(pipeline.name, results)

            # 提取總結（先補入 gq_ 欄位，使 extract_summary_from_df 可計算 avg_gq_*）
            df = pd.DataFrame(results)
            if gq:
                for raw_col, gq_col in EvaluationReporter._GQ_COL_MAP.items():
                    df[gq_col] = gq.get(raw_col, None)
            summaries = reporter.extract_summary_from_df(pipeline.name, df)
            summary_records.extend(summaries)
    
    # 生成全局比較報告
    reporter.generate_global_summary(summary_records)
    EvaluationReporter.append_master_summary(
        summary_records=summary_records,
        root_results_dir=results_root_dir,
        run_dir=base_eval_dir,
        postfix=postfix,
        is_fast_test=is_fast_test,
        run_metadata=run_metadata,
        pipeline_metadata_map=pipeline_metadata_map,
    )


async def run_evaluation_with_token_budget(
    qa_datasets: List[Dict],
    pipelines: List[Any],
    postfix: str = "",
    baseline_method: str = "vector_hybrid",
    results_root_dir: str = "/home/End_to_End_RAG/results",
    is_fast_test: bool = False,
    run_metadata: Dict[str, Any] = None,
    pipeline_metadata_map: Dict[str, Dict[str, Any]] = None,
) -> None:
    """
    執行包含token budget控制的兩階段評估流程
    
    階段1: 評估baseline方法（通常是Vector RAG），計算平均token使用量
    階段2: 根據baseline調整其他方法的參數後進行評估
    
    Args:
        qa_datasets: QA 資料集列表
        pipelines: Pipeline Wrapper 列表
        postfix: 結果資料夾名稱後綴
        baseline_method: 用作baseline的方法名稱關鍵字
    """
    from datetime import datetime
    from llama_index.core import Settings as LlamaSettings
    from src.evaluation.evaluator import RAGEvaluator
    from src.rag.token_budget_controller import TokenBudgetController
    from src.evaluation.token_analysis import analyze_evaluation_results
    
    # 建立評估器和controller
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_eval_dir = os.path.join(
        results_root_dir,
        f"evaluation_results_{timestamp}{postfix}_token_budget"
    )
    
    eval_metrics_mode = (run_metadata or {}).get("eval_metrics_mode", "kfc_only")
    evaluator = RAGEvaluator(
        eval_llm=LlamaSettings.eval_llm,
        base_eval_dir=base_eval_dir,
        metrics_mode=eval_metrics_mode,
    )
    print(f"🧪 評估模式: {evaluator.metrics_mode}")
    reporter = EvaluationReporter(base_dir=base_eval_dir)
    token_controller = TokenBudgetController()
    
    print(f"📁 建立實驗數據主資料夾: {base_eval_dir}")
    EvaluationReporter.write_run_config(
        run_dir=base_eval_dir,
        postfix=postfix,
        is_fast_test=is_fast_test,
        run_metadata=run_metadata,
        pipeline_metadata_map=pipeline_metadata_map,
    )
    print("\n" + "="*80)
    print("🎯 兩階段Token Budget評估流程")
    print("="*80)
    
    # 階段1: 評估baseline方法
    print(f"\n📊 階段1: 評估Baseline方法（尋找包含 '{baseline_method}' 的pipeline）")
    print("-"*80)
    
    baseline_pipeline = None
    other_pipelines = []
    
    for pipeline in pipelines:
        if baseline_method.lower() in pipeline.name.lower():
            baseline_pipeline = pipeline
        else:
            other_pipelines.append(pipeline)
    
    if baseline_pipeline is None:
        print(f"⚠️  警告: 找不到包含 '{baseline_method}' 的pipeline，使用第一個pipeline作為baseline")
        if pipelines:
            baseline_pipeline = pipelines[0]
            other_pipelines = pipelines[1:]
    
    summary_records = []
    
    if baseline_pipeline:
        print(f"✅ Baseline方法: {baseline_pipeline.name}")
        
        # 評估baseline
        results = await evaluator.evaluate_pipeline(baseline_pipeline, qa_datasets)
        
        if results:
            # 儲存結果
            gq_baseline = getattr(baseline_pipeline, "_graph_quality", None)
            reporter.save_pipeline_results(
                baseline_pipeline.name,
                results,
                schema_info=getattr(baseline_pipeline, "schema_info", None),
                graph_quality=gq_baseline,
            )
            reporter.save_retrieval_content(baseline_pipeline.name, results)
            df = pd.DataFrame(results)
            if gq_baseline:
                for raw_col, gq_col in EvaluationReporter._GQ_COL_MAP.items():
                    df[gq_col] = gq_baseline.get(raw_col, None)
            summaries = reporter.extract_summary_from_df(baseline_pipeline.name, df)
            summary_records.extend(summaries)
            
            # 提取token資訊
            if "context_tokens" in df.columns:
                tokens_per_query = df["context_tokens"].dropna().tolist()
                
                if tokens_per_query:
                    # 設定baseline
                    token_controller.set_baseline(baseline_pipeline.name, tokens_per_query)
                    
                    print(f"\n✅ Baseline Token統計:")
                    print(f"   平均: {token_controller.baseline_tokens} tokens")
                    print(f"   目標範圍 (含10%緩衝): ≤ {token_controller.get_target_tokens()} tokens")
                else:
                    print("⚠️  Baseline結果中沒有token資料")
            else:
                print("⚠️  Baseline結果中沒有context_tokens欄位")
    
    # 階段2: 評估其他方法
    if other_pipelines and token_controller.baseline_tokens:
        print("\n" + "="*80)
        print(f"📊 階段2: 評估其他方法（共 {len(other_pipelines)} 個）")
        print("-"*80)
        
        # 輸出建議參數
        print("\n💡 LightRAG建議參數 (基於baseline):")
        for mode in ["hybrid", "local", "global", "mix", "naive"]:
            params = token_controller.adjust_lightrag_params(mode=mode)
            print(f"   {mode}: max_total={params['max_total_tokens']}, "
                  f"entity={params['max_entity_tokens']}, "
                  f"relation={params['max_relation_tokens']}, "
                  f"chunk_top_k={params['chunk_top_k']}")

        print("\n✅ 將自動套用 token budget 到支援的 pipeline")
        print("-"*80 + "\n")
        
        for pipeline in other_pipelines:
            if hasattr(pipeline, "apply_token_budget"):
                mode = getattr(pipeline, "mode", "hybrid")
                params = token_controller.adjust_lightrag_params(mode=mode)
                pipeline.apply_token_budget(params)

            # 評估pipeline
            results = await evaluator.evaluate_pipeline(pipeline, qa_datasets)
            
            if results:
                # 儲存結果
                gq = getattr(pipeline, "_graph_quality", None)
                reporter.save_pipeline_results(
                    pipeline.name,
                    results,
                    schema_info=getattr(pipeline, "schema_info", None),
                    graph_quality=gq,
                )
                reporter.save_retrieval_content(pipeline.name, results)
                df = pd.DataFrame(results)
                if gq:
                    for raw_col, gq_col in EvaluationReporter._GQ_COL_MAP.items():
                        df[gq_col] = gq.get(raw_col, None)
                summaries = reporter.extract_summary_from_df(pipeline.name, df)
                summary_records.extend(summaries)
                
                # 添加token統計
                if "context_tokens" in df.columns:
                    tokens_per_query = df["context_tokens"].dropna().tolist()
                    if tokens_per_query:
                        token_controller.add_method_stats(pipeline.name, tokens_per_query)
    
    # 生成全局比較報告
    reporter.generate_global_summary(summary_records)
    EvaluationReporter.append_master_summary(
        summary_records=summary_records,
        root_results_dir=results_root_dir,
        run_dir=base_eval_dir,
        postfix=f"{postfix}_token_budget",
        is_fast_test=is_fast_test,
        run_metadata=run_metadata,
        pipeline_metadata_map=pipeline_metadata_map,
    )
    
    # 生成token分析報告
    if token_controller.baseline_tokens:
        print("\n" + "="*80)
        print("📈 Token使用分析")
        print("="*80)
        
        # 儲存token controller統計
        token_stats_path = os.path.join(base_eval_dir, "token_budget_stats.json")
        token_controller.save_stats(token_stats_path)
        
        # 生成報告
        token_report = token_controller.generate_report()
        print(token_report)
        
        # 儲存報告文本
        token_report_path = os.path.join(base_eval_dir, "token_budget_report.txt")
        with open(token_report_path, "w", encoding="utf-8") as f:
            f.write(token_report)
        print(f"💾 Token budget報告已儲存: {token_report_path}")
        
        # 使用token_analysis.py生成詳細分析
        try:
            analyze_evaluation_results(base_eval_dir, baseline_method=baseline_pipeline.name if baseline_pipeline else None)
        except Exception as e:
            print(f"⚠️  生成詳細token分析時發生錯誤: {e}")
    
    print("\n✅ 兩階段評估完成！")
    print(f"📁 結果目錄: {base_eval_dir}")
