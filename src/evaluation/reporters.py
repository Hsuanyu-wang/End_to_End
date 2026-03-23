"""
評估報告生成器

負責生成與儲存評估報告
"""

import os
from typing import List, Dict, Any
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
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "ragas_faithfulness",
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
        results: List[Dict[str, Any]]
    ) -> str:
        """
        儲存單一 Pipeline 的詳細評估結果
        
        Args:
            pipeline_name: Pipeline 名稱
            results: 評估結果列表
        
        Returns:
            儲存的 CSV 檔案路徑
        """
        if not results:
            print(f"  ⚠️ {pipeline_name} 沒有評估結果")
            return None
        
        # 建立 Pipeline 專屬資料夾
        pipeline_dir = os.path.join(self.base_dir, pipeline_name)
        os.makedirs(pipeline_dir, exist_ok=True)
        
        # 轉換為 DataFrame
        df = pd.DataFrame(results)
        # 補齊 token 欄位，確保 detailed_results 欄位一致
        for token_col in ["context_tokens", "entity_tokens", "relation_tokens", "chunk_tokens"]:
            if token_col not in df.columns:
                df[token_col] = 0
        
        # 加入平均 row
        df = self._add_average_row(df)
        
        # 儲存為 CSV
        csv_path = os.path.join(pipeline_dir, "detailed_results.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        
        print(f"  ✅ 已儲存明細報表至: {csv_path}")
        
        return csv_path
    
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
        
        # 計算平均值
        avg_row = {col: df[col].mean() for col in numeric_cols}
        avg_row["idx"] = "平均"
        
        # 填補文字類型的欄位為空字串（schema 欄位在平均行保持空白）
        for col in df.columns:
            if col not in avg_row:
                avg_row[col] = ""
        
        # 合併
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
        
        return df
    
    def extract_summary_from_df(
        self,
        pipeline_name: str,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        從 DataFrame 提取平均指標作為總結
        
        Args:
            pipeline_name: Pipeline 名稱
            df: 包含評估結果的 DataFrame
        
        Returns:
            總結字典
        """
        if df.empty:
            return {"pipeline_name": pipeline_name}
        
        # 以固定欄位輸出，避免不同方法產生不同 summary header
        summary = {"pipeline_name": pipeline_name}
        for col in self.SUMMARY_NUMERIC_COLUMNS:
            if col not in df.columns:
                df[col] = 0
            summary[f"avg_{col}"] = df[col].mean()
        
        return summary
    
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
    def append_master_summary(
        summary_records: List[Dict[str, Any]],
        root_results_dir: str,
        run_dir: str,
        postfix: str = "",
        is_fast_test: bool = False,
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
    
    evaluator = RAGEvaluator(eval_llm=LlamaSettings.eval_llm, base_eval_dir=base_eval_dir)
    reporter = EvaluationReporter(base_dir=base_eval_dir)
    
    print(f"📁 建立實驗數據主資料夾: {base_eval_dir}")
    
    summary_records = []
    
    for pipeline in pipelines:
        # 評估 Pipeline
        results = await evaluator.evaluate_pipeline(pipeline, qa_datasets)
        
        # 儲存詳細結果
        if results:
            reporter.save_pipeline_results(pipeline.name, results)
            
            # 提取總結
            df = pd.DataFrame(results)
            summary = reporter.extract_summary_from_df(pipeline.name, df)
            
            # 合併圖譜品質指標（若有）
            gq = getattr(pipeline, "_graph_quality", None)
            if gq:
                from src.evaluation.metrics.graph_quality import GraphQualityMetrics
                for col in GraphQualityMetrics.summary_columns():
                    if col in gq:
                        summary[col] = gq[col]
            
            summary_records.append(summary)
    
    # 生成全局比較報告
    reporter.generate_global_summary(summary_records)
    EvaluationReporter.append_master_summary(
        summary_records=summary_records,
        root_results_dir=results_root_dir,
        run_dir=base_eval_dir,
        postfix=postfix,
        is_fast_test=is_fast_test,
    )


async def run_evaluation_with_token_budget(
    qa_datasets: List[Dict],
    pipelines: List[Any],
    postfix: str = "",
    baseline_method: str = "vector_hybrid",
    results_root_dir: str = "/home/End_to_End_RAG/results",
    is_fast_test: bool = False,
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
    
    evaluator = RAGEvaluator(eval_llm=LlamaSettings.eval_llm, base_eval_dir=base_eval_dir)
    reporter = EvaluationReporter(base_dir=base_eval_dir)
    token_controller = TokenBudgetController()
    
    print(f"📁 建立實驗數據主資料夾: {base_eval_dir}")
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
            reporter.save_pipeline_results(baseline_pipeline.name, results)
            df = pd.DataFrame(results)
            summary = reporter.extract_summary_from_df(baseline_pipeline.name, df)
            summary_records.append(summary)
            
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
                reporter.save_pipeline_results(pipeline.name, results)
                df = pd.DataFrame(results)
                summary = reporter.extract_summary_from_df(pipeline.name, df)
                summary_records.append(summary)
                
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
