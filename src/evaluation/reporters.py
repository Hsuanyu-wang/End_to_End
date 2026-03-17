"""
評估報告生成器

負責生成與儲存評估報告
"""

import os
from typing import List, Dict, Any
import pandas as pd


class EvaluationReporter:
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
        
        # 填補文字類型的欄位為空字串
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
        
        # 取得所有數值欄位
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        
        # 排除 idx
        if 'idx' in numeric_cols:
            numeric_cols.remove('idx')
        
        # 計算平均值
        summary = {"pipeline_name": pipeline_name}
        for col in numeric_cols:
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
        
        print(f"\n📊 已儲存外層綜合評估報告至: {summary_csv_path}")
        
        # 顯示總結
        print("\n" + "="*50)
        print("實驗結果總覽 (Summary)")
        print("="*50)
        print(df_summary.to_string(index=False))
        
        return summary_csv_path


async def run_evaluation(
    qa_datasets: List[Dict],
    pipelines: List[Any],
    postfix: str = ""
) -> None:
    """
    執行完整的評估流程（向後兼容函數）
    
    Args:
        qa_datasets: QA 資料集列表
        pipelines: Pipeline Wrapper 列表
        postfix: 結果資料夾名稱後綴
    """
    from datetime import datetime
    from llama_index.core import Settings
    from src.evaluation.evaluator import RAGEvaluator
    
    # 建立評估器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_eval_dir = f"/home/End_to_End_RAG/results/evaluation_results_{timestamp}{postfix}"
    
    evaluator = RAGEvaluator(eval_llm=Settings.eval_llm, base_eval_dir=base_eval_dir)
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
            summary_records.append(summary)
    
    # 生成全局比較報告
    reporter.generate_global_summary(summary_records)
