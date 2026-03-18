"""
Token使用分析工具

提供token使用的統計分析和視覺化功能
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime


class TokenAnalyzer:
    """
    Token使用分析器
    
    從評估結果中分析token使用情況，生成統計報告
    
    Attributes:
        results_df: 評估結果DataFrame
        method_name: 方法名稱
    """
    
    def __init__(self, results_csv_path: str = None, results_df: pd.DataFrame = None):
        """
        初始化TokenAnalyzer
        
        Args:
            results_csv_path: 評估結果CSV路徑
            results_df: 評估結果DataFrame（如果已載入）
        """
        if results_df is not None:
            self.results_df = results_df
        elif results_csv_path and os.path.exists(results_csv_path):
            self.results_df = pd.read_csv(results_csv_path)
        else:
            self.results_df = None
        
        self.method_name = self._extract_method_name(results_csv_path) if results_csv_path else "Unknown"
    
    def _extract_method_name(self, csv_path: str) -> str:
        """從CSV路徑提取方法名稱"""
        basename = os.path.basename(os.path.dirname(csv_path))
        return basename if basename else "Unknown"
    
    def get_token_stats(self) -> Dict[str, Any]:
        """
        獲取token使用統計
        
        Returns:
            統計資訊字典
        """
        if self.results_df is None or "context_tokens" not in self.results_df.columns:
            return {}
        
        tokens = self.results_df["context_tokens"].dropna()
        
        if len(tokens) == 0:
            return {}
        
        stats = {
            "method": self.method_name,
            "total_queries": len(tokens),
            "avg_tokens": float(tokens.mean()),
            "median_tokens": float(tokens.median()),
            "min_tokens": int(tokens.min()),
            "max_tokens": int(tokens.max()),
            "std_tokens": float(tokens.std()),
            "total_tokens_consumed": int(tokens.sum()),
            "percentile_25": float(tokens.quantile(0.25)),
            "percentile_75": float(tokens.quantile(0.75)),
        }
        
        # LightRAG詳細統計
        if "entity_tokens" in self.results_df.columns:
            entity_tokens = self.results_df["entity_tokens"].dropna()
            relation_tokens = self.results_df["relation_tokens"].dropna()
            chunk_tokens = self.results_df["chunk_tokens"].dropna()
            
            if len(entity_tokens) > 0:
                stats["avg_entity_tokens"] = float(entity_tokens.mean())
                stats["avg_relation_tokens"] = float(relation_tokens.mean())
                stats["avg_chunk_tokens"] = float(chunk_tokens.mean())
                
                total_avg = stats["avg_tokens"]
                if total_avg > 0:
                    stats["entity_ratio"] = stats["avg_entity_tokens"] / total_avg
                    stats["relation_ratio"] = stats["avg_relation_tokens"] / total_avg
                    stats["chunk_ratio"] = stats["avg_chunk_tokens"] / total_avg
        
        return stats
    
    def generate_token_summary(self) -> str:
        """
        生成token使用摘要文本
        
        Returns:
            摘要文本
        """
        stats = self.get_token_stats()
        
        if not stats:
            return "無token統計資料"
        
        lines = []
        lines.append(f"\n{'='*70}")
        lines.append(f"Token使用分析 - {stats['method']}")
        lines.append(f"{'='*70}")
        lines.append(f"\n基本統計:")
        lines.append(f"  查詢數量: {stats['total_queries']}")
        lines.append(f"  平均Tokens: {stats['avg_tokens']:.1f}")
        lines.append(f"  中位數: {stats['median_tokens']:.1f}")
        lines.append(f"  範圍: [{stats['min_tokens']}, {stats['max_tokens']}]")
        lines.append(f"  標準差: {stats['std_tokens']:.1f}")
        lines.append(f"  總消耗: {stats['total_tokens_consumed']:,} tokens")
        
        # LightRAG詳細資訊
        if "avg_entity_tokens" in stats:
            lines.append(f"\nLightRAG組件分布:")
            lines.append(f"  Entity平均: {stats['avg_entity_tokens']:.1f} ({stats['entity_ratio']*100:.1f}%)")
            lines.append(f"  Relation平均: {stats['avg_relation_tokens']:.1f} ({stats['relation_ratio']*100:.1f}%)")
            lines.append(f"  Chunk平均: {stats['avg_chunk_tokens']:.1f} ({stats['chunk_ratio']*100:.1f}%)")
        
        lines.append(f"{'='*70}\n")
        
        return "\n".join(lines)
    
    def compare_with_baseline(self, baseline_tokens: int) -> Dict[str, Any]:
        """
        與baseline比較
        
        Args:
            baseline_tokens: baseline的平均token數
        
        Returns:
            比較結果
        """
        stats = self.get_token_stats()
        
        if not stats:
            return {}
        
        avg_tokens = stats["avg_tokens"]
        diff = avg_tokens - baseline_tokens
        ratio = (avg_tokens / baseline_tokens) * 100 if baseline_tokens > 0 else 0
        
        return {
            "method": stats["method"],
            "avg_tokens": avg_tokens,
            "baseline_tokens": baseline_tokens,
            "difference": diff,
            "utilization_pct": ratio,
            "within_budget_10pct": abs(diff) <= baseline_tokens * 0.1,
            "within_budget_20pct": abs(diff) <= baseline_tokens * 0.2,
        }


class TokenReportGenerator:
    """
    Token分析報告生成器
    
    從多個評估結果生成綜合分析報告
    """
    
    def __init__(self, eval_results_dir: str):
        """
        初始化報告生成器
        
        Args:
            eval_results_dir: 評估結果目錄
        """
        self.eval_results_dir = eval_results_dir
        self.analyzers: Dict[str, TokenAnalyzer] = {}
        self.baseline_method: Optional[str] = None
        self.baseline_tokens: Optional[int] = None
        
        self._load_results()
    
    def _load_results(self):
        """載入所有評估結果"""
        if not os.path.exists(self.eval_results_dir):
            print(f"⚠️  評估目錄不存在: {self.eval_results_dir}")
            return
        
        for method_dir in os.listdir(self.eval_results_dir):
            method_path = os.path.join(self.eval_results_dir, method_dir)
            
            if not os.path.isdir(method_path):
                continue
            
            csv_path = os.path.join(method_path, "detailed_results.csv")
            
            if os.path.exists(csv_path):
                try:
                    analyzer = TokenAnalyzer(results_csv_path=csv_path)
                    self.analyzers[method_dir] = analyzer
                    print(f"✅ 載入: {method_dir}")
                except Exception as e:
                    print(f"⚠️  載入失敗 {method_dir}: {e}")
    
    def set_baseline(self, method_name: str):
        """
        設定baseline方法
        
        Args:
            method_name: baseline方法名稱
        """
        if method_name not in self.analyzers:
            print(f"⚠️  方法 {method_name} 不存在")
            return
        
        stats = self.analyzers[method_name].get_token_stats()
        if stats:
            self.baseline_method = method_name
            self.baseline_tokens = int(stats["avg_tokens"])
            print(f"✅ 設定Baseline: {method_name} (平均 {self.baseline_tokens} tokens)")
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """
        生成比較表格
        
        Returns:
            比較結果DataFrame
        """
        rows = []
        
        for method_name, analyzer in self.analyzers.items():
            stats = analyzer.get_token_stats()
            
            if not stats:
                continue
            
            row = {
                "Method": method_name,
                "Avg Tokens": stats["avg_tokens"],
                "Min": stats["min_tokens"],
                "Max": stats["max_tokens"],
                "Std Dev": stats["std_tokens"],
                "Total Consumed": stats["total_tokens_consumed"],
            }
            
            if self.baseline_tokens:
                utilization = (stats["avg_tokens"] / self.baseline_tokens) * 100
                row["Utilization (%)"] = utilization
                row["Diff from Baseline"] = stats["avg_tokens"] - self.baseline_tokens
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if not df.empty and "Avg Tokens" in df.columns:
            df = df.sort_values("Avg Tokens", ascending=False)
        
        return df
    
    def generate_report(self, output_path: str = None) -> str:
        """
        生成完整報告
        
        Args:
            output_path: 輸出檔案路徑（可選）
        
        Returns:
            報告文本
        """
        lines = []
        lines.append("\n" + "="*80)
        lines.append("Token Budget公平性分析報告")
        lines.append(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("="*80)
        
        if self.baseline_method and self.baseline_tokens:
            lines.append(f"\nBaseline方法: {self.baseline_method}")
            lines.append(f"Baseline平均Tokens: {self.baseline_tokens}")
        
        lines.append("\n" + "-"*80)
        lines.append("各方法Token使用比較")
        lines.append("-"*80)
        
        # 生成表格
        df = self.generate_comparison_table()
        
        if not df.empty:
            lines.append("\n" + df.to_string(index=False))
        else:
            lines.append("\n無可用資料")
        
        # 詳細分析
        lines.append("\n" + "-"*80)
        lines.append("詳細分析")
        lines.append("-"*80)
        
        for method_name, analyzer in sorted(self.analyzers.items()):
            summary = analyzer.generate_token_summary()
            lines.append(summary)
        
        lines.append("="*80 + "\n")
        
        report = "\n".join(lines)
        
        # 儲存報告
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"💾 報告已儲存: {output_path}")
        
        return report
    
    def save_comparison_csv(self, output_path: str):
        """
        儲存比較表格為CSV
        
        Args:
            output_path: 輸出CSV路徑
        """
        df = self.generate_comparison_table()
        
        if not df.empty:
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"💾 比較表已儲存: {output_path}")
        else:
            print("⚠️  沒有資料可儲存")


def analyze_evaluation_results(eval_results_dir: str, baseline_method: str = None):
    """
    分析評估結果並生成報告（便捷函數）
    
    Args:
        eval_results_dir: 評估結果目錄
        baseline_method: baseline方法名稱（可選）
    """
    generator = TokenReportGenerator(eval_results_dir)
    
    # 設定baseline
    if baseline_method and baseline_method in generator.analyzers:
        generator.set_baseline(baseline_method)
    elif generator.analyzers:
        # 自動選擇第一個包含"Vector"或"Hybrid"的方法作為baseline
        for method in generator.analyzers.keys():
            if "Vector" in method or "Hybrid" in method:
                generator.set_baseline(method)
                break
    
    # 生成報告
    report_path = os.path.join(eval_results_dir, "token_analysis_report.txt")
    csv_path = os.path.join(eval_results_dir, "token_comparison.csv")
    
    report = generator.generate_report(output_path=report_path)
    generator.save_comparison_csv(output_path=csv_path)
    
    print(report)
    
    return generator
