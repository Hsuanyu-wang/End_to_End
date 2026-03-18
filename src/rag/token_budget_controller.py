"""
Token Budget控制器

提供動態token budget控制功能，確保不同RAG方法在相同token預算下比較
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class TokenBudgetStats:
    """Token預算統計資訊"""
    method_name: str
    avg_tokens: float
    min_tokens: int
    max_tokens: int
    std_tokens: float
    sample_count: int
    tokens_per_query: List[int]


class TokenBudgetController:
    """
    Token Budget控制器
    
    負責：
    1. 計算baseline method的token使用量
    2. 根據baseline動態調整其他方法的參數
    3. 驗證調整後的token使用是否符合預算
    
    Attributes:
        baseline_tokens: 基準token數量
        buffer_ratio: 緩衝比例（允許超出baseline的百分比）
        stats: 各方法的token統計
    """
    
    def __init__(self, baseline_tokens: Optional[int] = None, buffer_ratio: float = 0.1):
        """
        初始化Token Budget控制器
        
        Args:
            baseline_tokens: 基準token數量（如果為None，則需要先計算）
            buffer_ratio: 緩衝比例，預設10%
        """
        self.baseline_tokens = baseline_tokens
        self.buffer_ratio = buffer_ratio
        self.stats: Dict[str, TokenBudgetStats] = {}
    
    def set_baseline(self, method_name: str, tokens_per_query: List[int]):
        """
        設定baseline method的token使用量
        
        Args:
            method_name: 方法名稱
            tokens_per_query: 每個查詢的token數列表
        """
        if not tokens_per_query:
            raise ValueError("tokens_per_query不能為空")
        
        avg_tokens = np.mean(tokens_per_query)
        self.baseline_tokens = int(avg_tokens)
        
        stats = TokenBudgetStats(
            method_name=method_name,
            avg_tokens=avg_tokens,
            min_tokens=int(np.min(tokens_per_query)),
            max_tokens=int(np.max(tokens_per_query)),
            std_tokens=float(np.std(tokens_per_query)),
            sample_count=len(tokens_per_query),
            tokens_per_query=tokens_per_query
        )
        
        self.stats[method_name] = stats
        
        print(f"✅ 設定Baseline: {method_name}")
        print(f"   平均Tokens: {avg_tokens:.2f}")
        print(f"   範圍: [{stats.min_tokens}, {stats.max_tokens}]")
        print(f"   標準差: {stats.std_tokens:.2f}")
    
    def get_target_tokens(self) -> int:
        """
        獲取目標token數（考慮緩衝）
        
        Returns:
            目標token數
        """
        if self.baseline_tokens is None:
            raise ValueError("尚未設定baseline，請先呼叫set_baseline")
        
        return int(self.baseline_tokens * (1 + self.buffer_ratio))
    
    def adjust_lightrag_params(
        self, 
        current_params: Optional[Dict[str, int]] = None,
        mode: str = "hybrid"
    ) -> Dict[str, int]:
        """
        根據baseline調整LightRAG的token參數
        
        Args:
            current_params: 當前參數（可選）
            mode: LightRAG模式
        
        Returns:
            調整後的參數字典
        """
        target_tokens = self.get_target_tokens()
        
        # 根據模式分配token預算
        if mode in ["local", "global"]:
            # Local/Global模式主要使用KG資訊
            params = {
                "max_entity_tokens": int(target_tokens * 0.35),
                "max_relation_tokens": int(target_tokens * 0.45),
                "max_total_tokens": target_tokens,
            }
        elif mode == "hybrid":
            # Hybrid模式平衡KG和文本
            params = {
                "max_entity_tokens": int(target_tokens * 0.25),
                "max_relation_tokens": int(target_tokens * 0.35),
                "max_total_tokens": target_tokens,
            }
        elif mode in ["mix", "naive"]:
            # Mix/Naive模式主要使用文本
            params = {
                "max_entity_tokens": int(target_tokens * 0.15),
                "max_relation_tokens": int(target_tokens * 0.25),
                "max_total_tokens": target_tokens,
            }
        else:
            # Bypass模式不需要context
            params = {
                "max_entity_tokens": 0,
                "max_relation_tokens": 0,
                "max_total_tokens": target_tokens,
            }
        
        # chunk_top_k估算（假設每個chunk約200 tokens）
        available_chunk_tokens = target_tokens - params["max_entity_tokens"] - params["max_relation_tokens"]
        params["chunk_top_k"] = max(1, min(20, available_chunk_tokens // 200))
        
        return params
    
    def adjust_vector_top_k(
        self, 
        avg_tokens_per_chunk: int = 500,
        current_top_k: int = 2
    ) -> int:
        """
        根據baseline調整Vector RAG的top_k
        
        Args:
            avg_tokens_per_chunk: 每個chunk的平均token數
            current_top_k: 當前top_k值
        
        Returns:
            調整後的top_k
        """
        target_tokens = self.get_target_tokens()
        
        # 計算建議的top_k
        suggested_top_k = max(1, target_tokens // avg_tokens_per_chunk)
        
        return suggested_top_k
    
    def add_method_stats(self, method_name: str, tokens_per_query: List[int]):
        """
        添加方法的token統計
        
        Args:
            method_name: 方法名稱
            tokens_per_query: 每個查詢的token數列表
        """
        if not tokens_per_query:
            print(f"⚠️  {method_name} 沒有token資料")
            return
        
        stats = TokenBudgetStats(
            method_name=method_name,
            avg_tokens=float(np.mean(tokens_per_query)),
            min_tokens=int(np.min(tokens_per_query)),
            max_tokens=int(np.max(tokens_per_query)),
            std_tokens=float(np.std(tokens_per_query)),
            sample_count=len(tokens_per_query),
            tokens_per_query=tokens_per_query
        )
        
        self.stats[method_name] = stats
    
    def calculate_utilization(self, method_name: str) -> float:
        """
        計算方法的token利用率（相對於baseline）
        
        Args:
            method_name: 方法名稱
        
        Returns:
            利用率（百分比）
        """
        if method_name not in self.stats or self.baseline_tokens is None:
            return 0.0
        
        stats = self.stats[method_name]
        return (stats.avg_tokens / self.baseline_tokens) * 100
    
    def is_within_budget(self, method_name: str, strict: bool = False) -> bool:
        """
        檢查方法是否在預算內
        
        Args:
            method_name: 方法名稱
            strict: 是否嚴格檢查（不允許緩衝）
        
        Returns:
            是否在預算內
        """
        if method_name not in self.stats or self.baseline_tokens is None:
            return False
        
        stats = self.stats[method_name]
        threshold = self.baseline_tokens if strict else self.get_target_tokens()
        
        return stats.avg_tokens <= threshold
    
    def generate_report(self) -> str:
        """
        生成token使用報告
        
        Returns:
            報告文本
        """
        if not self.stats:
            return "沒有統計資料"
        
        lines = []
        lines.append("\n" + "="*80)
        lines.append("Token Budget分析報告")
        lines.append("="*80)
        
        if self.baseline_tokens:
            target = self.get_target_tokens()
            lines.append(f"\nBaseline Tokens: {self.baseline_tokens}")
            lines.append(f"Target Tokens (含{self.buffer_ratio*100:.0f}%緩衝): {target}")
        
        lines.append("\n" + "-"*80)
        lines.append(f"{'方法':<25} {'平均Tokens':>12} {'範圍':>20} {'標準差':>10} {'利用率':>8}")
        lines.append("-"*80)
        
        for method_name, stats in sorted(self.stats.items()):
            utilization = self.calculate_utilization(method_name)
            range_str = f"[{stats.min_tokens}, {stats.max_tokens}]"
            
            lines.append(
                f"{method_name:<25} "
                f"{stats.avg_tokens:>12.1f} "
                f"{range_str:>20} "
                f"{stats.std_tokens:>10.1f} "
                f"{utilization:>7.1f}%"
            )
        
        lines.append("="*80 + "\n")
        
        return "\n".join(lines)
    
    def save_stats(self, output_path: str):
        """
        儲存統計資料到JSON
        
        Args:
            output_path: 輸出檔案路徑
        """
        data = {
            "baseline_tokens": self.baseline_tokens,
            "buffer_ratio": self.buffer_ratio,
            "target_tokens": self.get_target_tokens() if self.baseline_tokens else None,
            "methods": {}
        }
        
        for method_name, stats in self.stats.items():
            data["methods"][method_name] = {
                "avg_tokens": stats.avg_tokens,
                "min_tokens": stats.min_tokens,
                "max_tokens": stats.max_tokens,
                "std_tokens": stats.std_tokens,
                "sample_count": stats.sample_count,
                "utilization": self.calculate_utilization(method_name),
                "tokens_per_query": stats.tokens_per_query
            }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Token統計已儲存到: {output_path}")
    
    @classmethod
    def load_stats(cls, input_path: str) -> 'TokenBudgetController':
        """
        從JSON載入統計資料
        
        Args:
            input_path: 輸入檔案路徑
        
        Returns:
            TokenBudgetController實例
        """
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        controller = cls(
            baseline_tokens=data.get("baseline_tokens"),
            buffer_ratio=data.get("buffer_ratio", 0.1)
        )
        
        for method_name, method_data in data.get("methods", {}).items():
            stats = TokenBudgetStats(
                method_name=method_name,
                avg_tokens=method_data["avg_tokens"],
                min_tokens=method_data["min_tokens"],
                max_tokens=method_data["max_tokens"],
                std_tokens=method_data["std_tokens"],
                sample_count=method_data["sample_count"],
                tokens_per_query=method_data.get("tokens_per_query", [])
            )
            controller.stats[method_name] = stats
        
        print(f"📖 Token統計已載入自: {input_path}")
        return controller
