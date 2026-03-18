"""
Token計數工具

提供統一的token計數功能，支援多種編碼模型
"""

import tiktoken
from typing import List, Union


class TokenCounter:
    """
    Token計數器
    
    使用tiktoken進行準確的token計數，支援多種編碼模型
    
    Attributes:
        encoding: tiktoken編碼器實例
        model_name: 使用的模型名稱
    """
    
    def __init__(self, model_name: str = "gpt-4"):
        """
        初始化TokenCounter
        
        Args:
            model_name: 編碼模型名稱，預設為gpt-4
        """
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
            self.model_name = model_name
        except KeyError:
            # 如果模型不存在，使用cl100k_base作為fallback
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.model_name = "cl100k_base"
            print(f"⚠️  模型 {model_name} 不存在，使用 cl100k_base 編碼")
    
    def count_tokens(self, text: str) -> int:
        """
        計算單一文本的token數量
        
        Args:
            text: 輸入文本
        
        Returns:
            token數量
        """
        if not text:
            return 0
        
        try:
            return len(self.encoding.encode(str(text)))
        except Exception as e:
            print(f"⚠️  Token計數失敗: {e}")
            return 0
    
    def count_tokens_batch(self, texts: List[str]) -> int:
        """
        批次計算多個文本的總token數量
        
        Args:
            texts: 文本列表
        
        Returns:
            總token數量
        """
        if not texts:
            return 0
        
        return sum(self.count_tokens(text) for text in texts)
    
    def count_tokens_with_details(self, texts: List[str]) -> dict:
        """
        計算多個文本的token數量並返回詳細資訊
        
        Args:
            texts: 文本列表
        
        Returns:
            包含總數和每個文本token數的字典
        """
        if not texts:
            return {
                "total_tokens": 0,
                "num_texts": 0,
                "tokens_per_text": [],
                "avg_tokens_per_text": 0
            }
        
        tokens_per_text = [self.count_tokens(text) for text in texts]
        total = sum(tokens_per_text)
        
        return {
            "total_tokens": total,
            "num_texts": len(texts),
            "tokens_per_text": tokens_per_text,
            "avg_tokens_per_text": total / len(texts) if texts else 0,
            "min_tokens": min(tokens_per_text) if tokens_per_text else 0,
            "max_tokens": max(tokens_per_text) if tokens_per_text else 0
        }
    
    def truncate_text_by_tokens(self, text: str, max_tokens: int) -> str:
        """
        根據 token 數量截斷文本
        
        Args:
            text: 輸入文本
            max_tokens: 最大 token 數
        
        Returns:
            截斷後的文本
        """
        if not text or max_tokens <= 0:
            return ""
        
        try:
            tokens = self.encoding.encode(str(text))
            if len(tokens) <= max_tokens:
                return text
            
            # 截斷 tokens 並解碼回文本
            truncated_tokens = tokens[:max_tokens]
            truncated_text = self.encoding.decode(truncated_tokens)
            return truncated_text
        except Exception as e:
            print(f"⚠️  文本截斷失敗: {e}")
            # Fallback：簡單字符截斷（估算 1 token ≈ 4 字符）
            approx_chars = max_tokens * 4
            return text[:approx_chars]
    
    def estimate_cost(self, num_tokens: int, model_type: str = "gpt-4") -> float:
        """
        估算API成本（僅供參考）
        
        Args:
            num_tokens: token數量
            model_type: 模型類型
        
        Returns:
            估算成本（美元）
        """
        # 簡化的價格表（每1000 tokens）
        price_per_1k = {
            "gpt-4": 0.03,  # input tokens
            "gpt-3.5-turbo": 0.0015,
            "claude-3-sonnet": 0.003,
        }
        
        rate = price_per_1k.get(model_type, 0.03)
        return (num_tokens / 1000) * rate
    
    def __repr__(self) -> str:
        return f"TokenCounter(model={self.model_name})"


# 全局單例，避免重複初始化
_global_token_counter = None


def get_token_counter(model_name: str = "gpt-4") -> TokenCounter:
    """
    獲取全局TokenCounter單例
    
    Args:
        model_name: 編碼模型名稱
    
    Returns:
        TokenCounter實例
    """
    global _global_token_counter
    
    if _global_token_counter is None or _global_token_counter.model_name != model_name:
        _global_token_counter = TokenCounter(model_name)
    
    return _global_token_counter


# 便捷函數
def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """
    快速計算文本token數（便捷函數）
    
    Args:
        text: 輸入文本
        model_name: 編碼模型名稱
    
    Returns:
        token數量
    """
    counter = get_token_counter(model_name)
    return counter.count_tokens(text)


def count_tokens_batch(texts: List[str], model_name: str = "gpt-4") -> int:
    """
    快速批次計算token數（便捷函數）
    
    Args:
        texts: 文本列表
        model_name: 編碼模型名稱
    
    Returns:
        總token數量
    """
    counter = get_token_counter(model_name)
    return counter.count_tokens_batch(texts)
