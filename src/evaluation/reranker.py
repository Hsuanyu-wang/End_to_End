"""
Cross-Encoder Re-ranking 模組

使用 Cross-Encoder 模型對檢索結果進行重排序，
提升相關性排序的準確度
"""

from typing import List, Tuple
import numpy as np


class CrossEncoderReranker:
    """Cross-Encoder Re-ranker"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        初始化 Cross-Encoder Re-ranker
        
        Args:
            model_name: Cross-Encoder 模型名稱
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """載入 Cross-Encoder 模型"""
        try:
            from sentence_transformers import CrossEncoder
            print(f"🔄 載入 Cross-Encoder 模型: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            print("✅ Cross-Encoder 模型載入完成")
        except ImportError:
            print("⚠️  sentence-transformers 未安裝，Re-ranker 將使用簡化版本")
            self.model = None
        except Exception as e:
            print(f"⚠️  載入 Cross-Encoder 模型失敗: {e}")
            self.model = None
            
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        使用 Cross-Encoder 對文檔進行重排序
        
        Args:
            query: 查詢字串
            documents: 文檔列表
            top_k: 返回的 top-k 結果
            
        Returns:
            [(index, score), ...] 排序後的索引和分數列表
        """
        if not documents:
            return []
            
        if self.model is None:
            # Fallback: 使用簡單的詞頻匹配
            return self._simple_rerank(query, documents, top_k)
        
        try:
            # 計算 query-document pairs 的相關性分數
            pairs = [(query, doc) for doc in documents]
            scores = self.model.predict(pairs)
            
            # 排序並返回 top-k
            ranked_indices = np.argsort(scores)[::-1][:top_k]
            return [(int(idx), float(scores[idx])) for idx in ranked_indices]
            
        except Exception as e:
            print(f"⚠️  Re-ranking 失敗: {e}，使用簡化版本")
            return self._simple_rerank(query, documents, top_k)
    
    def _simple_rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        簡化版本的 re-ranking（基於詞頻匹配）
        
        Args:
            query: 查詢字串
            documents: 文檔列表
            top_k: 返回的 top-k 結果
            
        Returns:
            [(index, score), ...] 排序後的索引和分數列表
        """
        query_terms = set(query.lower().split())
        
        scores = []
        for idx, doc in enumerate(documents):
            doc_terms = set(doc.lower().split())
            # 計算 Jaccard 相似度
            intersection = len(query_terms & doc_terms)
            union = len(query_terms | doc_terms)
            score = intersection / union if union > 0 else 0.0
            scores.append((idx, score))
        
        # 按分數排序
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def batch_rerank(
        self, 
        queries: List[str], 
        documents_list: List[List[str]], 
        top_k: int = 10
    ) -> List[List[Tuple[int, float]]]:
        """
        批次 re-ranking
        
        Args:
            queries: 查詢列表
            documents_list: 文檔列表的列表
            top_k: 返回的 top-k 結果
            
        Returns:
            每個查詢的排序結果列表
        """
        results = []
        for query, documents in zip(queries, documents_list):
            results.append(self.rerank(query, documents, top_k))
        return results
