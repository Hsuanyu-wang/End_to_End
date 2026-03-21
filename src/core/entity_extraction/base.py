"""
實體抽取組件抽象基類

定義了實體抽取器的統一介面，所有實體抽取實作都應繼承此基類。
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

# 從頂層 formats 模組導入
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from formats import Entity, EntityList, Schema


class BaseEntityExtractor(ABC):
    """
    實體抽取器抽象基類
    
    所有實體抽取組件（LightRAG、規則式、NER模型等）都應實作此介面。
    
    關鍵設計原則：
    1. 輸入：純文本 + 可選的 Schema 約束
    2. 輸出：標準 Entity 物件列表
    3. 支援批次處理
    
    Examples:
        >>> class LightRAGEntityExtractor(BaseEntityExtractor):
        ...     def extract(self, text, schema=None):
        ...         # LightRAG 的實體抽取邏輯
        ...         return EntityList([...])
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化實體抽取器
        
        Args:
            config: 配置字典，可包含模型參數、API設定等
        """
        self.config = config or {}
        self._initialize()
    
    def _initialize(self):
        """子類可覆寫此方法進行初始化"""
        pass
    
    @abstractmethod
    def extract(
        self, 
        text: str, 
        schema: Optional[Schema] = None,
        **kwargs
    ) -> EntityList:
        """
        從文本中抽取實體
        
        這是核心抽象方法，所有子類必須實作。
        
        Args:
            text: 輸入文本
            schema: 可選的 Schema 約束（用於指導抽取）
            **kwargs: 額外參數（如 top_k、confidence_threshold 等）
        
        Returns:
            EntityList: 抽取到的實體列表
        
        Raises:
            ValueError: 當輸入無效時
            RuntimeError: 當抽取過程失敗時
        """
        pass
    
    def extract_batch(
        self,
        texts: List[str],
        schema: Optional[Schema] = None,
        **kwargs
    ) -> List[EntityList]:
        """
        批次抽取實體
        
        預設實作為逐一處理，子類可覆寫以提供更高效的批次處理。
        
        Args:
            texts: 文本列表
            schema: 可選的 Schema 約束
            **kwargs: 額外參數
        
        Returns:
            List[EntityList]: 每個文本對應的實體列表
        """
        results = []
        for text in texts:
            entities = self.extract(text, schema, **kwargs)
            results.append(entities)
        return results
    
    def get_name(self) -> str:
        """
        獲取抽取器名稱
        
        Returns:
            str: 抽取器名稱（如 "lightrag", "spacy_ner", "rule_based"）
        """
        return self.__class__.__name__
    
    def get_config(self) -> Dict[str, Any]:
        """獲取當前配置"""
        return self.config.copy()
    
    def validate_input(self, text: str) -> bool:
        """
        驗證輸入文本
        
        Args:
            text: 輸入文本
        
        Returns:
            bool: 是否有效
        """
        if not text or not isinstance(text, str):
            return False
        if len(text.strip()) == 0:
            return False
        return True
