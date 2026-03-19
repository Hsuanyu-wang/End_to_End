"""
End-to-End RAG 評估框架主套件

提供完整的 RAG Pipeline 評估功能
"""

__version__ = "1.0.0"
__author__ = "End-to-End RAG Team"

from src.config import get_settings, ModelSettings, my_settings, DataConfig, LightRAGConfig
from src.data import data_processing, DataProcessor, QADataLoader
from src.evaluation import RAGEvaluator, EvaluationReporter, run_evaluation
from src.rag.wrappers import (
    BaseRAGWrapper,
    VectorRAGWrapper,
    LightRAGWrapper,
    TemporalLightRAGWrapper,
)

__all__ = [
    "get_settings",
    "ModelSettings",
    "my_settings",
    "DataConfig",
    "LightRAGConfig",
    "data_processing",
    "DataProcessor",
    "QADataLoader",
    "RAGEvaluator",
    "EvaluationReporter",
    "run_evaluation",
    "BaseRAGWrapper",
    "VectorRAGWrapper",
    "LightRAGWrapper",
    "TemporalLightRAGWrapper",
]
