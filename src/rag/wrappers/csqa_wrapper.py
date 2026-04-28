# src/rag/wrappers/csqa_wrapper.py
from src.rag.wrappers.base_wrapper import BaseRAGWrapper
from src.graph_builder.csqa_builder import CSQAGraphBuilder
from src.graph_retriever.csqa_retriever import CSQARetriever

class CSQAWrapper(BaseRAGWrapper):
    def __init__(self, config):
        self.builder = CSQAGraphBuilder(config.llm, config.vector_db, config.graph_db)
        self.retriever = CSQARetriever(config.llm, config.vector_db, config.graph_db)
        
    def insert(self, documents):
        self.builder.build_graph(documents)
        
    def query(self, text: str) -> str:
        return self.retriever.retrieve(text)