# graph_builder/baseline_builder.py
from typing import List
from llama_index.core import Document, PropertyGraphIndex
from .base_builder import BaseGraphBuilder

class BaselineGraphBuilder(BaseGraphBuilder):
    def build(self, documents: List[Document]):
        print("[Baseline Plugin] 啟動 LlamaIndex 預設建圖邏輯...")
        PropertyGraphIndex.from_documents(
            documents,
            property_graph_store=self.graph_store,
            llm=self.settings.llm,
            embed_model=self.settings.embed_model,
            show_progress=True
        )
        print(">> [Baseline Plugin] 建圖完成！")