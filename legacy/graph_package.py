# main_graph_pipeline.py
import os
import yaml
import argparse
import nest_asyncio
from dotenv import load_dotenv
from llama_index.core import PropertyGraphIndex

# 模組匯入
from model_settings import get_settings
from data_processing import data_processing
from graph_store_utils import SafeNeo4jStore
from graph_builder import get_graph_builder

nest_asyncio.apply()
load_dotenv()

class GraphPipelineManager:
    def __init__(self, Settings, config_path="/home/End_to_End_RAG/config.yml"):
        self.Settings = Settings
        self.config = yaml.safe_load(open(config_path))
        
        # 統一管理資料庫實例
        self.graph_store = SafeNeo4jStore(
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            url=os.getenv("NEO4J_URI"),
            database=self.config["graph"]["neo4j_database"]
        )

    def build_graph(self, method: str = "baseline", init_db: bool = False):
        """階段一：建圖"""
        if init_db:
            self.graph_store.delete_all()

        docs = data_processing(mode="unstructured_text")
        if not docs:
            raise ValueError("無文本可供建圖。")

        # 透過工廠模式取得對應的 Plugin 並執行
        builder = get_graph_builder(method, self.graph_store, self.Settings)
        builder.build(docs)

    def get_retriever_engine(self):
        """階段二：檢索"""
        graph_index = PropertyGraphIndex.from_existing(
            property_graph_store=self.graph_store,
            llm=self.Settings.llm,
            embed_model=self.Settings.embed_model
        )
        return graph_index.as_query_engine()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modular Knowledge Graph Pipeline")
    parser.add_argument("--init_neo4j", action="store_true", help="清空 Neo4j 舊資料")
    parser.add_argument("--mode", choices=["build", "retrieve", "end_to_end"], default="end_to_end")
    parser.add_argument("--build_method", choices=["baseline", "ontology_learning"], default="ontology_learning")
    args = parser.parse_args()

    Settings = get_settings()
    manager = GraphPipelineManager(Settings)

    if args.mode in ["build", "end_to_end"]:
        print("=== 階段一：Graph Construction ===")
        manager.build_graph(method=args.build_method, init_db=args.init_neo4j)

    if args.mode in ["retrieve", "end_to_end"]:
        print("\n=== 階段二：Graph Retrieval ===")
        engine = manager.get_retriever_engine()
        response = engine.query("C客戶現在的splunk版本是什麼")
        print(f"Response: {response}")