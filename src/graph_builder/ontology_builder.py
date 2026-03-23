# graph_builder/ontology_builder.py
import json
import yaml
import math
from tqdm import tqdm
from typing import List, Dict
from llama_index.core import Document, PropertyGraphIndex, PromptTemplate
from llama_index.core.schema import TransformComponent
from llama_index.core.graph_stores import EntityNode, Relation
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

from .base_builder import BaseGraphBuilder
from src.rag.schema.evolution import evolve_schema_with_pydantic, DynamicKGSchema

class CustomMetadataExtractor(TransformComponent):
    # (此處為您原有的 metadata 抽取邏輯，為節省篇幅省略細節，請直接貼上您原本的實作)
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            if "kg_nodes" not in node.metadata:
                node.metadata["kg_nodes"] = []
            if "kg_relations" not in node.metadata:
                node.metadata["kg_relations"] = []
            
            record_id = node.metadata.get("NO", "Unknown")
            customer = node.metadata.get("Customer")
            engineer = node.metadata.get("Engineers")
            # m_category = node.metadata.get("維護類別")
            
            record_node = EntityNode(
                name=f"Record_{record_id}", 
                label="Record", 
                properties={
                    "id": record_id,
                    "service_start": node.metadata.get("Service Start"),
                    "service_end": node.metadata.get("Service End")
                }
            )
            node.metadata["kg_nodes"].append(record_node)

            if customer:
                customer_node = EntityNode(name=customer, label="Customer")
                rel = Relation(source_id=record_node.id, target_id=customer_node.id, label="BELONGS_TO")
                node.metadata["kg_nodes"].append(customer_node)
                node.metadata["kg_relations"].append(rel)
            
            if engineer:
                engineer_node = EntityNode(name=engineer, label="Engineer")
                rel = Relation(source_id=engineer_node.id, target_id=record_node.id, label="HANDLED")
                node.metadata["kg_nodes"].append(engineer_node)
                node.metadata["kg_relations"].append(rel)

        return nodes

class OntologyGraphBuilder(BaseGraphBuilder):
    """
    Ontology Graph Builder
    
    使用動態 Schema 演化的本體學習建圖
    支援持久化與增量更新
    """
    
    def __init__(
        self,
        graph_store = None,
        settings = None,
        data_type: str = "DI",
        fast_test: bool = False,
        output_dir: str = None
    ):
        """
        初始化 Ontology Builder
        
        Args:
            graph_store: 圖譜儲存後端(可選)
            settings: 配置設定(可選)
            data_type: 資料類型
            fast_test: 是否為快速測試模式
            output_dir: 輸出目錄(用於儲存 Schema)
        """
        super().__init__(graph_store, settings)
        self.data_type = data_type
        self.fast_test = fast_test
        self.output_dir = output_dir or "./ontology_output"
        self.storage_path = None
        self.graph_index = None
    
    def get_name(self) -> str:
        return "Ontology"

    def build(self, documents: List[Document]):
        """
        使用動態 Schema 演化建立知識圖譜
        
        Args:
            documents: 文檔列表
            
        Returns:
            標準化的圖譜資料字典
        """
        import os
        from llama_index.core import StorageContext, load_index_from_storage
        from src.storage import get_storage_path
        
        # 取得儲存路徑
        self.storage_path = get_storage_path(
            storage_type="graph_index",
            data_type=self.data_type,
            method="ontology",
            fast_test=self.fast_test
        )
        
        print(f"📂 Ontology 儲存路徑: {self.storage_path}")
        
        # 確保輸出目錄存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Schema 檔案路徑
        schema_file = os.path.join(self.output_dir, "kg_schema_final.json")
        
        # 檢查是否已存在 cache
        if os.path.exists(self.storage_path) and os.path.exists(schema_file):
            print(f"✅ Ontology 索引已存在,載入現有索引和 Schema...")
            
            # 載入圖譜索引
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_path)
            self.graph_index = load_index_from_storage(storage_context)
            
            # 載入 Schema
            with open(schema_file, "r", encoding="utf-8") as f:
                dynamic_schema = json.load(f)
            
            print(f"✅ 已載入 Schema: {len(dynamic_schema.get('entities', []))} 實體類型, {len(dynamic_schema.get('relations', []))} 關係類型")
            
            schema_info = {
                "entities": dynamic_schema.get("entities", []),
                "relations": dynamic_schema.get("relations", []),
                "method": "ontology",
                "validation_schema": dynamic_schema.get("validation_schema", {}),
                "note": "動態演化 Schema"
            }
            
            graphml_path = os.path.join(self.storage_path, "graph_output.graphml")
            if not os.path.exists(graphml_path):
                graphml_path = None

            return {
                "nodes": [],
                "edges": [],
                "metadata": {
                    "num_documents": len(documents),
                    "fast_test": self.fast_test,
                    "cached": True
                },
                "schema_info": schema_info,
                "storage_path": self.storage_path,
                "graph_format": "property_graph",
                "graphml_path": graphml_path,
            }
        
        # 若無 cache，執行建圖
        print(f"🔨 [Ontology] 開始建立知識圖譜...")
        
        # 1. 讀取外部設定
        with open("kg_schema.json", "r", encoding="utf-8") as f:
            dynamic_schema = json.load(f)
        with open("kg_prompts.yml", "r", encoding="utf-8") as f:
            prompts_config = yaml.safe_load(f)
            
        prompt_template_str = prompts_config["ontology_learning"]["system_prompt"]
        
        # 2. 參數計算
        total_docs = len(documents)
        if self.fast_test and total_docs > 2:
            documents = documents[:2]
            total_docs = 2
            print(f"⚡ 快速測試模式:僅處理前 2 筆文檔")
        
        BATCH_SIZE = max(1, int(total_docs * 0.1))
        total_rounds = math.ceil(total_docs / BATCH_SIZE)
        MAX_EVOLVE_ROUNDS = max(1, int(total_rounds * 0.5))

        print(f"[Ontology Plugin] 啟動。預計處理 {total_rounds} 輪，前 {MAX_EVOLVE_ROUNDS} 輪動態演化 Schema。")
        
        # 3. 批次處理
        pbar = tqdm(range(0, total_docs, BATCH_SIZE), desc="Processing Batches")
        for i in pbar:
            batch = documents[i : i + BATCH_SIZE]
            round_num = (i // BATCH_SIZE) + 1
            pbar.set_postfix({"Round": round_num, "Schema_Size": len(dynamic_schema["entities"])})
            
            # Schema 演化（使用 src.rag.schema.evolution 統一入口）
            if round_num <= MAX_EVOLVE_ROUNDS:
                dynamic_schema = evolve_schema_with_pydantic(dynamic_schema, batch, self.settings.llm)
                
            # 建立 Extractor
            llm_extractor = SchemaLLMPathExtractor(
                llm=self.settings.llm,
                possible_entities=dynamic_schema["entities"],
                possible_relations=dynamic_schema["relations"],
                kg_validation_schema=dynamic_schema["validation_schema"],
                strict=False
            )
            
            # 寫入圖譜
            if self.graph_index is None:
                self.graph_index = PropertyGraphIndex.from_documents(
                    batch,
                    property_graph_store=self.graph_store,
                    kg_extractors=[CustomMetadataExtractor(), llm_extractor],
                    llm=self.settings.llm,
                    embed_model=self.settings.embed_model,
                    show_progress=False 
                )
            else:
                # 增量更新(如果支援)
                for doc in batch:
                    self.graph_index.insert(doc)
        
        # 4. 持久化
        if self.graph_index:
            self.graph_index.storage_context.persist(persist_dir=self.storage_path)
            print(f"💾 圖譜已持久化到: {self.storage_path}")
        
        # 5. 儲存最終 Schema
        with open(schema_file, "w", encoding="utf-8") as f:
            json.dump(dynamic_schema, f, ensure_ascii=False, indent=4)
        print(f"💾 最終 Schema 已儲存到: {schema_file}")
        print("\n✅ [Ontology Plugin] 建圖完成！")
        
        schema_info = {
            "entities": dynamic_schema.get("entities", []),
            "relations": dynamic_schema.get("relations", []),
            "method": "ontology",
            "validation_schema": dynamic_schema.get("validation_schema", {}),
            "note": "動態演化 Schema"
        }
        
        # 嘗試匯出 GraphML
        graphml_path = None
        try:
            from src.graph_adapter.base_adapter import GraphFormatAdapter
            G = GraphFormatAdapter.to_networkx(
                {"graph_index": self.graph_index, "graph_format": "property_graph"},
                source_format="property_graph"
            )
            if G.number_of_nodes() > 0:
                graphml_path = os.path.join(self.storage_path, "graph_output.graphml")
                GraphFormatAdapter.save_graphml(G, graphml_path)
        except Exception as e:
            print(f"⚠️  GraphML 匯出失敗: {e}")

        return {
            "nodes": [],
            "edges": [],
            "metadata": {
                "num_documents": len(documents),
                "fast_test": self.fast_test,
                "cached": False
            },
            "schema_info": schema_info,
            "storage_path": self.storage_path,
            "graph_format": "property_graph",
            "graphml_path": graphml_path,
        }