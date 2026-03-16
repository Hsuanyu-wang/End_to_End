# graph_builder/ontology_builder.py
import json
import yaml
import math
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing import List, Dict
from llama_index.core import Document, PropertyGraphIndex, PromptTemplate
from llama_index.core.schema import TransformComponent
from llama_index.core.graph_stores import EntityNode, Relation
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

from .base_builder import BaseGraphBuilder

# 定義結構
class DynamicKGSchema(BaseModel):
    entities: List[str] = Field(description="實體類型列表")
    relations: List[str] = Field(description="關係類型列表")
    validation_schema: Dict[str, List[str]] = Field(description="實體與關係的合法連接規則")

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
    
    def evolve_schema_with_pydantic(self, current_schema: dict, batch_docs: list, prompt_template_str: str) -> dict:
        sample_text = "\n\n".join([doc.text[:] for doc in batch_docs])
        prompt_str = prompt_template_str.format(
            current_schema=json.dumps(current_schema, ensure_ascii=False), 
            sample_text=sample_text
        )
        prompt_template = PromptTemplate(template=prompt_str)
        try:
            response_obj = self.settings.llm.structured_predict(
                DynamicKGSchema,
                prompt=prompt_template,
            )
            return response_obj.model_dump()
        except Exception as e:
            print(f">> Schema 更新失敗: {e}")
            return current_schema

    def build(self, documents: List[Document]):
        # 1. 讀取外部設定
        with open("kg_schema.json", "r", encoding="utf-8") as f:
            dynamic_schema = json.load(f)
        with open("kg_prompts.yml", "r", encoding="utf-8") as f:
            prompts_config = yaml.safe_load(f)
            
        prompt_template_str = prompts_config["ontology_learning"]["system_prompt"]
        
        # 2. 參數計算
        total_docs = len(documents)
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
            
            # Schema 演化
            if round_num <= MAX_EVOLVE_ROUNDS:
                dynamic_schema = self.evolve_schema_with_pydantic(dynamic_schema, batch, prompt_template_str)
                
            # 建立 Extractor
            llm_extractor = SchemaLLMPathExtractor(
                llm=self.settings.llm,
                possible_entities=dynamic_schema["entities"],
                possible_relations=dynamic_schema["relations"],
                kg_validation_schema=dynamic_schema["validation_schema"],
                strict=False
            )
            
            # 寫入圖譜
            PropertyGraphIndex.from_documents(
                batch,
                property_graph_store=self.graph_store,
                kg_extractors=[CustomMetadataExtractor(), llm_extractor],
                llm=self.settings.llm,
                embed_model=self.settings.embed_model,
                show_progress=False 
            )
        
        # 4. 儲存結果
        with open("kg_schema_final.json", "w", encoding="utf-8") as f:
            json.dump(dynamic_schema, f, ensure_ascii=False, indent=4)
        print("\n>> [Ontology Plugin] 建圖完成！最終 Schema 已儲存。")