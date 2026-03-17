import json
from typing import List
from llama_index.core import Document
from model_settings import get_settings
import os
def data_processing(mode: str = "natural_text", data_type: str = "DI") -> List[Document]:
    """
    處理原始資料並轉換為 LlamaIndex Document 物件
    :param mode: 資料模式，可選 "key_value_text", "natural_text", "unstructured_text"
    """
    documents = []
    # 假設這是你的設定檔路徑
    Settings = get_settings()
    if data_type == "DI":
        raw_file_path = Settings.raw_file_path_DI
    elif data_type == "GEN":
        raw_file_path = Settings.raw_file_path_GEN
    else:
        print("Error: Invalid data type")
        return None
    
    with open(raw_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            # 共用的 metadata (兩邊都會用到)
            doc_metadata = {}
            for key in ["NO", "Customer", "Engineers", "Service Start", "Service End"]: # , "維護類別"
                if key in data:
                    doc_metadata[key] = data[key]
            # print(doc_metadata)

            # 根據不同模式，組合不同的 text 內容
            if mode == "unstructured_text":
                # Graph 模式：帶有明確邊界與防呆指令，專給 LLM 抽取器看
                formatted_text = (
                    "=== BACKGROUND CONTEXT (DO NOT EXTRACT ENTITIES FROM THIS SECTION) ===\n"
                    f"Record NO: {data['NO']}\n"
                    f"Customer: {data['Customer']}\n"
                    f"Engineer: {data['Engineers']}\n"
                    # f"Maintenance Type: {data.get('維護類別', None)} \n"
                    "======================================================================\n\n"
                    "=== TARGET TEXT FOR EXTRACTION (ONLY EXTRACT ENTITIES AND RELATIONS FROM HERE) ===\n"
                    f"Description: {data['Description']}\n"
                    f"Action: {data['Action']}\n"
                    "=================================================================================="
                )
                doc = Document(
                    text=formatted_text,
                    metadata=doc_metadata,
                    # Graph 模式下，避免底層再度將 metadata 拼接到 text 前面造成重複
                    excluded_llm_metadata_keys=list(doc_metadata.keys())
                )
                
            if mode == "markdown":
                # 精進版 Markdown 模式：利用 Markdown 原生層級與標題提示，完美區分「背景資訊」與「抽取目標」
                formatted_text = (
                    f"# Maintenance Record (維護紀錄)\n\n"
                    
                    # 使用 ## Metadata 標籤，並在標題旁加上(Context Only)暗示LLM這是背景
                    f"## Metadata (Context Only)\n"
                    f"- **Record NO**: {data.get('NO', 'N/A')}\n"
                    f"- **Customer**: {data.get('Customer', 'N/A')}\n"
                    f"- **Engineers**: {data.get('Engineers', 'N/A')}\n"
                    f"- **Service Time**: {data.get('Service Start', 'N/A')} to {data.get('Service End', 'N/A')}\n"
                    f"- **Maintenance Type**: {data.get('維護類別', 'N/A')}\n\n"
                    
                    # 使用 ## Target Details 標籤，明確指示抽取範圍
                    f"## Target Details (For Entity and Relation Extraction)\n"
                    f"### Description (狀態描述)\n"
                    f"{data.get('Description', '無描述')}\n\n"
                    
                    f"### Action (處置動作)\n"
                    f"{data.get('Action', '無動作')}\n"
                )
                doc = Document(
                    text=formatted_text,
                    metadata=doc_metadata,
                    # 由於已經在文本內透過 ## Metadata 完整呈現了，這裡設定排除，
                    # 避免 LlamaIndex 底層把 metadata 再度以 "key: value" 的形式硬塞在文本最前面，造成重複與混亂
                    excluded_llm_metadata_keys=list(doc_metadata.keys()),
                    excluded_embed_metadata_keys=list(doc_metadata.keys())
                )
            
            elif mode == "natural_text":
                # Vector 模式預設行為：乾淨的自然語意描述，沒有指令雜訊，最適合 Embedding
                # 將 metadata 自然地融入句子或以乾淨的 Key-Value 呈現
                formatted_text = (
                    f"這是一筆關於 {data['Customer']} 的維護紀錄。\n處理工程師為 {data['Engineers']}\n"
                    f"進行時間為 {data['Service Start']} 至 {data['Service End']}。\n"
                    f"狀態描述 (Description): {data['Description']}\n"
                    f"處置動作 (Action): {data['Action']}"
                )
                doc = Document(
                    text=formatted_text,
                    metadata=doc_metadata,
                    excluded_embed_metadata_keys=list(doc_metadata.keys()),
                    # 【新增設定】同樣排除讓 LLM 看到重複的 metadata（保持 Prompt 乾淨）
                    excluded_llm_metadata_keys=list(doc_metadata.keys())
                )
            elif mode == "key_value_text":
                formatted_text = (
                    f"NO: {data['NO']}\nCustomer: {data['Customer']}\nEngineers: {data['Engineers']}\nStart Time: {data['Service Start']}\nEnd Time: {data['Service End']}\nMaintenance Type: {data.get('維護類別', None)}\nDescription: {data['Description']}\nAction: {data['Action']}"
                )
                doc = Document(
                # text=f"Description: {data['Description']}\nAction: {data['Action']}",
                text=formatted_text,
                metadata=doc_metadata
                )
            else:
                print("Error: Invalid mode")
                return None
            documents.append(doc)
            
    return documents

if __name__ == "__main__":
    mode = "natural_text"
    data_type = "DI"
    documents = data_processing(mode=mode, data_type=data_type)
    if documents is not None:
        if not os.path.exists(f"/home/End_to_End_RAG/Data/{mode}_{data_type}"):
            os.makedirs(f"/home/End_to_End_RAG/Data/{mode}_{data_type}", exist_ok=True)
        with open(f"/home/End_to_End_RAG/Data/{mode}_{data_type}/documents.jsonl", "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(json.dumps(doc.model_dump(), ensure_ascii=False) + "\n")
        print(f"/home/End_to_End_RAG/Data/{mode}_{data_type}/documents.jsonl 已保存")
    else:
        print("Error: No documents to save")