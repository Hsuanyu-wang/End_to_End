from kg_gen import KGGen
import json
from tqdm import tqdm

# Initialize KGGen with Ollama configuration
# 如果套件支援 litellm 格式，可以使用 model="ollama/llama3"
# 由於已經設定了 OPENAI_API_BASE，這裡直接填寫模型名稱也可以
kg = KGGen(
    model="openai/qwen2.5-coder:7b",       # 請替換成你在 Ollama 中下載的模型，例如 "llama3", "mistral", "gemma2"
    temperature=0.0,      # 知識圖譜生成建議保持 0.0 以降低幻覺
    api_key="ollama",      # 填入任意字串即可，Ollama 本地端不需要真實的 API Key
    api_base="http://192.168.63.184:11434/v1"
)

documents_texts = []
file_path = '/home/End_to_End_RAG/Data/natural_text/documents.jsonl'
with open(file_path, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

with open(file_path, 'r', encoding='utf-8') as f:
    # 使用 tqdm 包裝迭代器
    for line in tqdm(f, total=total_lines, desc="解析 JSONL"):
        if not line.strip(): 
            continue
        data = json.loads(line)
        text = data.get('text_resource', {}).get('text', '')
        if text:
            documents_texts.append(text)

# 將所有紀錄串接成一個大字串，並使用分隔線隔開每筆紀錄
combined_documents_text = "\n\n---\n\n".join(documents_texts)

# # ==========================================
# # EXAMPLE 1: Single string with context
# # ==========================================
# graph_1 = kg.generate(
#     input_data=combined_documents_text,
#     context="Family relationships",
#     output_folder="/home/End_to_End_RAG/method/kggen_output"
# )
# print("--- Graph 1 ---")
# print("Entities:", graph_1.entities)
# print("Relations:", graph_1.relations)


# ==========================================
# EXAMPLE 2: Large text with chunking and clustering
# ==========================================
all_graphs = []
print("開始生成知識圖譜 (Chunking Mode)...")

for doc in tqdm(documents_texts, desc="生成進度"):
    # 這裡可以逐筆 generate，最後再 aggregate
    g = kg.generate(input_data=doc, context="這是一份關於 IT 系統維護紀錄、Splunk 軟體保養與客戶工程師對應關係的知識圖譜")
    all_graphs.append(g)
    
# 最後合併並聚類
print("正在合併圖譜並進行實體聚類 (Clustering)...")
final_graph = kg.aggregate(all_graphs)
graph_2 = kg.cluster(final_graph, context="這是一份關於 IT 系統維護紀錄、Splunk 軟體保養與客戶工程師對應關係的知識圖譜")
    
# print("\n--- Graph 2 ---")
print("\n--- 知識圖譜生成完成 ---")
print(f"總共找到 {len(graph_2.entities)} 個實體 (Entities)\n")
print("Entities:", graph_2.entities)
print(f"總共找到 {len(graph_2.relations)} 筆關係 (Relations)\n")
print("Relations:", graph_2.relations)
if hasattr(graph_2, 'entity_clusters'):
    print("Entity Clusters:", graph_2.entity_clusters)

# # ==========================================
# # EXAMPLE 3: Messages array
# # ==========================================
# graph_3 = kg.generate(input_data=combined_documents_text, output_folder="/home/End_to_End_RAG/method/kggen_output")
# print("\n--- Graph 3 ---")
# print("Entities:", graph_3.entities)
# print("Relations:", graph_3.relations)


# # ==========================================
# # EXAMPLE 4: Combining multiple graphs
# # ==========================================
# text1 = "Linda is Joe's mother. Ben is Joe's brother."
# text2 = "Andrew is Joseph's father. Judy is Andrew's sister. Joseph also goes by Joe."

# graph4_a = kg.generate(input_data=text1,output_folder="/home/End_to_End_RAG/method/kggen_output")
# graph4_b = kg.generate(input_data=text2,output_folder="/home/End_to_End_RAG/method/kggen_output")

# # Combine the graphs
# combined_graph = kg.aggregate([graph4_a, graph4_b])

# # Optionally cluster the combined graph
# clustered_graph = kg.cluster(
#     combined_graph,
#     context="Family relationships",
# )
# print("\n--- Graph 4 (Clustered) ---")
# print("Entities:", clustered_graph.entities)
# print("Relations:", clustered_graph.relations)