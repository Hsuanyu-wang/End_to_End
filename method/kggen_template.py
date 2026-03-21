from kg_gen import KGGen

# Initialize KGGen with Ollama configuration
# 如果套件支援 litellm 格式，可以使用 model="ollama/llama3"
# 由於已經設定了 OPENAI_API_BASE，這裡直接填寫模型名稱也可以
kg = KGGen(
    model="openai/qwen2.5:7b",       # 請替換成你在 Ollama 中下載的模型，例如 "llama3", "mistral", "gemma2"
    temperature=0.0,      # 知識圖譜生成建議保持 0.0 以降低幻覺
    api_key="ollama",      # 填入任意字串即可，Ollama 本地端不需要真實的 API Key
    api_base="http://192.168.63.184:11434/v1"
)

# ==========================================
# EXAMPLE 1: Single string with context
# ==========================================
text_input = "Linda is Josh's mother. Ben is Josh's brother. Andrew is Josh's father."
graph_1 = kg.generate(
    input_data=text_input,
    context="Family relationships",
    output_folder="/home/End_to_End_RAG/method/kggen_output"
)
print("--- Graph 1 ---")
print("Entities:", graph_1.entities)
print("Relations:", graph_1.relations)


# ==========================================
# EXAMPLE 2: Large text with chunking and clustering
# ==========================================
# 先建立一個範例 large_text.txt 供測試
sample_text = """
Neural networks are a type of machine learning model. Deep learning is a subset of machine learning
that uses multiple layers of neural networks. Supervised learning requires training data to learn
patterns. Machine learning is a type of AI technology that enables computers to learn from data.
AI, also known as artificial intelligence, is related to the broader field of artificial intelligence.
Neural nets (NN) are commonly used in ML applications. Machine learning (ML) has revolutionized
many fields of study.
"""
with open('large_text.txt', 'w') as f:
    f.write(sample_text)

with open('large_text.txt', 'r') as f:
    large_text = f.read()

graph_2 = kg.generate(
    input_data=large_text,
    chunk_size=5000,  # Process text in chunks of 5000 chars
    cluster=True,      # Cluster similar entities and relations
    output_folder="/home/End_to_End_RAG/method/kggen_output"
)
print("\n--- Graph 2 ---")
print("Entities:", graph_2.entities)
print("Relations:", graph_2.relations)
if hasattr(graph_2, 'entity_clusters'):
    print("Entity Clusters:", graph_2.entity_clusters)


# ==========================================
# EXAMPLE 3: Messages array
# ==========================================
messages = [
    {"role": "user", "content": "What is the capital of France?"}, 
    {"role": "assistant", "content": "The capital of France is Paris."}
]
graph_3 = kg.generate(input_data=messages, output_folder="/home/End_to_End_RAG/method/kggen_output")
print("\n--- Graph 3 ---")
print("Entities:", graph_3.entities)
print("Relations:", graph_3.relations)


# ==========================================
# EXAMPLE 4: Combining multiple graphs
# ==========================================
text1 = "Linda is Joe's mother. Ben is Joe's brother."
text2 = "Andrew is Joseph's father. Judy is Andrew's sister. Joseph also goes by Joe."

graph4_a = kg.generate(input_data=text1,output_folder="/home/End_to_End_RAG/method/kggen_output")
graph4_b = kg.generate(input_data=text2,output_folder="/home/End_to_End_RAG/method/kggen_output")

# Combine the graphs
combined_graph = kg.aggregate([graph4_a, graph4_b])

# Optionally cluster the combined graph
clustered_graph = kg.cluster(
    combined_graph,
    context="Family relationships",
)
print("\n--- Graph 4 (Clustered) ---")
print("Entities:", clustered_graph.entities)
print("Relations:", clustered_graph.relations)